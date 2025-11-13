from typing import Tuple, Union, List

import torch
import torch.nn as nn
from torch.distributions import Normal

from utils.model_utils import weights_init


class ActorNonLinear(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self,
                 state_size: Union[List[int], Tuple[int], int],
                 action_size: int,
                 seed: int = 0,
                 fc_units: List[int] = None):
        """
        :param state_size (int): Dimension of each state
        :param action_size (int): Dimension of each action
        :param seed (int): Random seed
        :param fc_units (int): Number of nodes in hidden layer
        """
        super(ActorNonLinear, self).__init__()

        if fc_units is None:
            fc_units = [128, 128]

        self.seed = torch.manual_seed(seed)
        self.state_norm = nn.BatchNorm1d(state_size)

        self.fc_units = fc_units
        self.fc_units.insert(0, state_size)
        self.fc_layers = nn.ModuleList([nn.Sequential(nn.Linear(fc_in, fc_out), nn.ReLU(inplace=False))
                                        for fc_in, fc_out in zip(self.fc_units[:-1], self.fc_units[1:])])

        self.fc_out = nn.Linear(self.fc_units[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        for layer_idx in range(len(self.fc_layers)):
            self.fc_layers[layer_idx].apply(weights_init)

        # self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_out.apply(weights_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = self.state_norm(state)
        for layer_idx in range(len(self.fc_layers)):
            x = self.fc_layers[layer_idx](x)

        return torch.tanh(self.fc_out(x))


class ActorNonDeterministic(nn.Module):
    def __init__(self,
                 state_size: Union[Tuple[int], int],
                 action_size: int,
                 max_action: Union[float, int],
                 device: torch.device,
                 seed: int = 0,
                 fc_units: List[int] = None):
        """
        Specially for SAC
        :param state_size (int): Dimension of each state
        :param action_size (int): Dimension of each action
        :param seed (int): Random seed
        :param fc1_units (int): Number of nodes in first hidden layer
        :param fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorNonDeterministic, self).__init__()

        if fc_units is None:
            fc_units = [128, 128]

        self.seed = torch.manual_seed(seed)
        self.max_action = max_action
        self.device = device

        self.state_norm = nn.BatchNorm1d(state_size)

        self.fc_units = fc_units
        self.fc_units.insert(0, state_size)
        self.fc_layers = nn.ModuleList([nn.Sequential(nn.Linear(fc_in, fc_out), nn.ReLU(inplace=False))
                                        for fc_in, fc_out in zip(self.fc_units[:-1], self.fc_units[1:])])

        self.mu = nn.Linear(self.fc_units[-1], action_size)
        self.sigma = nn.Linear(self.fc_units[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        for layer_idx in range(len(self.fc_layers)):
            self.fc_layers[layer_idx].apply(weights_init)

        self.mu.apply(weights_init)
        self.sigma.apply(weights_init)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = self.state_norm(state)
        for layer_idx in range(len(self.fc_layers)):
            x = self.fc_layers[layer_idx](x)

        mu = self.mu(x)
        sigma = self.sigma(x)

        sigma = torch.clamp(sigma, min=1e-6, max=1.0)

        return mu, sigma

    def sample_normal(self, state: torch.Tensor, reparameterize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            # Noisy sampling
            actions = probabilities.rsample()
        else:
            # Directly probabilistic sampling
            actions = probabilities.sample()

        # TODO why ?
        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)

        # Log probabilities to calculate the loss
        log_probabilities = probabilities.log_prob(actions)
        log_probabilities -= torch.log(1 - action.pow(2) + 1e-6)
        log_probabilities = log_probabilities.sum(1, keepdim=True)

        return action, log_probabilities


class Critic(nn.Module):
    """ Critic Model."""

    def __init__(self,
                 state_size: Union[Tuple[int], int],
                 action_size: int,
                 seed: int = 0,
                 fcs_units=None,
                 fc_units=None):
        """
        :param state_size (int): Dimension of each state
        :param action_size (int): Dimension of each action
        :param seed (int): Random seed
        :param fcs1_units (int): Number of nodes in the first hidden layer
        :param fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        if fc_units is None:
            fc_units = [128]
        if fcs_units is None:
            fcs_units = [128]

        self.seed = torch.manual_seed(seed)

        self.state_norm = nn.BatchNorm1d(state_size)

        # Fully connected layers for state-only
        self.fcs_units = fcs_units
        self.fcs_units.insert(0, state_size)
        self.fcs_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(fc_in, fc_out), nn.ReLU(inplace=False), nn.BatchNorm1d(fc_out))
             for fc_in, fc_out in zip(self.fcs_units[:-1], self.fcs_units[1:])])

        # Fully connected layers for state-action fused layers
        self.fc_units = fc_units
        self.fc_units.insert(0, self.fcs_units[-1] + action_size)
        self.fc_layers = nn.ModuleList([nn.Sequential(nn.Linear(fc_in, fc_out), nn.ReLU(inplace=False))
                                        for fc_in, fc_out in zip(self.fc_units[:-1], self.fc_units[1:])])

        self.fc_out = nn.Linear(self.fc_units[-1], 1)

        self.dropout = nn.Dropout(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        for layer_idx in range(len(self.fcs_layers)):
            self.fcs_layers[layer_idx].apply(weights_init)

        for layer_idx in range(len(self.fc_layers)):
            self.fc_layers[layer_idx].apply(weights_init)
        self.fc_out.apply(weights_init)
        # self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        xs = self.state_norm(state)

        # Process state only
        for layer_idx in range(len(self.fcs_layers)):
            xs = self.fcs_layers[layer_idx](xs)

        # Fuse action and state
        x = torch.cat((xs, action), dim=1)

        # Process Fused
        for layer_idx in range(len(self.fc_layers)):
            x = self.fc_layers[layer_idx](x)

        x = self.dropout(x)

        return self.fc_out(x)


class Value(nn.Module):
    """ Value Model. [SAC]"""

    def __init__(self, state_size: Union[Tuple[int], int],
                 seed: int = 0,
                 fcs_units: List[int] = None,
                 fc_units: List[int] = None):
        """
        :param state_size (int): Dimension of each state
        :param seed (int): Random seed
        :param fcs1_units (int): Number of nodes in the first hidden layer
        :param fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Value, self).__init__()
        if fc_units is None:
            fc_units = [128]
        if fcs_units is None:
            fcs_units = [128]

        self.seed = torch.manual_seed(seed)

        self.state_norm = nn.BatchNorm1d(state_size)

        # Fully connected layers for state-only
        self.fcs_units = fcs_units
        self.fcs_units.insert(0, state_size)
        self.fcs_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(fc_in, fc_out), nn.ReLU(inplace=False), nn.BatchNorm1d(fc_out))
             for fc_in, fc_out in zip(self.fcs_units[:-1], self.fcs_units[1:])])

        # Fully connected layers for state-action fused layers
        self.fc_units = fc_units
        self.fc_units.insert(0, self.fcs_units[-1])
        self.fc_layers = nn.ModuleList([nn.Sequential(nn.Linear(fc_in, fc_out), nn.ReLU(inplace=False))
                                        for fc_in, fc_out in zip(self.fc_units[:-1], self.fc_units[1:])])

        self.fc_out = nn.Linear(self.fc_units[-1], 1)

        self.dropout = nn.Dropout(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        for layer_idx in range(len(self.fcs_layers)):
            self.fcs_layers[layer_idx].apply(weights_init)

        for layer_idx in range(len(self.fc_layers)):
            self.fc_layers[layer_idx].apply(weights_init)
        self.fc_out.apply(weights_init)
        # self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        xs = self.state_norm(state)
        # Process state only
        for layer_idx in range(len(self.fcs_layers)):
            xs = self.fcs_layers[layer_idx](xs)

        # Process Fused
        for layer_idx in range(len(self.fc_layers)):
            xs = self.fc_layers[layer_idx](xs)

        x = self.dropout(xs)

        return self.fc_out(xs)


class ActorCNN(nn.Module):
    def __init__(self):
        super(ActorCNN, self).__init__()


class CriticCNN(nn.Module):
    def __init__(self):
        super(CriticCNN, self).__init__()


class ValueCNN(nn.Module):
    def __init__(self):
        super(ValueCNN, self).__init__()
