import os
import random
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import device

import utils.memory_modules as mem_modules
from model.networks import ActorNonDeterministic, Critic, Value
from utils.model_utils import soft_update
from utils.noise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACAgent(object):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size: Union[List[int], Tuple[int], int],
                 action_size: int,
                 random_seed: int,
                 max_action: Union[int, float],
                 config: Dict[str, Any]):
        """
        :param state_size (int): dimension of each state
        :param action_size (int): dimension of each action
        :param random_seed (int): random seed
        :param config (dict) : dictionary of hyper-parameters

            Example Config :
            {
                LR_ACTOR : 1e-4,
                LR_CRITIC : 1e-4,
                CRITIC_COUNT : 2, # TD3
                WEIGHT_DECAY : 1e-6,
                UPDATE_EVERY : 2,
                POLICY_DELAY : 2,
                BUFFER_SIZE : 1e6,
                BATCH_SIZE : 128,
                GAMMA : 0.99, # Discount Rate
                TAU : 1e-3 # Target Model Update Rate
                ACTION_MIN : -1.0,
                ACTION_MAX : 1.0,
                SCALE : 2.0
                CLIP_ACTOR_GRADIENTS : false,
                CLIP_CRITIC_GRADIENTS : true,
                MEMORY_MODULE : NaivePrioritizedBuffer # ReplayBuffer
            }
        """
        random.seed(random_seed)

        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Actor Network
        self.actor_network = ActorNonDeterministic(state_size=state_size,
                                                   action_size=action_size,
                                                   max_action=max_action,
                                                   device=device,
                                                   seed=random_seed,
                                                   fc_units=[128, 128]).to(device)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.config["LR_ACTOR"])

        # Critic Networks
        self.critic_network_1 = Critic(state_size, action_size, random_seed, fcs_units=[128], fc_units=[128]).to(device)
        self.critic_network_2 = Critic(state_size, action_size, random_seed, fcs_units=[128], fc_units=[128]).to(device)

        self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(),
                                             lr=self.config["LR_CRITIC"],
                                             weight_decay=self.config["WEIGHT_DECAY"])
        self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(),
                                             lr=self.config["LR_CRITIC"],
                                             weight_decay=self.config["WEIGHT_DECAY"])

        # Value Networks
        self.local_value_network = Value(state_size, random_seed, fcs_units=[128], fc_units=[128]).to(device)
        self.target_value_network = Value(state_size, random_seed, fcs_units=[128], fc_units=[128]).to(device)

        self.value_optimizer_local = optim.Adam(self.local_value_network.parameters(),
                                                lr=self.config["LR_CRITIC"],
                                                weight_decay=self.config["WEIGHT_DECAY"])

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.t_step = 0
        self.update_every = self.config["UPDATE_EVERY"]

        # Replay memory
        self.memory = getattr(mem_modules, self.config["MEMORY_MODULE"])(int(self.config["BUFFER_SIZE"]),
                                                                         int(self.config["BATCH_SIZE"]), random_seed)

        self.scale = self.config["SCALE"]

    @torch.no_grad()
    def act(self, state: np.ndarray) -> np.ndarray:
        self.actor_network.eval()
        # Convert State to Torch Tensor
        state = torch.from_numpy(state).float().to(device)
        actions, _ = self.actor_network.sample_normal(state, reparameterize=False)

        return actions.cpu().numpy()[0]

    def add_observation(self):
        self.t_step += 1

    def step(self,
             state: np.ndarray,
             action: Union[int, float, List[float], np.ndarray],
             reward: Union[int, float],
             next_state: np.ndarray,
             done: bool) -> None:
        # Save experience in replay memory, and use random sample from buffer to learn.
        self.memory.add(state, action, reward, next_state, done)

        self.add_observation()

        # If enough samples are available in memory, train models
        if len(self.memory) > self.config["BATCH_SIZE"]:
            experiences = self.memory.sample()
            self.learn(experiences, self.config["GAMMA"])

    def learn(self, experiences: Tuple[torch.Tensor], gamma: float) -> None:
        torch.autograd.set_detect_anomaly(True)

        self.actor_network.train()

        # TODO Understand this update function
        if self.memory.is_priority_buffer:
            states, actions, rewards, next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        rewards = rewards.view(-1)
        dones = dones.view(-1)

        value_target_next = self.update_value_networks(states, next_states, dones)

        self.update_actor(states)

        # Update Critic Networks
        self.update_critic_networks(states, value_target_next, actions, rewards, gamma)

        # Update target network
        soft_update(self.local_value_network, self.target_value_network, tau=1.0)

    def update_value_networks(self,
                              states: torch.Tensor,
                              next_states: torch.Tensor,
                              dones: torch.Tensor) -> torch.Tensor:
        # Update Value Network
        value_local = self.local_value_network(states).view(-1)
        value_target_next = self.target_value_network(next_states).view(-1)
        value_target_next[dones == 1.0] = 0.0

        new_actions, log_probabilities = self.actor_network.sample_normal(states, reparameterize=False)
        log_probabilities = log_probabilities.view(-1)

        # Reducing overestimation bias, follow the most pessimistic way to converge better
        q1_new_policy = self.critic_network_1(states, new_actions)
        q2_new_policy = self.critic_network_2(states, new_actions)

        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value_optimizer_local.zero_grad()
        value_target = critic_value - log_probabilities
        value_loss = 0.5 * F.mse_loss(value_local, value_target)

        # Retain graph is important here! TODO why!
        value_loss.backward(retain_graph=True)
        self.value_optimizer_local.step()

        return value_target_next

    def update_critic_networks(self, states: torch.Tensor,
                               value_target_next: torch.Tensor,
                               actions: torch.Tensor,
                               rewards: torch.Tensor,
                               gamma: float) -> None:
        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()

        # TODO why?
        q_hat = self.scale * rewards + gamma * value_target_next

        q1_old_policy = self.critic_network_1(states, actions).view(-1)
        q2_old_policy = self.critic_network_2(states, actions).view(-1)

        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss

        critic_loss.backward()

        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

    def update_actor(self, states: torch.Tensor) -> None:
        new_actions, log_probabilities = self.actor_network.sample_normal(states, reparameterize=True)
        log_probabilities = log_probabilities.view(-1)

        q1_new_policy = self.critic_network_1(states, new_actions)
        q2_new_policy = self.critic_network_2(states, new_actions)

        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probabilities - critic_value

        actor_loss = torch.mean(actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

    def reset(self) -> None:
        # Reset Noise Generator
        self.noise.reset()

    def save(self, name: str = "SAC", folder: Path = "saved_models") -> None:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        # Save Actor Network
        torch.save(self.actor_network.state_dict(), f"{folder}/{name}_actor_checkpoint_actor.pth")

        # Save Critic Network
        torch.save(self.critic_network_1.state_dict(), f"{folder}/{name}_critic_network_1_checkpoint_actor.pth")
        torch.save(self.critic_network_2.state_dict(), f"{folder}/{name}_critic_network_2_checkpoint_actor.pth")

        # Save Value Networks
        torch.save(self.local_value_network.state_dict(), f"{folder}/{name}_local_value_checkpoint_actor.pth")
        torch.save(self.target_value_network.state_dict(), f"{folder}/{name}_target_value_checkpoint_actor.pth")

    def load(self, actor_model_path: Path, critic_model_path: Path, value_model_path: Path) -> None:
        # Reload pre-trained models
        self.actor_network.load_state_dict(actor_model_path)
        self.critic_network_1.load_state_dict(critic_model_path)
        self.local_value_network.load_state_dict(value_model_path)

        soft_update(self.local_value_network, self.target_value_network, tau=1.0)
        soft_update(self.critic_network_1, self.critic_network_2, tau=1.0)
