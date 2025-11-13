import os
import random
from ctypes import Union
from pathlib import Path
from typing import List, Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils.memory_modules as mem_modules
from model.networks import ActorNonLinear, Critic
from utils.model_utils import soft_update, hard_copy_weights
from utils.noise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3Agent(object):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size: Union[List[int], Tuple[int], int],
                 action_size: int,
                 random_seed: int,
                 config: Dict[str, Any]):
        """
        Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            config (dict) : dictionary of hyper-parameters

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
                CLIP_ACTOR_GRADIENTS : false,
                CLIP_CRITIC_GRADIENTS : true,
                MEMORY_MODULE : NaivePrioritizedBuffer # ReplayBuffer
            }
        """
        random.seed(random_seed)

        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Actor Network (w/ Target Network)
        self.actor_local = ActorNonLinear(state_size, action_size, random_seed, fc_units=[512, 512]).to(device)
        self.actor_target = ActorNonLinear(state_size, action_size, random_seed, fc_units=[512, 512]).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config["LR_ACTOR"])
        hard_copy_weights(self.actor_target, self.actor_local)

        # Critic Networks (w/ Target Networks)
        self.critic_locals = []
        self.critic_targets = []
        self.critic_optimizers = []

        for critic_network_idx in range(self.config["CRITIC_COUNT"]):
            # Create a Critic Local Network
            self.critic_locals.append(
                Critic(state_size, action_size, random_seed, fcs_units=[512], fc_units=[512]).to(device))

            # Create a Critic Target Network
            self.critic_targets.append(
                Critic(state_size, action_size, random_seed, fcs_units=[512], fc_units=[512]).to(device))

            self.critic_optimizers.append(optim.Adam(self.critic_locals[critic_network_idx].parameters(),
                                                     lr=self.config["LR_CRITIC"],
                                                     weight_decay=self.config["WEIGHT_DECAY"]))

            hard_copy_weights(self.critic_targets[critic_network_idx], self.critic_locals[critic_network_idx])

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.t_step = 0
        self.update_every = self.config["UPDATE_EVERY"]

        # Replay memory
        self.memory = getattr(mem_modules, self.config["MEMORY_MODULE"])(int(self.config["BUFFER_SIZE"]),
                                                                         int(self.config["BATCH_SIZE"]), random_seed)

    def step(self,
             state: np.ndarray,
             action: Union[int, float, List[float], np.ndarray],
             reward: Union[int, float],
             next_state: np.ndarray,
             done: bool):
        # Save experience in replay memory, and use random sample from buffer to learn.
        self.memory.add(state, action, reward, next_state, done)

        self.add_observation()

        # If enough samples are available in memory, train models
        if len(self.memory) > self.config["BATCH_SIZE"]:
            experiences = self.memory.sample()
            self.learn(experiences, self.config["GAMMA"])

    def add_observation(self) -> None:
        self.t_step += 1

    def act(self, state: np.ndarray, add_noise=True) -> np.ndarray:
        # Convert State to Torch Tensor
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, self.config["ACTION_MIN"], self.config["ACTION_MAX"])

    def reset(self) -> None:
        # Reset Noise Generator
        self.noise.reset()

    def learn(self, experiences: Tuple[torch.Tensor], gamma: float) -> None:
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param experiences : tuple of (s, a, r, s', done) tuples
        :param gamma: discount factor
        :param t_step : total_step count
        update_every : update model once in every n steps
        """

        self.update_critics(experiences=experiences, gamma=gamma)

        if self.t_step % self.config["POLICY_DELAY"] == 0:
            # Update policy network with delay
            self.update_local_actor(experiences)

        if self.t_step % self.update_every == 0:
            # ----------------------- update target networks ----------------------- #
            for critic_network_idx in range(self.config["CRITIC_COUNT"]):
                soft_update(self.critic_locals[critic_network_idx],
                            self.critic_targets[critic_network_idx],
                            self.config["TAU"])
            soft_update(self.actor_local, self.actor_target, self.config["TAU"])

    def update_critics(self, experiences: Tuple[torch.Tensor], gamma: float) -> None:
        if self.memory.is_priority_buffer:
            states, actions, rewards, next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
        # ---------------------------- Update Critics ---------------------------- #
        # Get predicted next-state actions with noise and Q values from target models
        noise = torch.tensor(self.noise.sample()).float().to(device)

        # Add noise to actor target to increase exploration
        actions_next = torch.clamp(self.actor_target(next_states) + noise,
                                   self.config["ACTION_MIN"],
                                   self.config["ACTION_MAX"])

        # Find Minimum q_target_next
        q_targets_nexts = torch.zeros(self.config["CRITIC_COUNT"], len(states)).to(device)

        for critic_network_idx in range(self.config["CRITIC_COUNT"]):
            q_targets_next = self.critic_targets[critic_network_idx](next_states, actions_next)
            q_targets_nexts[critic_network_idx, :] = torch.squeeze(q_targets_next, 1)

        q_targets_next = q_targets_nexts.min(dim=0).values.unsqueeze(1)

        # Compute minimum Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        for critic_network_idx in range(self.config["CRITIC_COUNT"]):
            # Compute critic loss
            q_expected = self.critic_locals[critic_network_idx](states, actions)
            critic_loss = F.mse_loss(q_expected, q_targets)

            if self.memory.is_priority_buffer:
                critic_loss = critic_loss * weights

                # Update Priorities
                if critic_network_idx == 0:
                    prios = critic_loss + 1e-5
                    self.memory.update_priorities(indices.data.cpu().numpy(),
                                                  prios.data.cpu().numpy())
                critic_loss = critic_loss.mean()

            self.critic_optimizers[critic_network_idx].zero_grad()
            critic_loss.backward(retain_graph=True)

            # Clip gradients of critic
            if self.config["CLIP_CRITIC_GRADIENTS"]:
                torch.nn.utils.clip_grad_norm_(self.critic_locals[critic_network_idx].parameters(), 1)

            # Apply Changes
            self.critic_optimizers[critic_network_idx].step()

    def update_local_actor(self, experiences: Tuple[torch.Tensor]) -> None:
        if self.memory.is_priority_buffer:
            states, actions, rewards, next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
        # Compute actor loss
        actions_pred = self.actor_local(states)
        # Gradient Ascent
        actor_loss = -self.critic_locals[0](states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Clip gradients of critic
        if self.config["CLIP_ACTOR_GRADIENTS"]:
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)

        # Apply Changes
        self.actor_optimizer.step()

    def save(self, name: str = "agent_1_", folder: Path = "saved_models") -> None:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        # Save actor and critic models
        torch.save(self.actor_local.state_dict(), f"{folder}/{name}_actor_checkpoint_actor.pth")
        for critic_network_idx in range(self.config["CRITIC_COUNT"]):
            torch.save(self.critic_locals[critic_network_idx].state_dict(),
                       f"{folder}/{name}_critic_{critic_network_idx + 1}_checkpoint_actor.pth")

    def load(self, actor_model_path: Path, critic_model_paths: List[Path]) -> None:
        # Reload pre-trained models
        self.actor_local.load_state_dict(actor_model_path)

        for critic_network_idx in range(self.config["CRITIC_COUNT"]):
            self.critic_locals[critic_network_idx].load_state_dict(critic_model_paths[critic_network_idx])
