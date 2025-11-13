import os
import random
from ctypes import Union
from pathlib import Path
from typing import List, Tuple, Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils.memory_modules as mem_modules
from agents.algorithms.agent import Agent
from model.networks import ActorNonLinear, Critic
from utils.model_utils import soft_update, hard_copy_weights
from utils.noise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size: Union[List[int], Tuple[int], int],
                 action_size: int,
                 random_seed: int,
                 config: Dict[str, Any]):
        """
        :param state_size: dimension of each state
        :param action_size : dimension of each action
        :param random_seed : random seed
        :param config : dictionary of hyper-parameters

             Example Config :
            {
                LR_ACTOR : 1e-4,
                LR_CRITIC : 1e-4,
                WEIGHT_DECAY : 1e-6,
                UPDATE_EVERY : 2,
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

        # Actor Network
        self.actor_local = ActorNonLinear(state_size, action_size, random_seed, fc_units=[512, 512]).to(device)
        self.actor_target = ActorNonLinear(state_size, action_size, random_seed, fc_units=[512, 512]).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config["LR_ACTOR"])

        # Critic Network
        self.critic_local = Critic(state_size, action_size, random_seed, fcs_units=[512], fc_units=[512]).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, fcs_units=[512], fc_units=[512]).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.config["LR_CRITIC"],
                                           weight_decay=self.config["WEIGHT_DECAY"])

        hard_copy_weights(self.actor_target, self.actor_local)
        hard_copy_weights(self.critic_target, self.critic_local)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        self.t_step = 0
        self.update_every = self.config["UPDATE_EVERY"]

        # Replay memory
        self.memory = getattr(mem_modules, self.config["MEMORY_MODULE"])(int(self.config["BUFFER_SIZE"]),
                                                                         int(self.config["BATCH_SIZE"]),
                                                                         random_seed)

    def step(self, state: np.ndarray,
             action: Union[int, float, List[float], np.ndarray],
             reward: Union[int, float],
             next_state: np.ndarray,
             done: bool) -> None:
        # Save experience in replay memory, and use random sample from buffer to learn.
        self.memory.add(state, action, reward, next_state, done)

        # If enough samples are available in memory, train models
        if len(self.memory) > self.config["BATCH_SIZE"]:
            experiences = self.memory.sample()
            self.learn(experiences, self.config["GAMMA"])

    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        # Returns noisy actions for given state as per current policy.
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

    def add_observation(self) -> None:
        self.t_step += 1

    def learn(self, experiences: Tuple[torch.Tensor], gamma: float) -> None:
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param experiences : tuple of (s, a, r, s', done) tuples
        :param gamma : discount factor
        """

        self.update_local_critic(experiences=experiences, gamma=gamma)

        self.update_local_actor(experiences)

        if self.t_step % self.update_every == 0:
            # ----------------------- update target networks ----------------------- #
            soft_update(self.critic_local, self.critic_target, self.config["TAU"])
            soft_update(self.actor_local, self.actor_target, self.config["TAU"])

    def update_local_critic(self, experiences: Tuple[torch.Tensor], gamma: float) -> None:
        if self.memory.is_priority_buffer:
            states, actions, rewards, next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)

        if self.memory.is_priority_buffer:
            critic_loss = critic_loss * weights

            # Update Priorities
            prios = critic_loss + 1e-5
            self.memory.update_priorities(indices.data.cpu().numpy(),
                                          prios.data.cpu().numpy())

            critic_loss = critic_loss.mean()

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Clip gradients of critic
        if self.config["CLIP_CRITIC_GRADIENTS"]:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        # Apply Changes
        self.critic_optimizer.step()

    def update_local_actor(self, experiences: Tuple[torch.Tensor]) -> None:
        if self.memory.is_priority_buffer:
            states, actions, rewards, next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

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
        torch.save(self.critic_local.state_dict(), f"{folder}/{name}_critic_checkpoint_actor.pth")

    def load(self, actor_model_path: Path, critic_model_path: Path) -> None:
        # Reload pre-trained models
        self.actor_local.load_state_dict(actor_model_path)
        self.critic_local.load_state_dict(critic_model_path)
