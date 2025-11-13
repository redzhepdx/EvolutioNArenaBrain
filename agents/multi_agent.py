import numpy as np
import torch

import agents.algorithms as algos
from utils.memory_modules import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GenericMultiObserverSingleAgent(object):
    def __init__(self, agent, num_observers=2):
        self.agent = agent
        self.num_observers = num_observers

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for agent_idx in range(self.num_observers):
            # Save experience / reward
            self.agent.memory.add(states[agent_idx, :],
                                  actions[agent_idx, :],
                                  rewards[agent_idx],
                                  next_states[agent_idx, :],
                                  dones[agent_idx])

        # Increment agent's total step count
        self.agent.add_observation()

        # If enough samples are available in memory, train models
        if len(self.agent.memory) > (self.agent.config["BATCH_SIZE"] * self.num_observers):
            experiences = self.agent.memory.sample()
            self.agent.learn(experiences, self.agent.config["GAMMA"])

    def reset(self):
        self.agent.reset()

    def act(self, states, add_noise=True):
        return self.agent.act(states, add_noise=add_noise)

    def save(self):
        self.agent.save()


class GenericSingleObserverMultiAgent(object):
    def __init__(self, state_size, action_size, random_seed, config, num_agents=2, algo="T3D"):
        """
        Initialize an Multi Agent object.(Self Play)


        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.num_agents = num_agents
        self.config = config
        self.t_step = 0
        self.update_every = 4

        # Agents
        self.agents = [getattr(algos, f"{algo}Agent")(state_size, action_size, random_seed, config) for _ in
                       range(num_agents)]

        # Replay memory
        self.memory = ReplayBuffer(self.config["BUFFER_SIZE"], self.config["BATCH_SIZE"], random_seed)

    def act(self, states):
        return np.array([agent.act(state) for agent, state in zip(self.agents, states)])

    def step(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        self.t_step += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.config["BATCH_SIZE"]:
            for agent_idx in range(self.num_agents):
                experiences = self.memory.sample()
                self.agents[agent_idx].t_step = self.t_step
                self.agents[agent_idx].learn(experiences, gamma=self.config["GAMMA"])

    def reset(self):
        for agent_idx in range(self.num_agents):
            self.agents[agent_idx].reset()

    def save(self):
        for agent_idx in range(self.num_agents):
            self.agents[agent_idx].save(name=f"agent_{agent_idx + 1}")


class GenericMultiObserverMultiAgent(object):
    def __init__(self):
        """
        Distributed or Multi-threaded Self Play
        """
        pass

    def act(self, states):
        pass

    def step(self, states, actions, rewards, next_states, dones):
        pass

    def reset(self):
        pass

    def save(self):
        pass
