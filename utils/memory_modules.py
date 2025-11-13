import random
from collections import namedtuple, deque

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO Update
class HindsightReplayMemory(object):
    def __init__(self):
        pass


# TODO Update
class NaivePrioritizedBuffer(object):
    """Adapted From https://github.com/higgsfield/RL-Adventure"""

    def __init__(self, buffer_size, batch_size, seed, prob_alpha=0.6):
        random.seed(seed)

        self.prob_alpha = prob_alpha
        self.capacity = buffer_size
        self.batch_size = batch_size
        self.is_priority_buffer = True

        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim

        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)

        max_prio = np.amax(self.priorities) if len(self.memory) else 1.0

        self.memory.append(e)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # Convert state, action, reward information to torch tensor
        states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in samples if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(device)

        indices = torch.from_numpy(indices).float().to(device)
        weights = torch.from_numpy(weights).float().to(device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices.astype(np.uint).tolist(), batch_priorities.tolist()):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        random.seed(seed)

        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.is_priority_buffer = False

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
