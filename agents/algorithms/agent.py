from abc import ABCMeta, abstractmethod


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def act(self, state, add_noise=True):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def add_observation(self):
        pass

    @abstractmethod
    def learn(self, experiences, gamma):
        pass

    @abstractmethod
    def save(self, name="agent_1_", folder="saved_models"):
        pass

    @abstractmethod
    def load(self, **kwargs):
        pass
