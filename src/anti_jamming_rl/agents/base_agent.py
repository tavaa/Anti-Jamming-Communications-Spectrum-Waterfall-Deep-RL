from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for reinforcement learning agents. 
    RL agents should inherit from this class and implement the required methods.
    """
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def observe(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_mode(self, training):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass