import numpy as np
from anti_jamming_rl.agents.base_agent import BaseAgent


class FixedAgent(BaseAgent):
    """
    Selects always a fixed action for the entire simulation.
    """
    def __init__(self, num_actions, rng):
        self._action = int(rng.integers(0, num_actions))

    def select_action(self, state):
        return self._action

    def observe(self, *a, **k):
        return {}

    def set_mode(self, training):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class RandomAgent(BaseAgent):
    """
    Selects actions uniformly at random.
    """
    def __init__(self, num_actions, rng):
        self.num_actions, self.rng = num_actions, rng

    def select_action(self, state):
        return int(self.rng.integers(0, self.num_actions))

    def observe(self, *a, **k):
        return {}

    def set_mode(self, training):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class ReactiveAgent(BaseAgent):
    """Stay on the current channel if it was free in the last observation.
    If the current channel was jammed, switch uniformly at random among
    the other K-1 channels.

    The agent uses only the observed state and does not learn.
    """

    def __init__(self, num_actions, rng):
        self.num_actions = num_actions
        self.rng = rng
        self._current = int(rng.integers(0, num_actions))

    def select_action(self, state):
        last_spectrum = state[-1]

        # Current channel was free -> stay on it.
        if last_spectrum[self._current] < 0.5:
            return self._current

        # Current channel was jammed -> choose uniformly among the
        # other K-1 channels (never reselect the currently jammed one).
        candidates = np.arange(self.num_actions)
        candidates = candidates[candidates != self._current]

        self._current = int(self.rng.choice(candidates))
        return self._current

    def observe(self, state, action, reward, next_state):
        return {}

    def set_mode(self, training):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass