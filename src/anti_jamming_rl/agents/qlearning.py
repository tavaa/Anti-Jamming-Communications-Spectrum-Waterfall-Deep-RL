import pickle
from typing import Dict, Tuple

import numpy as np

from anti_jamming_rl.agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Tabular Q-learning agent with a reduced state representation.

    Reduces the waterfall observation to a tuple consisting of the jammed
    channel indices at the latest time step and the agent's previous action,
    accounting for switching penalties in the state transition.

    Args:
        num_actions: Number of selectable discrete actions (channels).
        alpha_lr: Learning rate for tabular TD error updates.
        gamma: Discount factor for future rewards.
        epsilon_start: Initial exploration rate.
        epsilon_end: Asymptotic exploration rate.
        epsilon_decay_steps: Step horizon over which epsilon decays linearly.
        rng: Deterministic NumPy random generator instance.
    """

    def __init__(
        self,
        num_actions: int,
        alpha_lr: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        rng: np.random.Generator,
    ) -> None:
        self.num_actions = num_actions
        self.alpha, self.gamma = alpha_lr, gamma
        self.eps_start, self.eps_end, self.eps_decay_steps = (
            epsilon_start,
            epsilon_end,
            epsilon_decay_steps,
        )
        self.rng = rng
        self.training = True
        self._step_count = 0
        self._last_action = -1
        self._last_phi = None
        self.q_table: Dict[Tuple[Tuple[int, ...], int], np.ndarray] = {}

    def _reduce(self, state: np.ndarray) -> Tuple[Tuple[int, ...], int]:
        """Extract compressed state feature phi from full waterfall.

        Args:
            state: Array of shape (T, K) representing binary spectrum history.

        Returns:
            Tuple of (jammed_channels, last_action), where jammed_channels
            is a tuple of indices where occupancy exceeds 0.5 at step T-1.
        """
        last_spectrum = state[-1]
        jammed_channels = tuple(np.flatnonzero(last_spectrum > 0.5))
        return (jammed_channels, self._last_action)

    def _epsilon(self) -> float:
        """Compute current exploration rate under a linear decay schedule."""
        frac = min(1.0, self._step_count / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _greedy(self, phi: Tuple[Tuple[int, ...], int]) -> int:
        """Select greedy action breaking ties uniformly at random.

        Args:
            phi: Feature tuple (jammed_channels, last_action).

        Returns:
            Action index with maximal Q-value for feature phi.
        """
        if phi not in self.q_table:
            self.q_table[phi] = np.zeros(self.num_actions)
        q = self.q_table[phi]
        best = np.max(q)
        cands = np.flatnonzero(q == best)
        return int(self.rng.choice(cands))

    def select_action(self, state: np.ndarray) -> int:
        """Select action via epsilon-greedy exploration or greedy exploitation.

        Args:
            state: Waterfall matrix of shape (T, K).

        Returns:
            Action index in [0, num_actions - 1].
        """
        phi = self._reduce(state)
        self._last_phi = phi
        if self.training and self.rng.random() < self._epsilon():
            a = int(self.rng.integers(0, self.num_actions))
        else:
            a = self._greedy(phi)
        self._last_action = a
        return a

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ) -> Dict[str, float]:
        """Update tabular Q-values using 1-step TD update.

        Args:
            state: Waterfall matrix before action execution of shape (T, K).
            action: Action executed by agent.
            reward: Scalar reward received.
            next_state: Resulting waterfall matrix of shape (T, K).

        Returns:
            Dictionary with step metrics if training, empty dictionary otherwise.
        """
        if not self.training:
            return {}
        phi = self._last_phi
        jammed_next = tuple(np.flatnonzero(next_state[-1] > 0.5))
        phi_next = (jammed_next, action)

        if phi not in self.q_table:
            self.q_table[phi] = np.zeros(self.num_actions)
        if phi_next not in self.q_table:
            self.q_table[phi_next] = np.zeros(self.num_actions)

        td_target = reward + self.gamma * np.max(self.q_table[phi_next])
        td_error = td_target - self.q_table[phi][action]
        self.q_table[phi][action] += self.alpha * td_error
        self._step_count += 1
        return {"epsilon": self._epsilon(), "td_error": td_error}

    def set_mode(self, training: bool) -> None:
        """Set execution mode and reset variables.

        Args:
            training: True for learning with exploration, False for evaluation.
        """
        self.training = training
        self._last_action = -1
        self._last_phi = None

    def save(self, path: str) -> None:
        """Serialize tabular Q-dictionary to a pickle file.

        Args:
            path: Destination file path (.pkl).
        """
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path: str) -> None:
        """Deserialize tabular Q-dictionary from disk.

        Args:
            path: Source file path (.pkl).
        """
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)