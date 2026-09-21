import random
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anti_jamming_rl.agents.base_agent import BaseAgent


class QNetwork(nn.Module):
    """Deep Q-Network with 2D convolutions over waterfall spectrograms.

    Applies circular padding along the frequency axis to preserve
    topological adjacency between channel 0 and channel K - 1.

    Args:
        T: Time window length (number of past spectrum snapshots).
        K: Number of frequency channels.
        num_actions: Size of discrete action space.
        fc_hidden: Dimension of the fully connected hidden layer.
    """

    def __init__(self, T: int, K: int, num_actions: int, fc_hidden: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), padding_mode="circular"
        )
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), padding_mode="circular"
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, T, K)
            flat_dim = self._conv(dummy).shape[1]
        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_actions)

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers.

        Args:
            x: Input tensor of shape (B, 1, T, K).

        Returns:
            Flattened feature tensor of shape (B, flat_dim).
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action Q-values for input batch.

        Args:
            x: Input tensor of shape (B, 1, T, K).

        Returns:
            Q-values of shape (B, num_actions).
        """
        x = self._conv(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    """Fixed-capacity experience replay buffer for off-policy transitions.

    Args:
        capacity: Maximum number of stored transitions.
        rng: Deterministic random generator instance for uniform sampling.
    """

    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.rng = rng
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
    ) -> None:
        """Append a transition tuple to the ring buffer.

        Args:
            s: Current state observation of shape (T, K).
            a: Executed action index.
            r: Scalar reward obtained.
            s2: Next state observation of shape (T, K).
        """
        self.buffer.append((s, a, r, s2))

    def sample(
        self, n: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample an uncorrelated mini-batch of transitions.

        Args:
            n: Number of samples to draw.

        Returns:
            Tuple of arrays (states, actions, rewards, next_states) with shapes:
            - states: (n, T, K)
            - actions: (n,)
            - rewards: (n,)
            - next_states: (n, T, K)
        """
        batch = self.rng.sample(self.buffer, n)
        s, a, r, s2 = zip(*batch)
        return np.stack(s), np.array(a), np.array(r, dtype=np.float32), np.stack(s2)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with target network and experience replay.

    Args:
        T: Time window length.
        K: Total frequency channels.
        num_actions: Number of selectable actions.
        gamma: Discount factor.
        learning_rate: Adam optimizer learning rate.
        epsilon_start: Initial exploration rate.
        epsilon_end: Asymptotic exploration rate.
        epsilon_decay_steps: Steps over which epsilon decays linearly.
        replay_capacity: Capacity of experience replay buffer.
        warmup_size: Minimum buffer size required before learning steps begin.
        batch_size: Mini-batch size per optimization step.
        target_sync_steps: Interval between target network weight updates.
        seed: Random seed for initialization and exploration.
        device: Device identifier ('cpu' or 'cuda').
        fc_hidden: Latent dimension of hidden linear layer.
    """

    def __init__(
        self,
        T: int,
        K: int,
        num_actions: int,
        gamma: float,
        learning_rate: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        replay_capacity: int,
        warmup_size: int,
        batch_size: int,
        target_sync_steps: int,
        seed: int,
        device: str = "cpu",
        fc_hidden: int = 128,
    ) -> None:
        self.T, self.K, self.num_actions = T, K, num_actions
        self.gamma = gamma
        self.eps_start, self.eps_end, self.eps_decay_steps = (
            epsilon_start,
            epsilon_end,
            epsilon_decay_steps,
        )
        self.warmup_size, self.batch_size, self.target_sync_steps = (
            warmup_size,
            batch_size,
            target_sync_steps,
        )
        self.device, self.training, self._step_count = device, True, 0

        torch.manual_seed(seed)
        self.online = QNetwork(T, K, num_actions, fc_hidden=fc_hidden).to(device)
        self.target = QNetwork(T, K, num_actions, fc_hidden=fc_hidden).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

        self.replay = ReplayBuffer(replay_capacity, random.Random(seed))
        self.rng = np.random.default_rng(seed)

    def _to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert a 2D numpy observation into a 3D tensor on target device.

        Args:
            state: Array of shape (T, K).

        Returns:
            Tensor of shape (1, T, K).
        """
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _epsilon(self) -> float:
        """Compute the current epsilon value via linear schedule."""
        frac = min(1.0, self._step_count / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy exploration or greedy policy.

        Args:
            state: Waterfall matrix of shape (T, K).

        Returns:
            Chosen action index in [0, num_actions - 1].
        """
        if self.training and self.rng.random() < self._epsilon():
            return int(self.rng.integers(0, self.num_actions))
        with torch.no_grad():
            q = self.online(self._to_tensor(state).unsqueeze(0))
        return int(torch.argmax(q, dim=1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ) -> Dict[str, float]:
        """Record step transition, run gradient step if ready, and sync target weights.

        Args:
            state: Transition start state of shape (T, K).
            action: Action executed.
            reward: Scalar reward received.
            next_state: Transition end state of shape (T, K).

        Returns:
            Dictionary containing step metrics if training, empty dict otherwise.
        """
        if not self.training:
            return {}
        self.replay.push(state, action, reward, next_state)
        self._step_count += 1
        if len(self.replay) >= max(self.warmup_size, self.batch_size):
            self._update()
        if self._step_count % self.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {}

    def _update(self) -> None:
        """Sample mini-batch, compute 1-step MSE loss, and optimize online network."""
        s, a, r, s2 = self.replay.sample(self.batch_size)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.tensor(s2, dtype=torch.float32, device=self.device).unsqueeze(1)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device)

        q_values = self.online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(s2_t).max(dim=1).values
            td_target = r_t + self.gamma * next_q

        loss = F.mse_loss(q_values, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def set_mode(self, training: bool) -> None:
        """Switch agent between training and evaluation mode.

        Args:
            training: True to enable exploration and parameter updates, False for greedy evaluation.
        """
        self.training = training

    def save(self, path: str) -> None:
        """Serialize online network weights to disk.

        Args:
            path: Destination file path (.pt).
        """
        torch.save(self.online.state_dict(), path)

    def load(self, path: str) -> None:
        """Restore weights into both online and target networks.

        Args:
            path: Source file path containing serialized state dictionary.
        """
        sd = torch.load(path, map_location=self.device)
        self.online.load_state_dict(sd)
        self.target.load_state_dict(sd)