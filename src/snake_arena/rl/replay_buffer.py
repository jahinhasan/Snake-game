"""Uniform experience replay buffer with preallocated NumPy storage.

Fixed-capacity ring buffer: O(1) push, no Python-object churn, and a hard
memory ceiling (capacity * (2*obs+2) floats), which satisfies the
"no unbounded buffers" requirement.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_size: int, rng: np.random.Generator | None = None) -> None:
        self.capacity = int(capacity)
        self.obs_size = obs_size
        self.rng = rng or np.random.default_rng()
        self.obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = 1.0 if done else 0.0
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def can_sample(self, batch_size: int) -> bool:
        return len(self) >= batch_size

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.rng.integers(0, len(self), size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.dones[idx]),
        )
