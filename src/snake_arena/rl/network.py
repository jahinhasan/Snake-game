"""Q-network architectures (CPU-first; CUDA optional, never required)."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """MLP mapping the flat observation to per-action Q-values.

    Structure: obs -> [Linear(h) + ReLU] * len(hidden_sizes) -> Linear(4).
    Small by design: on a 30-dim observation, 128x128 is plenty and trains
    comfortably on an ordinary CPU.
    """

    def __init__(self, obs_size: int, hidden_sizes: list[int] | None = None, n_actions: int = 4) -> None:
        super().__init__()
        hidden = hidden_sizes or [128, 128]
        layers: list[nn.Module] = []
        in_dim = obs_size
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def act_greedy(self, obs: torch.Tensor) -> int:
        q = self.forward(obs.unsqueeze(0))
        return int(q.argmax(dim=1).item())

    @torch.no_grad()
    def q_values(self, obs: torch.Tensor) -> list[float]:
        q = self.forward(obs.unsqueeze(0)).squeeze(0)
        return [float(v) for v in q]


def pick_device() -> torch.device:
    """CPU by default; CUDA only when genuinely available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
