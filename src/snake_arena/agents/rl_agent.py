"""DQN inference agent.

Wraps a (possibly untrained) Q-network as a standard agent. A freshly
initialized network is reported honestly as *untrained* everywhere: the UI
badge, agent ``describe()``, decision diagnostics, and benchmark rows all
carry the flag, so an untrained model can never masquerade as an intelligent
one. Use the Training dashboard or ``scripts/train.py`` to actually train.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..game.environment import OBS_SIZE, ObservationBuilder
from ..game.types import ALL_ACTIONS, Action
from ..rl.network import QNetwork, pick_device
from ..rl.trainer import DQNTrainer
from .base import AgentDecision, BaseAgent, StrategyHints

if TYPE_CHECKING:
    from ..game.environment import StateView
    from ..game.state import GameState
    from ..config import RLConfig


class RLAgent(BaseAgent):
    """Greedy policy agent backed by a Q-network."""

    name = "rl"

    def __init__(
        self,
        trainer: DQNTrainer | None = None,
        rl_cfg: "RLConfig | None" = None,
        device: torch.device | None = None,
    ) -> None:
        self.trainer = trainer or DQNTrainer(rl_cfg, obs_size=OBS_SIZE, device=device)
        self.obs_builder = ObservationBuilder()
        self._net = self.trainer.online
        self._net.eval()

    # -- checkpoint helpers ----------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path, rl_cfg: "RLConfig | None" = None) -> "RLAgent":
        trainer = DQNTrainer(rl_cfg, obs_size=OBS_SIZE)
        trainer.load(path)
        return cls(trainer=trainer)

    def load_checkpoint(self, path) -> dict:
        payload = self.trainer.load(path)
        self._net = self.trainer.online
        self._net.eval()
        return payload

    @property
    def trained(self) -> bool:
        return self.trainer.episodes_done > 0

    def describe(self) -> str:
        if self.trained:
            return f"DQN ({self.trainer.episodes_done} episodes trained)"
        return "DQN (untrained: random init - train it first!)"

    # -- decision ---------------------------------------------------------------

    def decide(
        self,
        state: "GameState",
        snake_id: str,
        view: "StateView | None" = None,
        hints: StrategyHints | None = None,
    ) -> AgentDecision:
        t0 = time.perf_counter()
        view = view or build_view(state, snake_id)
        obs = self.obs_builder.build(state, snake_id, view)
        with torch.no_grad():
            tensor = torch.from_numpy(obs.astype(np.float32)).to(self.trainer.device)
            q = self._net(tensor.unsqueeze(0)).squeeze(0)
        q_values = {action: float(q[int(action)]) for action in ALL_ACTIONS}
        # Rank actions by Q-value; the DecisionManager uses this ranking when
        # the LLM strategy demands extra caution (risk gating).
        ranked = sorted(ALL_ACTIONS, key=lambda a: q_values[a], reverse=True)
        action = ranked[0]
        notes: list[str] = []
        if not self.trained:
            notes.append("untrained network: actions are essentially random")
        target = view.target
        return AgentDecision(
            agent=self.name,
            action=action,
            path=[view.me.head],
            target_food_id=target.id if target else None,
            target_pos=target.pos if target else None,
            q_values=q_values,
            notes=notes,
            diagnostics={
                "trained": self.trained,
                "episodes": self.trainer.episodes_done,
                "epsilon_at_eval": 0.0,
                "q_ranking": [a.name.lower() for a in ranked],
                "obs": obs,
                "mode": "policy",
            },
            computation_ms=(time.perf_counter() - t0) * 1000.0,
        )


def build_view(state: "GameState", snake_id: str):
    from ..game.environment import build_state_view

    return build_state_view(state, snake_id)
