"""Common agent contract.

Every agent receives a structured view of the game and returns an
``AgentDecision`` *proposal*. Proposals are never applied directly: the
DecisionManager passes them through the SafetyValidator first. Agents are
stateless across ticks apart from optional diagnostics, which keeps them
safe to reuse across benchmark episodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..game.types import Action

if TYPE_CHECKING:
    from ..game.environment import StateView
    from ..game.state import GameState


@dataclass
class StrategyHints:
    """Strategic parameters produced by the LLM planner (or fallback).

    Classical agents interpret these as soft weighting; they are advisory and
    can never make an agent output an illegal move (safety still applies).
    """

    strategy: str = "balanced"          # defensive|balanced|aggressive|greedy|survival
    food_priority: float = 0.5          # 0..1 how much to chase food
    survival_priority: float = 0.5      # 0..1 how much to weight survival
    risk_tolerance: float = 0.5         # 0..1 willingness to take risky routes
    target_food_id: str | None = None   # optional explicit target from the LLM
    reason: str = ""
    source: str = "default"             # default | llm | fallback
    latency_ms: float = 0.0

    @property
    def is_aggressive(self) -> bool:
        return self.strategy in ("aggressive", "greedy")

    @property
    def is_defensive(self) -> bool:
        return self.strategy in ("defensive", "survival")


@dataclass
class AgentDecision:
    """A proposed action plus full diagnostics for the UI and metrics."""

    agent: str
    action: Action
    path: list[tuple[int, int]] = field(default_factory=list)
    target_food_id: str | None = None
    target_pos: tuple[int, int] | None = None
    q_values: dict[Action, float] | None = None
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    computation_ms: float = 0.0

    @property
    def path_length(self) -> int:
        return max(0, len(self.path) - 1)


class BaseAgent(ABC):
    """Abstract agent: observe -> decide (propose)."""

    name: str = "base"

    @abstractmethod
    def decide(
        self,
        state: "GameState",
        snake_id: str,
        view: "StateView | None" = None,
        hints: StrategyHints | None = None,
    ) -> AgentDecision:
        """Propose an action for ``snake_id`` for the current tick."""

    def reset(self) -> None:
        """Clear any per-episode state (default: none)."""

    def describe(self) -> str:
        return self.name
