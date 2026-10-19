"""Mandatory safety checker between agent decisions and movement.

Every AI action must pass through :class:`SafetyValidator` before it reaches
the engine. Validation is pure game logic -- it never trusts agent output or
LLM text -- and rejects moves that would:

    * leave the board (wall),
    * immediately reverse into the snake's own neck,
    * enter the snake's own body (tail-aware),
    * enter an opponent's body (tail-aware),
    * enter a dead end: reachable free area smaller than the snake.

If the preferred action fails, callers can ask for the safest alternative
ranked by reachable free area (then by distance to the current target).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import SafetyConfig
from ..game.collision import flood_fill_count, is_deadly_for
from ..game.environment import StateView, build_state_view
from ..game.types import ALL_ACTIONS, DIR_VECTORS, Action, Cell, manhattan

if TYPE_CHECKING:
    from ..game.snake import Snake
    from ..game.state import GameState


@dataclass
class SafetyReport:
    action: Action
    safe: bool
    reason: str | None = None
    free_area: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (action_name, reason)

    @property
    def intervention(self) -> bool:
        """True when the preferred action had to be replaced."""
        return bool(self.rejected)


class SafetyValidator:
    """Validates proposed actions against real game-transition rules."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    # -- single action -------------------------------------------------------

    def check(
        self,
        state: "GameState",
        snake_id: str,
        action: Action,
        view: StateView | None = None,
    ) -> SafetyReport:
        snake = state.snakes[snake_id]
        view = view or build_state_view(state, snake_id)

        if action not in ALL_ACTIONS:
            return SafetyReport(action, safe=False, reason="invalid_action")

        dx, dy = DIR_VECTORS[action]
        cell: Cell = (snake.head[0] + dx, snake.head[1] + dy)

        if not state.is_inside(cell):
            return SafetyReport(action, False, "wall", 0)

        if snake.is_reversal(action):
            return SafetyReport(action, False, "reversal", 0)

        deadly = is_deadly_for(state, snake, cell, treat_opponent_head_vacated=True)
        if deadly == "self":
            return SafetyReport(action, False, "self_collision", 0)
        if deadly == "opponent":
            return SafetyReport(action, False, "opponent_collision", 0)

        area = flood_fill_count(cell, view.blocked, state.width, state.height, self.config.flood_fill_limit)
        if self.config.dead_end_check:
            min_area = max(2, int(snake.length * self.config.min_escape_area_factor))
            if area < min_area:
                return SafetyReport(action, False, "dead_end", area)
        return SafetyReport(action, True, None, area)

    # -- candidate ranking ---------------------------------------------------

    def evaluate_all(
        self,
        state: "GameState",
        snake_id: str,
        view: StateView | None = None,
    ) -> dict[Action, SafetyReport]:
        return {action: self.check(state, snake_id, action, view) for action in ALL_ACTIONS}

    def safe_actions(
        self,
        state: "GameState",
        snake_id: str,
        view: StateView | None = None,
    ) -> list[Action]:
        reports = self.evaluate_all(state, snake_id, view)
        return [a for a, r in reports.items() if r.safe]

    def select_safe_action(
        self,
        state: "GameState",
        snake_id: str,
        preferred: Action,
        fallback_order: list[Action] | None = None,
        view: StateView | None = None,
        target: Cell | None = None,
    ) -> tuple[Action, SafetyReport]:
        """Return (action, report): the preferred action if safe, otherwise
        the best alternative by reachable free area (then target distance).

        The report records every rejected candidate so the UI can show why
        the agent's first choice was overridden. If no action is safe the
        preferred action is returned with ``safe=False`` -- the engine then
        applies its terminal/unavoidable-collision rules; we never invent an
        illegal move.
        """
        view = view or build_state_view(state, snake_id)
        reports = self.evaluate_all(state, snake_id, view)
        rejected: list[tuple[str, str]] = []

        preferred_report = reports.get(preferred)
        if preferred_report is not None and preferred_report.safe:
            return preferred, preferred_report
        if preferred_report is not None:
            rejected.append((preferred.name.lower(), preferred_report.reason or "unsafe"))

        snake: Snake = state.snakes[snake_id]
        candidates = [a for a in (fallback_order or []) if a != preferred and reports[a].safe]
        candidates += [a for a, r in reports.items() if r.safe and a not in candidates and a != preferred]

        def rank(action: Action) -> tuple[float, float]:
            report = reports[action]
            dist = 0.0
            if target is not None:
                ddx, ddy = DIR_VECTORS[action]
                nxt = (snake.head[0] + ddx, snake.head[1] + ddy)
                dist = manhattan(nxt, target)
            return (-float(report.free_area), dist)

        if candidates:
            best = min(candidates, key=rank)
            return best, SafetyReport(best, True, None, reports[best].free_area, rejected)

        # No safe action exists: return the preferred action untouched and let
        # the engine resolve the unavoidable collision honestly.
        return preferred, SafetyReport(preferred, False, "no_safe_action", 0, rejected)

    def risky_cells(self, state: "GameState", snake_id: str) -> set[Cell]:
        """Cells flagged as risky for visualization (would fail validation)."""
        view = build_state_view(state, snake_id)
        risky: set[Cell] = set()
        snake: Snake = state.snakes[snake_id]
        for action in ALL_ACTIONS:
            ddx, ddy = DIR_VECTORS[action]
            cell = (snake.head[0] + ddx, snake.head[1] + ddy)
            if not self.check(state, snake_id, action, view).safe and state.is_inside(cell):
                risky.add(cell)
        return risky
