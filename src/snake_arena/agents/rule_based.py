"""Rule-based baseline agent: explicit heuristics, no graph search.

Scoring rule per candidate action (documented, deterministic):

    U(a) = 2.0 * progress(a)            # manhattan improvement toward the
                                        # best-value non-trap food
         + 0.5 * min(1, free_area(a)/8) # short-term space after the move
         - 3.0 * trap(a)                # stepping onto trap food (if any
                                        # safe food exists)
         - 1.5 * contested(a)           # cell the opponent can reach next tick

Deadly moves are excluded up front; the safety layer re-checks the winner
anyway (defense in depth, and it keeps every agent behind the same pipeline).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..game.collision import flood_fill_count, is_deadly_for
from ..game.environment import StateView, build_state_view
from ..game.types import ALL_ACTIONS, DIR_VECTORS, Action
from .base import AgentDecision, BaseAgent, StrategyHints

if TYPE_CHECKING:
    from ..game.food import Food
    from ..game.state import GameState


class RuleBasedAgent(BaseAgent):
    """Survival-first heuristic agent; the classical baseline to beat."""

    name = "rule_based"

    def __init__(self, flood_limit: int = 900) -> None:
        self.flood_limit = flood_limit

    def decide(
        self,
        state: "GameState",
        snake_id: str,
        view: StateView | None = None,
        hints: StrategyHints | None = None,
    ) -> AgentDecision:
        t0 = time.perf_counter()
        view = view or build_state_view(state, snake_id)
        hints = hints or StrategyHints()
        me = view.me
        head = me.head

        positives = [f for f in state.foods if f.value > 0]
        target: Food | None = None
        if positives:
            if hints.is_aggressive:
                target = max(positives, key=lambda f: f.value)
            else:
                target = max(positives, key=lambda f: f.value / (1 + abs(f.pos[0] - head[0]) + abs(f.pos[1] - head[1])))
        trap_cells = {f.pos for f in state.foods if f.value < 0}

        opponent_cells: set[tuple[int, int]] = set()
        for opp in view.opponents:
            ox, oy = opp.head
            for ddx, ddy in DIR_VECTORS.values():
                cell = (ox + ddx, oy + ddy)
                if state.is_inside(cell):
                    opponent_cells.add(cell)

        scores: dict[Action, float] = {}
        areas: dict[Action, int] = {}
        notes: list[str] = []
        for action in ALL_ACTIONS:
            ddx, ddy = DIR_VECTORS[action]
            cell = (head[0] + ddx, head[1] + ddy)
            if not state.is_inside(cell):
                scores[action] = -1e9
                areas[action] = 0
                continue
            if is_deadly_for(state, me, cell):
                scores[action] = -1e9
                areas[action] = 0
                continue
            area = flood_fill_count(cell, view.blocked, state.width, state.height, self.flood_limit)
            if area < me.length:
                scores[action] = -1e6 + area  # dead-endish: only pick if forced
                areas[action] = area
                continue
            utility = 0.0
            if target is not None:
                before = abs(target.pos[0] - head[0]) + abs(target.pos[1] - head[1])
                after = abs(target.pos[0] - cell[0]) + abs(target.pos[1] - cell[1])
                utility += 2.0 * (before - after)
            utility += 0.5 * min(1.0, area / 8.0)
            if cell in trap_cells and positives:
                utility -= 3.0
            if cell in opponent_cells:
                utility -= 1.5
            scores[action] = utility
            areas[action] = area

        best = max(scores, key=lambda a: scores[a])
        if scores[best] <= -1e9:
            notes.append("all moves deadly; holding direction (unavoidable)")
        computation_ms = (time.perf_counter() - t0) * 1000.0
        return AgentDecision(
            agent=self.name,
            action=best,
            path=[head],
            target_food_id=target.id if target else None,
            target_pos=target.pos if target else None,
            notes=notes,
            diagnostics={
                "scores": {a.name.lower(): round(s, 2) for a, s in scores.items()},
                "areas": {a.name.lower(): ar for a, ar in areas.items()},
                "mode": "food" if target else "wander",
            },
            computation_ms=computation_ms,
        )
