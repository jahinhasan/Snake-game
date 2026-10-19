"""Real A* graph search on the arena grid plus the A*-based snake agent.

Algorithm (Hart, Nilsson & Raphael 1968):

    open  = min-priority queue keyed by f(n) = g(n) + h(n)
    g(n)  = cost from start; h(n) = Manhattan distance to the goal
            (admissible and consistent on a 4-connected unit-cost grid)

Each expansion costs 1 per step plus a small turn penalty so resulting paths
are smoother (nicer to watch) without ever exceeding an un-penalized path's
length by more than the penalty budget. Search treats tail-aware blocked
cells as walls: tails without pending growth are *not* walls because they
vacate on the next tick.

The agent re-plans every tick (cheap on these board sizes) which removes any
possibility of acting on a stale path after the world changed.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import TYPE_CHECKING

from ..game.collision import flood_fill_count
from ..game.environment import StateView, build_state_view
from ..game.types import DIR_VECTORS, Action, Cell, manhattan
from .base import AgentDecision, BaseAgent, StrategyHints

if TYPE_CHECKING:
    from ..game.food import Food
    from ..game.state import GameState


def astar_search(
    start: Cell,
    goal: Cell,
    width: int,
    height: int,
    blocked: set[Cell],
    turn_penalty: float = 0.0,
    max_expansions: int = 8000,
) -> list[Cell] | None:
    """A* shortest path from ``start`` to ``goal``; None if unreachable.

    Returns the path including both endpoints. ``blocked`` cells are walls;
    ``start`` is never treated as blocked (the snake's head stands there).
    """
    if start == goal:
        return [start]
    if start in blocked:
        blocked = blocked - {start}

    open_heap: list[tuple[float, int, Cell]] = []
    counter = 0
    g_score: dict[Cell, float] = {start: 0.0}
    came_from: dict[Cell, tuple[Cell, Action]] = {}
    h0 = manhattan(start, goal)
    heapq.heappush(open_heap, (float(h0), counter, start))
    counter += 1
    closed: set[Cell] = set()
    expansions = 0

    while open_heap and expansions < max_expansions:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expansions += 1
        if current == goal:
            return _reconstruct(came_from, current)
        cur_g = g_score[current]
        prev_dir = came_from[current][1] if current in came_from else None
        for action, (dx, dy) in DIR_VECTORS.items():
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in closed or nxt in blocked:
                continue
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            step_cost = 1.0 + (turn_penalty if prev_dir is not None and action != prev_dir else 0.0)
            tentative = cur_g + step_cost
            if tentative < g_score.get(nxt, math.inf):
                g_score[nxt] = tentative
                came_from[nxt] = (current, action)
                heapq.heappush(open_heap, (tentative + manhattan(nxt, goal), counter, nxt))
                counter += 1
    return None


def _reconstruct(came_from: dict[Cell, tuple[Cell, Action]], end: Cell) -> list[Cell]:
    path = [end]
    while end in came_from:
        end = came_from[end][0]
        path.append(end)
    path.reverse()
    return path


def route_is_survivable(
    path: list[Cell],
    blocked: set[Cell],
    width: int,
    height: int,
    snake_length: int,
    flood_limit: int = 900,
) -> bool:
    """True if following ``path`` still leaves room to escape at its end.

    Approximation: flood-fill from the path's final cell with the path
    interior treated as occupied by our future body. If the reachable pocket
    is smaller than the snake, the food is a bait trap corridor.
    """
    if len(path) < 2:
        return True
    interior = set(path[1:-1])
    area = flood_fill_count(path[-1], blocked | interior, width, height, flood_limit)
    return area >= snake_length


class AStarAgent(BaseAgent):
    """Food-targeting agent powered by real A* search and escape checks."""

    name = "astar"

    def __init__(
        self,
        turn_penalty: float = 0.02,
        risk_aversion: float = 0.6,
        value_weight: float = 1.0,
        flood_limit: int = 900,
        survival_escape_ratio: float = 1.1,
    ) -> None:
        self.turn_penalty = turn_penalty
        self.risk_aversion = risk_aversion
        self.value_weight = value_weight
        self.flood_limit = flood_limit
        self.survival_escape_ratio = survival_escape_ratio

    # -- scoring -------------------------------------------------------------

    def score_food(self, food: "Food", view: StateView, hints: StrategyHints) -> float:
        """Documented utility: value over distance, minus risk and urgency.

        U(f) = w_p * value / (1 + dist) / 10
               - w_r * risk(f)
               - 0.3 * (1 if food expires before we can arrive)
        where w_p follows the strategy's food_priority and
        w_r = risk_aversion * (1 - risk_tolerance).
        """
        dist = manhattan(food.pos, view.me.head)
        w_p = 0.35 + 1.3 * hints.food_priority
        w_r = self.risk_aversion * (1.0 - hints.risk_tolerance)
        utility = w_p * (food.value / (1.0 + dist)) / 10.0
        risk = {"none": 0.0, "low": 0.15, "medium": 0.4, "high": 0.75, "very_high": 1.0}
        utility -= w_r * risk.get(food.risk, 0.4)
        remaining = food.lifetime_ticks - (view.state.tick - food.spawned_tick)
        if remaining < dist:
            utility -= 0.3
        return utility

    # -- decision ------------------------------------------------------------

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
        blocked = set(view.blocked)

        candidates: list[tuple[float, Food]] = []
        for food in state.foods:
            if food.value < 0:
                continue  # never target trap food
            candidates.append((self.score_food(food, view, hints), food))
        candidates.sort(key=lambda pair: pair[0], reverse=True)

        # LLM may pin an explicit target; try it first when reachable.
        if hints.target_food_id:
            candidates.sort(key=lambda pair: (pair[1].id != hints.target_food_id, -pair[0]))

        notes: list[str] = []
        chosen_path: list[Cell] | None = None
        chosen_food: Food | None = None
        best_score = -math.inf

        for score, food in candidates:
            path = astar_search(
                me.head, food.pos, state.width, state.height, blocked, self.turn_penalty
            )
            if path is None:
                continue
            if not route_is_survivable(
                path, blocked, state.width, state.height, me.length, self.flood_limit
            ):
                notes.append(f"food {food.type_name}@{food.pos} reachable but pocket too small")
                continue
            chosen_path, chosen_food, best_score = path, food, score
            break

        if chosen_path is None and hints.target_food_id:
            # Retry without the pinned target (already covered above) - nothing
            # extra to do; fall through to survival behaviour.
            notes.append("LLM-pinned target unreachable; falling back")

        # Survival fallback: chase our own vacating tail (classic safe circling).
        if chosen_path is None:
            tail_goal = me.tail
            tail_blocked = blocked - {me.tail}
            path = astar_search(
                me.head, tail_goal, state.width, state.height, tail_blocked, self.turn_penalty
            )
            if path and len(path) > 1:
                chosen_path = path
                notes.append("survival: tail-chase path")
                chosen_food = None

        computation_ms = (time.perf_counter() - t0) * 1000.0

        if chosen_path and len(chosen_path) > 1:
            nxt = chosen_path[1]
            action = _action_towards(me.head, nxt)
            return AgentDecision(
                agent=self.name,
                action=action,
                path=chosen_path,
                target_food_id=chosen_food.id if chosen_food else None,
                target_pos=chosen_food.pos if chosen_food else None,
                notes=notes,
                diagnostics={
                    "best_score": best_score,
                    "food_scores": {f.id: round(s, 3) for s, f in candidates[:5]},
                    "mode": "food" if chosen_food else "survival",
                },
                computation_ms=computation_ms,
            )

        # No path at all: drift toward the best food to stay useful; the safety
        # layer will override if this is fatal.
        if candidates:
            _, food = candidates[0]
            action = _action_towards(me.head, food.pos)
            notes.append("no safe route found; steering toward best food")
            return AgentDecision(
                agent=self.name,
                action=action,
                path=[me.head, food.pos],
                target_food_id=food.id,
                target_pos=food.pos,
                notes=notes,
                diagnostics={"mode": "desperate", "food_scores": {}},
                computation_ms=computation_ms,
            )

        action = me.direction
        notes.append("no food and no path; holding direction")
        return AgentDecision(
            agent=self.name,
            action=action,
            notes=notes,
            diagnostics={"mode": "hold"},
            computation_ms=computation_ms,
        )


def _action_towards(a: Cell, b: Cell) -> Action:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) >= abs(dy):
        return Action.RIGHT if dx > 0 else Action.LEFT
    return Action.DOWN if dy > 0 else Action.UP
