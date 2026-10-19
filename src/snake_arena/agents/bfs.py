"""Real breadth-first search on the arena grid plus the BFS snake agent.

BFS explores the grid in waves using a FIFO queue; on a unit-cost grid it
returns a *shortest* path in steps (no cost weights, no heuristic). It shares
only the tail-aware occupancy snapshot with the rest of the system -- the
search itself is an independent implementation from A*, which is exactly what
makes the A*-vs-BFS benchmark comparison meaningful.

The BFS agent targets the *nearest* non-trap food (pure distance baseline),
deliberately ignoring food values so benchmark charts can show what
value-aware targeting (A* agent) buys over plain shortest-path play.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

from ..game.collision import flood_fill_count
from ..game.environment import StateView, build_state_view
from ..game.types import DIR_VECTORS, Action, Cell, manhattan
from .base import AgentDecision, BaseAgent, StrategyHints
from .astar import _action_towards

if TYPE_CHECKING:
    from ..game.food import Food
    from ..game.state import GameState


def bfs_search(
    start: Cell,
    goal: Cell,
    width: int,
    height: int,
    blocked: set[Cell],
    max_nodes: int = 8000,
) -> list[Cell] | None:
    """Breadth-first shortest path from ``start`` to ``goal``; None if none.

    Classic wave expansion: pop from a FIFO queue, mark visited on enqueue
    (prevents duplicate work), reconstruct via parent links.
    """
    if start == goal:
        return [start]
    if start in blocked:
        blocked = blocked - {start}

    queue: deque[Cell] = deque([start])
    parent: dict[Cell, Cell] = {start: start}
    visited: set[Cell] = {start}
    nodes = 0

    while queue and nodes < max_nodes:
        current = queue.popleft()
        nodes += 1
        for dx, dy in DIR_VECTORS.values():
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in visited or nxt in blocked:
                continue
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            visited.add(nxt)
            parent[nxt] = current
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(nxt)
    return None


class BFSAgent(BaseAgent):
    """Shortest-path baseline: nearest non-trap food via real BFS."""

    name = "bfs"

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
        me = view.me
        blocked = set(view.blocked)
        notes: list[str] = []

        positives = [f for f in state.foods if f.value > 0]
        ranked: list[Food] = sorted(positives, key=lambda f: manhattan(f.pos, me.head))

        chosen_path: list[Cell] | None = None
        chosen_food: Food | None = None
        for food in ranked:
            path = bfs_search(me.head, food.pos, state.width, state.height, blocked)
            if path is None:
                continue
            interior = set(path[1:-1])
            area = flood_fill_count(path[-1], blocked | interior, state.width, state.height, self.flood_limit)
            if area < me.length:
                notes.append(f"food {food.type_name}@{food.pos} reachable but pocket too small")
                continue
            chosen_path, chosen_food = path, food
            break

        if chosen_path is None:  # survival fallback: chase our tail
            path = bfs_search(me.head, me.tail, state.width, state.height, blocked - {me.tail})
            if path and len(path) > 1:
                chosen_path = path
                notes.append("survival: tail-chase path")

        computation_ms = (time.perf_counter() - t0) * 1000.0
        if chosen_path and len(chosen_path) > 1:
            return AgentDecision(
                agent=self.name,
                action=_action_towards(me.head, chosen_path[1]),
                path=chosen_path,
                target_food_id=chosen_food.id if chosen_food else None,
                target_pos=chosen_food.pos if chosen_food else None,
                notes=notes,
                diagnostics={"mode": "food" if chosen_food else "survival", "search": "bfs"},
                computation_ms=computation_ms,
            )

        if ranked:
            food = ranked[0]
            notes.append("no safe route; steering toward nearest food")
            return AgentDecision(
                agent=self.name,
                action=_action_towards(me.head, food.pos),
                path=[me.head, food.pos],
                target_food_id=food.id,
                target_pos=food.pos,
                notes=notes,
                diagnostics={"mode": "desperate", "search": "bfs"},
                computation_ms=computation_ms,
            )
        return AgentDecision(
            agent=self.name,
            action=me.direction,
            notes=["nothing to target; holding direction"],
            diagnostics={"mode": "hold", "search": "bfs"},
            computation_ms=computation_ms,
        )
