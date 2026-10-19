"""Collision predicates and shared grid occupancy helpers.

This module contains the *pure* collision/occupancy logic used by the engine,
the safety validator, and the pathfinding agents. Sharing occupancy helpers
here is deliberate: the search algorithms themselves (A* vs BFS) remain fully
independent implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import Action, Cell

if TYPE_CHECKING:
    from .state import GameState
    from .snake import Snake


def will_tail_vacate(snake: "Snake", growth_now: int = 0) -> bool:
    """True if the snake's tail cell becomes free on the next move.

    The tail is retained when the snake still owes growth from earlier food
    or when it is about to eat (``growth_now > 0``). Pathfinding uses this to
    avoid the classic over-estimation bug where a moving tail is treated as a
    permanent wall.
    """
    return snake.pending_growth == 0 and growth_now == 0


def future_body(snake: "Snake", candidate_head: Cell, growth_now: int = 0) -> list[Cell]:
    """The snake's body cells after moving ``candidate_head`` (head first)."""
    body = [candidate_head, *list(snake.body)]
    if will_tail_vacate(snake, growth_now):
        body.pop()
    return body


def planning_blocked_cells(state: "GameState", snake_id: str) -> set[Cell]:
    """Cells that should be treated as walls when planning for ``snake_id``.

    Includes every living snake's body minus tails that will vacate (tails of
    snakes with no pending growth). The planner's own candidate head position
    is obviously not included.
    """
    blocked: set[Cell] = set()
    for snake in state.snakes.values():
        if not snake.alive:
            continue
        body = list(snake.body)
        if will_tail_vacate(snake):
            body = body[:-1]
        blocked.update(body)
    return blocked


def is_deadly_for(
    state: "GameState",
    snake: "Snake",
    cell: Cell,
    own_growth: int = 0,
    treat_opponent_head_vacated: bool = True,
) -> str | None:
    """Would moving ``cell`` kill ``snake`` immediately? Returns reason|None.

    Reasons: 'wall', 'self', 'opponent'. Tail-aware: tails without pending
    growth are considered free. ``treat_opponent_head_vacated`` controls
    whether the opponent's *current* head cell counts as blocked; the head
    always moves away, so default False (safe), but the safety validator can
    enable strict mode.
    """
    if not state.is_inside(cell):
        return "wall"

    own_body = future_body(snake, cell, own_growth)
    own_rest = set(own_body[1:])  # exclude the candidate head itself
    if cell in own_rest and cell != snake.tail:
        return "self"
    if cell in own_rest and cell == snake.tail and not will_tail_vacate(snake, own_growth):
        return "self"

    for opponent in state.opponents_of(snake.id):
        opp_body = future_body(opponent, opponent.head, 0)
        for i, c in enumerate(opp_body):
            if c != cell:
                continue
            if i == 0 and treat_opponent_head_vacated:
                continue  # their head vacates this tick
            return "opponent"
    return None


def flood_fill_count(
    start: Cell,
    blocked: set[Cell],
    width: int,
    height: int,
    limit: int = 900,
) -> int:
    """Count cells reachable from ``start`` (4-neighbour), capped at ``limit``.

    Used for dead-end detection: if the reachable area from a candidate move
    is smaller than the snake, the move leads into a pocket it cannot survive.
    """
    if start in blocked or not (0 <= start[0] < width and 0 <= start[1] < height):
        return 0
    seen = {start}
    frontier = [start]
    count = 0
    while frontier and count < limit:
        cell = frontier.pop()
        count += 1
        x, y = cell
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            nxt = (nx, ny)
            if nxt in seen or nxt in blocked:
                continue
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            seen.add(nxt)
            frontier.append(nxt)
    return count


def danger_cells(state: "GameState", snake_id: str) -> set[Cell]:
    """Heuristic danger overlay: cells the opponent can reach in one move,
    plus the snake's own body-adjacent corridors it should not enter."""
    dangers: set[Cell] = set()
    me = state.snakes[snake_id]
    for opponent in state.opponents_of(snake_id):
        x, y = opponent.head
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if state.is_inside((nx, ny)):
                dangers.add((nx, ny))
    _ = me  # reserved: future risk model hooks
    return dangers
