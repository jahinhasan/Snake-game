"""Core shared game types: actions, cells, and grid helpers.

Kept import-free from other game modules so that ``snake``, ``food`` and
``state`` can all depend on it without circular imports.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Iterator

# A grid cell is an (x, y) tuple of integers. Tuples are hashable and cheap,
# which makes them ideal for set-based collision and pathfinding code.
Cell = tuple[int, int]


class Action(IntEnum):
    """The complete discrete action space shared by humans and every agent."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


DIR_VECTORS: dict[Action, Cell] = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}

OPPOSITE: dict[Action, Action] = {
    Action.UP: Action.DOWN,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}

ALL_ACTIONS: tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)

VALID_ACTIONS = frozenset(ALL_ACTIONS)


def action_delta(action: Action) -> Cell:
    return DIR_VECTORS[action]


def add_delta(cell: Cell, delta: Cell) -> Cell:
    return (cell[0] + delta[0], cell[1] + delta[1])


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(cell: Cell, width: int, height: int) -> bool:
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def neighbors(cell: Cell) -> Iterator[tuple[Cell, Action]]:
    """Yield (neighbor_cell, action) for the four orthogonal moves."""
    x, y = cell
    for action, (dx, dy) in DIR_VECTORS.items():
        yield (x + dx, y + dy), action


def direction_from_cells(tail: Cell, head: Cell) -> Action:
    """Infer the Action that moves ``tail`` onto ``head`` (adjacent cells)."""
    dx, dy = head[0] - tail[0], head[1] - tail[1]
    for action, (ddx, ddy) in DIR_VECTORS.items():
        if (ddx, ddy) == (dx, dy):
            return action
    return Action.RIGHT  # safe default for zero-length snakes
