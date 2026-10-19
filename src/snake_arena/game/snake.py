"""Snake entity: body bookkeeping, growth, and per-snake statistics."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .types import Action, Cell, action_delta, direction_from_cells


@dataclass
class Snake:
    """One snake in the arena.

    ``body`` is a deque of cells ordered head-first. ``pending_growth`` counts
    segments still owed by eaten food: while positive, the tail does not
    vacate on the next step (which matters for collision and pathfinding).
    """

    id: str
    name: str
    is_human: bool
    body: deque[Cell]
    direction: Action
    alive: bool = True
    score: int = 0
    pending_growth: int = 0
    death_reason: str | None = None
    death_tick: int | None = None
    steps_since_food: int = 0
    foods_eaten: Counter = field(default_factory=Counter)
    rejected_moves: int = 0  # safety-checker interventions (diagnostics)

    @property
    def head(self) -> Cell:
        return self.body[0]

    @property
    def tail(self) -> Cell:
        return self.body[-1]

    @property
    def length(self) -> int:
        return len(self.body)

    def next_head(self, action: Action) -> Cell:
        dx, dy = action_delta(action)
        hx, hy = self.head
        return (hx + dx, hy + dy)

    def is_reversal(self, action: Action) -> bool:
        """True if the action would move the head onto the neck cell.

        A snake of length 1 has no neck, so any action is legal for it.
        """
        if self.length < 2:
            return False
        neck = self.body[1]
        return self.next_head(action) == neck

    def commit_move(self, new_head: Cell, growth_now: int = 0) -> None:
        """Move the head to ``new_head``.

        ``growth_now`` is the growth granted by food on the destination cell
        this tick. The tail is retained (snake grows by one) for any tick
        where owed growth remains; food adds its growth to the owed counter
        first, so growth=2 keeps the tail this tick *and* the next.
        """
        self.pending_growth += max(0, growth_now)
        self.body.appendleft(new_head)
        if self.pending_growth > 0:
            self.pending_growth -= 1  # tail retained this tick
        else:
            self.body.pop()
        if self.length > 1:
            self.direction = direction_from_cells(self.body[1], self.head)

    def body_set(self) -> set[Cell]:
        return set(self.body)

    @staticmethod
    def starting_body(start: Cell, direction: Action, length: int = 4) -> deque[Cell]:
        """Body cells for a snake of ``length`` with its head at ``start``."""
        dx, dy = action_delta(direction)
        cells: deque[Cell] = deque()
        for i in range(length):
            cells.append((start[0] - dx * i, start[1] - dy * i))
        return cells
