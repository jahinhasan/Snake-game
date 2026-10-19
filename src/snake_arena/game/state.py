"""GameState: the authoritative, UI-free representation of the arena.

The same state transitions power interactive play, RL training, and
benchmarking, so experiments can never drift from real game rules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

from .food import Food
from .snake import Snake
from .types import Cell


@dataclass
class GameState:
    width: int
    height: int
    snakes: dict[str, Snake] = field(default_factory=dict)
    foods: list[Food] = field(default_factory=list)
    tick: int = 0
    max_foods: int = 4
    moves_per_second: int = 10
    max_ticks: int = 3000
    rng: random.Random = field(default_factory=random.Random)
    game_id: str = ""

    # -- grid queries -----------------------------------------------------

    def is_inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def in_bounds_iterator(self) -> Iterator[Cell]:
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y)

    def snake_at(self, cell: Cell) -> Snake | None:
        for snake in self.snakes.values():
            if snake.alive and cell in snake.body:
                return snake
        return None

    def food_at(self, cell: Cell) -> Food | None:
        for food in self.foods:
            if food.pos == cell:
                return food
        return None

    def occupied_cells(self, include_food: bool = False) -> set[Cell]:
        """All cells currently occupied by living snakes (optionally food)."""
        cells: set[Cell] = set()
        for snake in self.snakes.values():
            if snake.alive:
                cells.update(snake.body)
        if include_food:
            cells.update(f.pos for f in self.foods)
        return cells

    def free_cell_count(self) -> int:
        return self.width * self.height - len(self.occupied_cells(include_food=True))

    def living_snakes(self) -> list[Snake]:
        return [s for s in self.snakes.values() if s.alive]

    def opponents_of(self, snake_id: str) -> list[Snake]:
        return [s for s in self.snakes.values() if s.id != snake_id and s.alive]

    def get_snake(self, snake_id: str) -> Snake:
        return self.snakes[snake_id]

    @property
    def seconds_elapsed(self) -> float:
        return self.tick / max(1, self.moves_per_second)

    @property
    def board_area(self) -> int:
        return self.width * self.height
