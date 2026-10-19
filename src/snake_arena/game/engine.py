"""GameEngine: the authoritative simulator for Snake Arena.

One transition function is used by interactive play, RL training, and
benchmarking. Movement is simultaneous for all snakes; collision resolution
is tail-aware (a tail without pending growth vacates its cell this tick) and
head-to-head aware (same-cell and mutual-swap both count as head-on).
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..config import AppConfig
from .collision import future_body
from .food import Food, FoodCatalog, FoodStats, FoodSystem
from .state import GameState
from .snake import Snake
from .types import VALID_ACTIONS, Action, Cell

log = logging.getLogger("snake_arena.engine")


@dataclass
class SnakeSpec:
    """Declarative snake placement used to build a game."""

    id: str
    name: str
    is_human: bool
    start: Cell
    direction: Action = Action.RIGHT
    length: int = 4


@dataclass
class GameEvent:
    kind: str  # "eat" | "trap" | "death" | "expire" | "reversal_ignored"
    snake_id: str = ""
    food_id: str | None = None
    food_type: str | None = None
    value: int = 0
    reason: str | None = None


@dataclass
class StepResult:
    events: list[GameEvent] = field(default_factory=list)
    done: bool = False
    winners: list[str] = field(default_factory=list)
    outcome: str = "ongoing"  # ongoing | survivor_win | draw | death | timeout
    expired: list[Food] = field(default_factory=list)

    @property
    def eats(self) -> list[GameEvent]:
        return [e for e in self.events if e.kind in ("eat", "trap")]


class GameEngine:
    """Simulates the arena. Deterministic for a fixed seed."""

    def __init__(
        self,
        width: int,
        height: int,
        moves_per_second: int = 10,
        max_foods: int = 4,
        seed: int | None = None,
        time_limit_seconds: int = 300,
        catalog: FoodCatalog | None = None,
        foods_path: Path | None = None,
        snake_specs: Iterable[SnakeSpec] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.moves_per_second = max(1, moves_per_second)
        self.max_foods = max(0, max_foods)
        self.time_limit_seconds = time_limit_seconds
        self.base_seed = seed
        self.catalog = catalog if catalog is not None else FoodCatalog.load(foods_path)
        self.food_system = FoodSystem(self.catalog, self.max_foods, self.moves_per_second)
        self.food_stats = FoodStats()
        self.specs: list[SnakeSpec] = []
        if snake_specs:
            self.specs = list(snake_specs)
        else:
            self.specs = [
                SnakeSpec(
                    id="ai",
                    name="AI",
                    is_human=False,
                    start=(max(2, width // 4), height // 2),
                    direction=Action.RIGHT,
                )
            ]
        self.game_id = uuid.uuid4().hex[:10]
        self.rng = random.Random(seed)
        self.state = self._build_state(self.rng, seed)
        self.finished = False
        self.winner_ids: list[str] = []
        self.outcome = "ongoing"

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, config: AppConfig, snake_specs: Iterable[SnakeSpec] | None = None, seed: int | None = None) -> "GameEngine":
        resolved_seed = seed if seed is not None else config.game.seed
        return cls(
            width=config.board.width,
            height=config.board.height,
            moves_per_second=config.game.moves_per_second,
            max_foods=config.game.max_foods,
            seed=resolved_seed,
            time_limit_seconds=config.game.time_limit_seconds,
            snake_specs=snake_specs,
        )

    def _build_state(self, rng: random.Random, seed: int | None) -> GameState:
        state = GameState(
            width=self.width,
            height=self.height,
            tick=0,
            max_foods=self.max_foods,
            moves_per_second=self.moves_per_second,
            max_ticks=max(60, self.time_limit_seconds * self.moves_per_second),
            rng=rng,
            game_id=self.game_id,
        )
        for spec in self.specs:
            state.snakes[spec.id] = Snake(
                id=spec.id,
                name=spec.name,
                is_human=spec.is_human,
                body=Snake.starting_body(spec.start, spec.direction, spec.length),
                direction=spec.direction,
            )
        self.food_system.initialize(state, rng)
        return state

    def reset(self, seed: int | None = None) -> GameState:
        """Rebuild a fresh episode. A new seed changes the food randomness."""
        self.game_id = uuid.uuid4().hex[:10]
        self.rng = random.Random(seed)
        if seed is not None:
            self.base_seed = seed
        self.food_stats = FoodStats()
        self.state = self._build_state(self.rng, seed)
        self.finished = False
        self.winner_ids = []
        self.outcome = "ongoing"
        return self.state

    # -- simulation ---------------------------------------------------------

    @property
    def max_ticks(self) -> int:
        return max(60, self.time_limit_seconds * self.moves_per_second)

    def step(self, actions: dict[str, Action] | None = None) -> StepResult:
        """Advance the simulation by exactly one tick.

        ``actions`` maps snake id -> Action; missing/invalid entries keep the
        snake's current direction; reversal attempts are ignored (the snake
        keeps moving forward) which enforces the no-self-reversal rule at the
        engine level for humans and agents alike.
        """
        actions = actions or {}
        if self.finished:
            return StepResult(events=[], done=True, winners=self.winner_ids, outcome=self.outcome)

        state = self.state
        state.tick += 1
        events: list[GameEvent] = []
        alive = state.living_snakes()

        # 1. Resolve each snake's intended action.
        chosen: dict[str, Action] = {}
        for snake in alive:
            act = actions.get(snake.id)
            if act is None or act not in VALID_ACTIONS:
                act = snake.direction
            if snake.is_reversal(act):
                events.append(GameEvent("reversal_ignored", snake.id))
                act = snake.direction
            chosen[snake.id] = act

        candidates = {s.id: s.next_head(chosen[s.id]) for s in alive}
        growth_preview = {
            s.id: (food.growth if (food := state.food_at(candidates[s.id])) is not None else 0)
            for s in alive
        }

        # 2. Wall deaths.
        dead: dict[str, str] = {}
        for snake in alive:
            if not state.is_inside(candidates[snake.id]):
                dead[snake.id] = "wall"
        survivors = [s for s in alive if s.id not in dead]

        # 3. Head-on deaths: identical destination, or mutual head swap.
        for i in range(len(survivors)):
            for j in range(i + 1, len(survivors)):
                a, b = survivors[i], survivors[j]
                same_cell = candidates[a.id] == candidates[b.id]
                swap = candidates[a.id] == b.head and candidates[b.id] == a.head
                if same_cell or swap:
                    dead[a.id] = dead[b.id] = "head_on"
        survivors = [s for s in survivors if s.id not in dead]

        # 4. Body collisions, resolved simultaneously with tail awareness.
        futures = {
            s.id: future_body(s, candidates[s.id], growth_preview[s.id]) for s in survivors
        }
        for snake in survivors:
            own_rest = set(futures[snake.id][1:])
            if candidates[snake.id] in own_rest:
                dead[snake.id] = "self"
        for snake in survivors:
            if snake.id in dead:
                continue
            for other in survivors:
                if other.id == snake.id:
                    continue
                if candidates[snake.id] in set(futures[other.id]):
                    dead[snake.id] = "opponent"
        survivors = [s for s in survivors if s.id not in dead]

        # 5. Commit movement and food consumption for surviving snakes.
        for snake in survivors:
            food = state.food_at(candidates[snake.id])
            snake.commit_move(candidates[snake.id], growth_preview[snake.id])
            snake.steps_since_food += 1
            if food is not None:
                state.foods.remove(food)
                snake.score += food.value
                snake.steps_since_food = 0
                snake.foods_eaten[food.type_name] += 1
                self.food_stats.record(snake.id, food.type_name)
                events.append(
                    GameEvent(
                        "trap" if food.value < 0 else "eat",
                        snake.id,
                        food.id,
                        food.type_name,
                        food.value,
                    )
                )

        # 6. Apply deaths.
        for snake_id, reason in dead.items():
            snake = state.snakes[snake_id]
            snake.alive = False
            snake.death_reason = reason
            snake.death_tick = state.tick
            events.append(GameEvent("death", snake_id, reason=reason))

        # 7. Expire and refill food.
        expired = self.food_system.maintain(state, self.rng)
        for food in expired:
            events.append(GameEvent("expire", food_id=food.id, food_type=food.type_name))

        # 8. Termination checks.
        result = StepResult(events=events, expired=expired)
        alive_after = state.living_snakes()
        if dead:
            result.done = True
            if len(alive_after) == 1 and len(self.specs) > 1:
                result.winners = [alive_after[0].id]
                result.outcome = "survivor_win"
            elif len(alive_after) == 1:
                result.winners = [alive_after[0].id]
                result.outcome = "survivor_win"
            elif not alive_after:
                result.outcome = "draw" if len(self.specs) > 1 else "death"
        elif state.tick >= self.max_ticks:
            result.done = True
            result.outcome = "timeout"
            if alive_after:
                best = max(s.score for s in alive_after)
                result.winners = [s.id for s in alive_after if s.score == best]
                if len(result.winners) == len(alive_after) and len(alive_after) > 1:
                    result.outcome = "draw"

        if result.done:
            self.finished = True
            self.winner_ids = list(result.winners)
            self.outcome = result.outcome
        return result

    # -- convenience introspection ------------------------------------------

    def human_snake(self) -> Snake | None:
        for snake in self.state.snakes.values():
            if snake.is_human:
                return snake
        return None

    def ai_snake(self) -> Snake | None:
        for snake in self.state.snakes.values():
            if not snake.is_human and snake.alive:
                return snake
        return None

    def elapsed_seconds(self) -> float:
        return self.state.seconds_elapsed
