"""Multi-food system: food types, items, catalog loading, spawning.

Five configurable food types (normal / bonus / golden / special / trap) each
carry a score value, a risk tier, a spawn probability and a lifetime. Spawn
probabilities are normalized at load time so malformed configs stay safe.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import Cell

if TYPE_CHECKING:  # circular-import guard: state only needed for typing
    from .state import GameState

log = logging.getLogger("snake_arena.food")

FOOD_SCHEMA_VERSION = 1

# Numeric risk tier used by agents and metrics (0 = harmless, 1 = riskiest).
RISK_SCORES: dict[str, float] = {
    "none": 0.0,
    "low": 0.15,
    "medium": 0.4,
    "high": 0.75,
    "very_high": 1.0,
}


@dataclass
class FoodType:
    name: str
    display_name: str
    value: int
    risk: str
    probability: float
    lifetime_seconds: tuple[float, float]
    growth: int
    color: str
    shape: str
    description: str = ""

    @property
    def risk_score(self) -> float:
        return RISK_SCORES.get(self.risk, 0.5)


@dataclass
class Food:
    """A single food item on the board."""

    id: str
    type_name: str
    pos: Cell
    value: int
    growth: int
    risk: str
    spawned_tick: int
    lifetime_ticks: int
    color: str = "#ffffff"
    shape: str = "circle"

    @property
    def remaining_ticks(self) -> int:
        return self.lifetime_ticks

    def is_expired(self, tick: int) -> bool:
        return tick - self.spawned_tick >= self.lifetime_ticks

    def remaining_ratio(self, tick: int) -> float:
        """1.0 right after spawn, 0.0 at expiry (used for visual blinking)."""
        if self.lifetime_ticks <= 0:
            return 0.0
        return max(0.0, 1.0 - (tick - self.spawned_tick) / self.lifetime_ticks)


class FoodCatalog:
    """Validated set of food types with normalized spawn probabilities."""

    def __init__(self, types: list[FoodType]) -> None:
        if not types:
            raise ValueError("FoodCatalog needs at least one food type")
        self.types: dict[str, FoodType] = {t.name: t for t in types}
        self._names = [t.name for t in types]
        raw = [max(0.0, float(t.probability)) for t in types]
        total = sum(raw)
        if total <= 0:
            log.warning("All food probabilities were <= 0; using uniform distribution.")
            raw = [1.0] * len(types)
            total = float(len(types))
        self.probabilities = [p / total for p in raw]

    def __contains__(self, name: str) -> bool:
        return name in self.types

    def get(self, name: str) -> FoodType:
        return self.types[name]

    def weighted_choice(self, rng) -> FoodType:
        roll = rng.random()
        cumulative = 0.0
        for name, prob in zip(self._names, self.probabilities):
            cumulative += prob
            if roll < cumulative:
                return self.types[name]
        return self.types[self._names[-1]]

    @classmethod
    def load(cls, path: Path | None = None) -> "FoodCatalog":
        """Load foods.json with schema validation and safe fallbacks."""
        food_path = Path(path) if path else None
        raw_types: list[dict[str, Any]]
        try:
            if food_path is None:
                from ..utils import paths

                food_path = paths.default_foods_path()
            data = json.loads(food_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("types"), list):
                raise ValueError("foods.json must contain a 'types' list")
            raw_types = data["types"]
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            log.error("Failed to load food config (%s); using built-in defaults.", exc)
            raw_types = DEFAULT_FOOD_TYPES

        types: list[FoodType] = []
        for i, item in enumerate(raw_types):
            try:
                types.append(_parse_food_type(item))
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("Skipping invalid food type #%s (%s)", i, exc)
        if not types:
            types = [_parse_food_type(item) for item in DEFAULT_FOOD_TYPES]
        return cls(types)


def _parse_food_type(item: dict[str, Any]) -> FoodType:
    lifetime = item.get("lifetime_seconds", [20, 30])
    if not isinstance(lifetime, (list, tuple)) or len(lifetime) < 2:
        lifetime = [20, 30]
    lo, hi = float(lifetime[0]), float(lifetime[1])
    if lo > hi:
        lo, hi = hi, lo
    lo, hi = max(1.0, lo), max(1.0, hi)
    risk = str(item.get("risk", "low")).lower()
    if risk not in RISK_SCORES:
        risk = "medium"
    return FoodType(
        name=str(item["name"]),
        display_name=str(item.get("display_name", item["name"]).title()),
        value=int(item.get("value", 10)),
        risk=risk,
        probability=float(item.get("probability", 0.2)),
        lifetime_seconds=(lo, hi),
        growth=max(0, int(item.get("growth", 1))),
        color=str(item.get("color", "#7ee787")),
        shape=str(item.get("shape", "circle")),
        description=str(item.get("description", "")),
    )


DEFAULT_FOOD_TYPES: list[dict[str, Any]] = [
    dict(name="normal", display_name="Normal", value=10, risk="low", probability=0.55,
         lifetime_seconds=[25, 45], growth=1, color="#7ee787", shape="circle"),
    dict(name="bonus", display_name="Bonus", value=25, risk="medium", probability=0.20,
         lifetime_seconds=[14, 26], growth=1, color="#58a6ff", shape="diamond"),
    dict(name="golden", display_name="Golden", value=50, risk="high", probability=0.13,
         lifetime_seconds=[9, 16], growth=1, color="#e3b341", shape="star"),
    dict(name="special", display_name="Special", value=100, risk="very_high", probability=0.07,
         lifetime_seconds=[6, 11], growth=2, color="#bc8cff", shape="burst"),
    dict(name="trap", display_name="Trap", value=-20, risk="high", probability=0.05,
         lifetime_seconds=[18, 30], growth=0, color="#f85149", shape="skull"),
]


class FoodSystem:
    """Spawns, expires, and refills food items on the board.

    All randomness flows through the engine's seeded RNG so that episodes are
    reproducible. Spawn always picks uniformly among genuinely free cells; if
    the board is full the spawn is skipped (never an infinite loop).
    """

    def __init__(self, catalog: FoodCatalog, max_foods: int, moves_per_second: int) -> None:
        self.catalog = catalog
        self.max_foods = max(0, max_foods)
        self.moves_per_second = max(1, moves_per_second)
        self._spawn_counter = 0

    def _new_food(self, ftype: FoodType, pos: Cell, tick: int, rng) -> Food:
        lifetime_seconds = rng.uniform(*ftype.lifetime_seconds)
        lifetime_ticks = max(1, int(round(lifetime_seconds * self.moves_per_second)))
        self._spawn_counter += 1
        return Food(
            id=f"{ftype.name}-{self._spawn_counter:04d}-{uuid.uuid4().hex[:4]}",
            type_name=ftype.name,
            pos=pos,
            value=ftype.value,
            growth=ftype.growth,
            risk=ftype.risk,
            spawned_tick=tick,
            lifetime_ticks=lifetime_ticks,
            color=ftype.color,
            shape=ftype.shape,
        )

    def spawn_one(self, state: "GameState", rng) -> Food | None:
        """Spawn one random food item in a free cell, if any exists."""
        occupied = state.occupied_cells(include_food=True)
        free = [
            (x, y)
            for y in range(state.height)
            for x in range(state.width)
            if (x, y) not in occupied
        ]
        if not free:
            return None  # board full: skip, never loop forever
        pos = free[rng.randrange(len(free))]
        ftype = self.catalog.weighted_choice(rng)
        food = self._new_food(ftype, pos, state.tick, rng)
        state.foods.append(food)
        return food

    def initialize(self, state: "GameState", rng) -> None:
        for _ in range(self.max_foods):
            self.spawn_one(state, rng)

    def maintain(self, state: "GameState", rng) -> list[Food]:
        """Expire old food and refill to ``max_foods``; returns expired items."""
        expired = [f for f in state.foods if f.is_expired(state.tick)]
        if expired:
            state.foods = [f for f in state.foods if not f.is_expired(state.tick)]
        while len(state.foods) < self.max_foods:
            if self.spawn_one(state, rng) is None:
                break
        return expired


@dataclass
class FoodEatenRecord:
    food_id: str
    type_name: str
    value: int
    snake_id: str


class FoodStats:
    """Aggregate food statistics for one episode (per snake and total)."""

    def __init__(self) -> None:
        self.by_type: Counter = Counter()
        self.by_snake_type: dict[str, Counter] = {}

    def record(self, snake_id: str, type_name: str) -> None:
        self.by_type[type_name] += 1
        self.by_snake_type.setdefault(snake_id, Counter())[type_name] += 1

    def total(self) -> int:
        return sum(self.by_type.values())
