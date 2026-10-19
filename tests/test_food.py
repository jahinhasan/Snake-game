"""Food catalog, spawning, lifetimes, and statistics."""

from __future__ import annotations

import random

from snake_arena.game.food import Food, FoodCatalog, FoodSystem, FoodStats
from snake_arena.game.types import Action
from snake_arena.utils import paths

from .conftest import make_engine


def test_catalog_loads_five_types_with_valid_probabilities():
    catalog = FoodCatalog.load(paths.default_foods_path())
    assert len(catalog.types) == 5
    types = list(catalog.types.values())
    assert abs(sum(catalog.probabilities) - 1.0) < 1e-6  # normalized for spawning
    values = {name: t.value for name, t in catalog.types.items()}
    assert values["normal"] == 10
    assert values["bonus"] == 25
    assert values["golden"] == 50
    assert values["special"] == 100
    assert values["trap"] == -20


def test_weighted_choice_stays_in_catalog():
    catalog = FoodCatalog.load()
    rng = random.Random(0)
    draws = {catalog.weighted_choice(rng).name for _ in range(500)}
    assert draws <= set(catalog.types.keys())
    assert "normal" in draws  # 55% probability: certain to appear in 500 draws


def test_engine_spawns_initial_food():
    engine = make_engine(seed=11, max_foods=4)
    assert len(engine.state.foods) == 4
    for food in engine.state.foods:
        assert engine.state.is_inside(food.pos)
        assert food.lifetime_ticks > 0


def test_food_expires_and_is_replaced():
    engine = make_engine(seed=12, max_foods=2)
    state = engine.state
    for food in state.foods:
        # Pretend each item spawned long ago with a short lifetime.
        food.spawned_tick = -10
        food.lifetime_ticks = 5
    assert all(f.is_expired(1) for f in state.foods)
    result = engine.step({"ai": Action.RIGHT})
    assert any(e.kind == "expire" for e in result.events)
    assert len(state.foods) == 2  # maintain() replaced the expired items
    assert all(not f.is_expired(state.tick) for f in state.foods)


def test_remaining_ratio_monotonic():
    food = Food(id="f", type_name="normal", pos=(1, 1), value=10, growth=1,
                risk="low", spawned_tick=0, lifetime_ticks=10)
    assert food.remaining_ratio(0) == 1.0
    assert food.remaining_ratio(5) == 0.5
    assert food.remaining_ratio(10) == 0.0
    assert food.remaining_ratio(99) == 0.0


def test_spawn_skips_when_board_is_full():
    catalog = FoodCatalog.load()
    engine = make_engine(seed=13, max_foods=0, width=6, height=4)
    system = FoodSystem(catalog, max_foods=1, moves_per_second=10)
    rng = random.Random(3)
    positions = [(x, y) for x in range(6) for y in range(4)]
    for i, pos in enumerate(positions):
        engine.state.foods.append(Food(id=f"f{i}", type_name="normal", pos=pos,
                                       value=10, growth=1, risk="low",
                                       spawned_tick=0, lifetime_ticks=1000))
    assert system.spawn_one(engine.state, rng) is None


def test_food_stats_counts():
    stats = FoodStats()
    stats.record("a", "normal")
    stats.record("a", "normal")
    stats.record("a", "golden")
    stats.record("b", "trap")
    assert stats.total() == 4
    assert stats.by_snake_type["a"]["normal"] == 2
    assert stats.by_snake_type["a"]["golden"] == 1
    assert stats.by_snake_type["b"]["trap"] == 1
