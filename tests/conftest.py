"""Shared test fixtures: import shim, default config, engine factories."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402

from snake_arena.config import AppConfig  # noqa: E402
from snake_arena.game.engine import GameEngine, SnakeSpec  # noqa: E402
from snake_arena.game.types import Action  # noqa: E402


@pytest.fixture
def config() -> AppConfig:
    cfg = AppConfig()
    cfg.llm.enabled = False  # tests never touch the network
    return cfg


def make_engine(
    seed: int | None = 7,
    width: int = 16,
    height: int = 12,
    snakes: list[SnakeSpec] | None = None,
    max_foods: int = 4,
    time_limit_seconds: int = 60,
) -> GameEngine:
    if snakes is None:
        snakes = [SnakeSpec("ai", "AI", False, (4, 6), Action.RIGHT)]
    return GameEngine(
        width=width,
        height=height,
        moves_per_second=10,
        max_foods=max_foods,
        seed=seed,
        time_limit_seconds=time_limit_seconds,
        snake_specs=snakes,
    )


@pytest.fixture
def solo_engine():
    return make_engine


def make_env(seed: int | None = None, width: int = 16, height: int = 12,
             max_foods: int = 3, max_steps: int = 300):
    """RL training environment (wraps the engine with observations)."""
    from snake_arena.game.environment import SnakeEnv

    return SnakeEnv(width=width, height=height, max_foods=max_foods,
                    max_steps=max_steps, seed=seed)


@pytest.fixture
def solo_env():
    return make_env
