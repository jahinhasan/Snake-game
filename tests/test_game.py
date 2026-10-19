"""Engine mechanics: movement, growth, deaths, food scoring, determinism."""

from __future__ import annotations

from collections import deque

from snake_arena.game.engine import SnakeSpec
from snake_arena.game.snake import Snake
from snake_arena.game.types import Action

from .conftest import make_engine


def test_movement_and_growth():
    engine = make_engine(seed=1, max_foods=0)
    snake = engine.state.snakes["ai"]
    start_len = snake.length
    head0 = snake.head

    snake.commit_move((head0[0] + 1, head0[1]), growth_now=0)
    assert snake.head == (head0[0] + 1, head0[1])
    assert snake.length == start_len  # no growth: tail moved too

    # growth=2 is owed gradually: one tail retained per tick.
    snake.commit_move((head0[0] + 2, head0[1]), growth_now=2)
    assert snake.length == start_len + 1
    assert snake.pending_growth == 1
    snake.commit_move((head0[0] + 3, head0[1]), growth_now=0)
    assert snake.length == start_len + 2
    assert snake.pending_growth == 0


def test_reversal_is_ignored():
    engine = make_engine(seed=2, max_foods=0)
    snake = engine.state.snakes["ai"]  # facing RIGHT
    result = engine.step({"ai": Action.LEFT})
    kinds = [e.kind for e in result.events]
    assert "reversal_ignored" in kinds
    assert snake.direction == Action.RIGHT  # still moving right


def test_wall_death():
    snakes = [SnakeSpec("ai", "AI", False, (13, 6), Action.RIGHT)]
    engine = make_engine(seed=3, width=16, height=12, snakes=snakes, max_foods=0)
    steps = 0
    while not engine.finished and steps < 10:
        engine.step({"ai": Action.RIGHT})
        steps += 1
    assert engine.finished
    assert engine.outcome == "death"
    snake = engine.state.snakes["ai"]
    assert not snake.alive
    assert snake.death_reason == "wall"


def test_self_collision_death():
    engine = make_engine(seed=4, width=8, height=8, max_foods=0)
    snake = engine.state.snakes["ai"]
    # Head (4,4) facing RIGHT (body[1] behind it); the cell ahead, (5,4),
    # holds a mid-body segment -> stepping RIGHT is self-collision.
    snake.body.clear()
    snake.body.extend([(4, 4), (3, 4), (5, 4), (3, 5)])
    snake.direction = Action.RIGHT
    engine.step({"ai": Action.RIGHT})
    assert engine.finished
    assert not snake.alive
    assert snake.death_reason == "self"


def test_head_on_same_cell_both_die():
    snakes = [
        SnakeSpec("a", "A", False, (4, 5), Action.RIGHT, length=3),
        SnakeSpec("b", "B", False, (6, 5), Action.LEFT, length=3),
    ]
    engine = make_engine(seed=5, snakes=snakes, max_foods=0)
    engine.step({"a": Action.RIGHT, "b": Action.LEFT})  # both enter (5,5)
    assert engine.finished
    assert engine.outcome == "draw"
    assert all(not s.alive for s in engine.state.snakes.values())


def test_head_on_swap_both_die():
    snakes = [
        SnakeSpec("a", "A", False, (5, 5), Action.RIGHT, length=3),
        SnakeSpec("b", "B", False, (6, 5), Action.LEFT, length=3),
    ]
    engine = make_engine(seed=6, snakes=snakes, max_foods=0)
    engine.step({"a": Action.RIGHT, "b": Action.LEFT})  # pass through each other
    assert engine.finished
    assert engine.outcome == "draw"


def test_tail_vacates_so_following_is_safe():
    snake = Snake(id="s", name="s", is_human=False,
                  body=deque([(1, 0), (2, 0), (3, 0)]),
                  direction=Action.RIGHT)
    from snake_arena.game.collision import future_body, will_tail_vacate

    assert will_tail_vacate(snake)
    body = future_body(snake, candidate_head=(0, 0))
    assert (3, 0) not in body  # tail moved out of the way
    assert (0, 0) in body  # new head included


def test_food_scoring_matches_events():
    engine = make_engine(seed=42, max_foods=4)
    snake = engine.state.snakes["ai"]
    # Plant a known food on the snake's path so eating is deterministic.
    from snake_arena.game.food import Food

    engine.state.foods.append(Food(id="planted", type_name="golden", pos=(8, 6),
                                   value=50, growth=1, risk="high",
                                   spawned_tick=0, lifetime_ticks=10_000))
    expected = 0
    ate = False
    for i in range(60):
        if engine.finished:
            break
        result = engine.step({"ai": Action.RIGHT})
        for event in result.events:
            if event.kind in ("eat", "trap"):
                expected += event.value
                ate = True
    assert ate, "planted food on the path must be eaten"
    assert snake.score == expected
    assert snake.score >= 50  # the planted golden food alone


def test_determinism_same_seed():
    def play(seed):
        engine = make_engine(seed=seed, max_foods=4)
        script = [Action.UP, Action.RIGHT, Action.RIGHT, Action.DOWN, Action.DOWN,
                  Action.LEFT, Action.DOWN, Action.RIGHT]
        for i in range(60):
            if engine.finished:
                break
            engine.step({"ai": script[i % len(script)]})
        return engine.state.tick, engine.state.snakes["ai"].score, len(engine.state.foods)

    assert play(123) == play(123)


def test_reset_returns_to_initial():
    engine = make_engine(seed=9, max_foods=4)
    for _ in range(20):
        engine.step({"ai": Action.UP})
    state = engine.reset(seed=9)
    assert engine.state.tick == 0
    assert engine.state.snakes["ai"].score == 0
    assert len(state.foods) > 0
