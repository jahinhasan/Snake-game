"""Safety validator: wall, reversal, self, dead-end, fallback selection."""

from __future__ import annotations

from snake_arena.game.engine import SnakeSpec
from snake_arena.game.types import Action
from snake_arena.safety.validator import SafetyValidator

from .conftest import make_engine


def _engine_head_at(x: int):
    snakes = [SnakeSpec("ai", "AI", False, (x - 3, 6), Action.RIGHT)]
    engine = make_engine(seed=20, width=16, height=12, snakes=snakes, max_foods=0)
    for _ in range(3):
        engine.step({"ai": Action.RIGHT})  # head reaches (x, 6)
    return engine


def test_wall_rejected():
    engine = _engine_head_at(15)
    validator = SafetyValidator()
    report = validator.check(engine.state, "ai", Action.RIGHT)
    assert not report.safe
    assert report.reason == "wall"


def test_reversal_rejected():
    engine = _engine_head_at(8)
    validator = SafetyValidator()
    report = validator.check(engine.state, "ai", Action.LEFT)
    assert not report.safe
    assert report.reason == "reversal"


def test_self_collision_rejected():
    engine = make_engine(seed=21, width=8, height=8, max_foods=0)
    snake = engine.state.snakes["ai"]
    # Head (4,4) facing RIGHT (body[1] behind it); the cell ahead, (5,4),
    # holds a mid-body segment (not the tail, so tail-vacating cannot save it).
    snake.body.clear()
    snake.body.extend([(4, 4), (3, 4), (5, 4), (3, 5)])
    snake.direction = Action.RIGHT
    validator = SafetyValidator()
    report = validator.check(engine.state, "ai", Action.RIGHT)
    assert not report.safe
    assert report.reason == "self_collision"


def test_opponent_collision_rejected():
    snakes = [
        SnakeSpec("ai", "AI", False, (4, 5), Action.RIGHT),
        SnakeSpec("op", "OP", False, (6, 5), Action.LEFT, length=3),
    ]
    engine = make_engine(seed=22, snakes=snakes, max_foods=0)
    validator = SafetyValidator()
    # Opponent body occupies (6,5),(7,5),(8,5); moving right twice is fatal,
    # but even one step: check the cell directly ahead of our head after the
    # opponent exists -- (6,5) is the opponent's head (tail-aware rules treat
    # heads conservatively), so assert on the opponent's *body* cell instead.
    report = validator.check(engine.state, "ai", Action.RIGHT)
    assert report.safe  # (5,5) is free
    # Step so our head is next to the opponent's mid-body.
    engine.step({"ai": Action.RIGHT, "op": Action.UP})
    snake = engine.state.snakes["ai"]
    assert snake.head == (5, 5)
    # Opponent moved UP: body now (6,4),(6,5),(7,5). Right of us is (6,5) = body.
    report = validator.check(engine.state, "ai", Action.RIGHT)
    assert not report.safe
    assert report.reason == "opponent_collision"


def test_dead_end_rejected():
    # 6x3 board (18 cells); a length-12 snake walls off all but a small
    # pocket. Moving DOWN from the corner head leaves ~7 reachable cells --
    # far fewer than the snake's length -- so the dead-end check fires.
    engine = make_engine(seed=23, width=6, height=3, max_foods=0)
    snake = engine.state.snakes["ai"]
    snake.body.clear()
    snake.body.extend([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
                       (5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (1, 2)])
    snake.direction = Action.RIGHT
    validator = SafetyValidator()
    report = validator.check(engine.state, "ai", Action.DOWN)
    assert not report.safe
    assert report.reason == "dead_end"
    assert 0 < report.free_area < snake.length
    # Contrast: the same check passes when the snake is short enough.
    validator_lenient = SafetyValidator()
    snake.body.clear()
    snake.body.extend([(0, 0), (1, 0), (2, 0)])
    report = validator_lenient.check(engine.state, "ai", Action.DOWN)
    assert report.safe


def test_select_safe_action_overrides_fatal_choice():
    engine = _engine_head_at(15)  # head at right wall
    validator = SafetyValidator()
    action, report = validator.select_safe_action(
        engine.state, "ai", Action.RIGHT,
        fallback_order=[Action.RIGHT, Action.UP, Action.LEFT, Action.DOWN],
    )
    assert action != Action.RIGHT
    assert report.safe
    assert report.intervention  # the preferred move was overridden
    assert any(name == "right" for name, _ in report.rejected)


def test_evaluate_all_covers_four_actions():
    engine = _engine_head_at(8)
    validator = SafetyValidator()
    reports = validator.evaluate_all(engine.state, "ai")
    assert set(reports.keys()) == set(Action)
    assert any(r.safe for r in reports.values())
    reversal = reports[Action.LEFT]
    assert not reversal.safe and reversal.reason == "reversal"


def test_no_safe_action_reports_honestly():
    # Cornered: head (0,0) of a 3x3 board, body covering both open neighbors
    # with *non-tail* cells (the tail cell sits elsewhere, so tail-awareness
    # cannot rescue either move): UP/LEFT hit walls, RIGHT/DOWN hit the body.
    engine = make_engine(seed=24, width=3, height=3, max_foods=0)
    snake = engine.state.snakes["ai"]
    snake.body.clear()
    snake.body.extend([(0, 0), (1, 0), (0, 1), (1, 1)])
    snake.direction = Action.RIGHT
    validator = SafetyValidator()
    action, report = validator.select_safe_action(engine.state, "ai", Action.UP)
    assert action == Action.UP  # preferred returned untouched...
    assert not report.safe
    assert report.reason == "no_safe_action"  # ...and honestly flagged fatal
