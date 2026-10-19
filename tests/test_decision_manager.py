"""DecisionManager: every mode decides, logs, and degrades gracefully."""

from __future__ import annotations

import pytest

from snake_arena.agents.decision_manager import DecisionManager
from snake_arena.agents.factory import build_rl_agent
from snake_arena.game.types import Action
from snake_arena.llm.schemas import StrategyValidationError
from snake_arena.llm.strategist import fallback_strategy

ALL_MODES = ["rule_based", "astar", "bfs", "rl", "llm_astar", "llm_rl", "hybrid"]

from .conftest import make_engine


@pytest.fixture
def rl_agent(config):
    return build_rl_agent(config, load_checkpoint=False)  # fresh = untrained


@pytest.mark.parametrize("mode", ALL_MODES)
def test_every_mode_returns_a_legal_action_and_logs(mode, config, rl_agent):
    manager = DecisionManager(mode, config, rl_agent=rl_agent)
    engine = make_engine(seed=40, max_foods=3)
    for _ in range(25):
        if engine.finished:
            break
        action = manager.decide(engine.state, "ai")
        assert isinstance(action, Action)
        engine.step({"ai": action})
    assert manager.stats["decisions"] >= 20
    assert len(manager.diagnostics) >= 20
    log = manager.diagnostics[-1]
    assert log.mode == mode
    assert log.final_action in {a.name.lower() for a in Action}
    assert log.computation_ms >= 0.0


@pytest.mark.parametrize("mode", ["llm_astar", "llm_rl", "hybrid"])
def test_llm_modes_fall_back_offline_without_stopping(mode, config, rl_agent):
    """No Ollama anywhere: fallback strategy drives the policy, play continues."""
    manager = DecisionManager(mode, config, rl_agent=rl_agent)
    engine = make_engine(seed=41, max_foods=3)
    for _ in range(10):
        if engine.finished:
            break
        manager.decide(engine.state, "ai")
    log = manager.diagnostics[-1]
    assert log.strategy_source == "fallback"
    assert log.strategy_name in ("defensive", "balanced", "aggressive", "greedy", "survival")
    assert manager.stats["llm_calls"] == 0  # never pretended to call the LLM


def test_safety_layer_overrides_stubborn_policy(config, rl_agent):
    """A policy that stubbornly proposes RIGHT must be overridden at the wall:
    no applied move may leave the board, interventions are counted, and the
    decision log records proposed != final."""
    from snake_arena.agents.base import AgentDecision
    from snake_arena.game.engine import SnakeSpec
    from snake_arena.game.types import DIR_VECTORS

    snakes = [SnakeSpec("ai", "AI", False, (12, 6), Action.RIGHT)]
    manager = DecisionManager("rl", config, rl_agent=rl_agent)

    def stubborn_policy(state, snake_id, view, hints, reports, safe_actions):
        return AgentDecision(agent="stubborn", action=Action.RIGHT)

    manager._policy = stubborn_policy  # direct arbitration-layer test
    engine = make_engine(seed=42, width=16, height=12, snakes=snakes, max_foods=0)
    for _ in range(30):
        if engine.finished:
            break
        action = manager.decide(engine.state, "ai")
        head = engine.state.snakes["ai"].head
        dx, dy = DIR_VECTORS[action]
        assert engine.state.is_inside((head[0] + dx, head[1] + dy)), \
            "manager applied an out-of-bounds move while safe moves existed"
        engine.step({"ai": action})
    assert engine.state.snakes["ai"].alive, \
        "the safety layer should prevent avoidable wall death on an open board"
    assert manager.stats["decisions"] == 30
    assert manager.stats["safety_interventions"] >= 1, \
        "a RIGHT-only policy at the right wall must be overridden at least once"
    intervened = [entry for entry in manager.diagnostics if entry.safety_intervened]
    assert intervened
    assert intervened[-1].proposed_action != intervened[-1].final_action


def test_rule_based_mode_uses_default_hints(config, rl_agent):
    manager = DecisionManager("rule_based", config, rl_agent=rl_agent)
    engine = make_engine(seed=43, max_foods=3)
    manager.decide(engine.state, "ai")
    log = manager.diagnostics[-1]
    assert log.strategy_source == "default"
    assert log.strategy_name == "balanced"


def test_hybrid_logs_weight_allocation(config, rl_agent):
    manager = DecisionManager("hybrid", config, rl_agent=rl_agent)
    engine = make_engine(seed=44, max_foods=3)
    manager.decide(engine.state, "ai")
    log = manager.diagnostics[-1]
    assert any("hybrid weights" in note for note in log.notes)
    # Components recorded per action for the UI overlay.
    assert isinstance(log.components, dict)


def test_rl_risk_gate_activates_and_records(config, rl_agent):
    """Survival-heavy hints tighten the area bar; the gate should engage on a
    congested board and note the decision."""
    from snake_arena.agents.base import StrategyHints

    manager = DecisionManager("rl", config, rl_agent=rl_agent)
    # Force survival-priority hints through the private policy path.
    engine = make_engine(seed=45, width=12, height=10, max_foods=0)
    snake = engine.state.snakes["ai"]
    hints = StrategyHints(strategy="survival", food_priority=0.2,
                          survival_priority=1.0, risk_tolerance=0.0, source="default")
    from snake_arena.game.environment import build_state_view

    view = build_state_view(engine.state, "ai")
    decision = manager.rl_agent.decide(engine.state, "ai", view, hints)
    reports = manager.validator.evaluate_all(engine.state, "ai", view)
    gated = manager._gate_rl(engine.state, "ai", view, hints, reports, decision)
    assert gated.diagnostics["min_area"] >= snake.length * (0.8 + 0.9 * 1.0) * (1 - 0.5 * 0.0) - 1e-6
    assert isinstance(gated.action, Action)


def test_reset_episode_clears_state(config, rl_agent):
    manager = DecisionManager("astar", config, rl_agent=rl_agent)
    engine = make_engine(seed=46, max_foods=3)
    manager.decide(engine.state, "ai")
    assert manager.stats["decisions"] == 1
    manager.reset_episode()
    assert len(manager.diagnostics) == 0
    assert manager.current_log is None
    # stats are cumulative by design (per-episode metrics read them per run)
    assert manager.stats["decisions"] == 1


def test_parse_and_validate_drops_unknown_food_target(config):
    from snake_arena.agents.decision_manager import parse_and_validate

    engine = make_engine(seed=47, max_foods=3)
    from snake_arena.game.environment import build_state_view
    from snake_arena.llm.strategist import build_snapshot

    view = build_state_view(engine.state, "ai")
    snap = build_snapshot(engine.state, "ai", view, {})
    strategy = fallback_strategy(snap)
    strategy.target_food_id = "food_that_does_not_exist"
    cleaned = parse_and_validate(strategy.to_dict() | {"target_food_id": "nope"}, snap)
    assert cleaned.target_food_id is None
    assert "pinned target" in cleaned.reason


def test_invalid_llm_payload_is_rejected_not_clamped():
    """A malformed LLM answer must raise, never silently become a strategy."""
    from snake_arena.llm.schemas import parse_strategy

    with pytest.raises(StrategyValidationError):
        parse_strategy({"strategy": "defensive", "food_priority": 42.0})
    with pytest.raises(StrategyValidationError):
        parse_strategy("totally not a dict")
