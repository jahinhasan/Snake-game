"""Integration: complete automated episodes through the full stack.

Engine -> DecisionManager (each mode) -> engine step -> game record ->
history + benchmark artifacts. No mocks: offline LLM fallback is exercised
for what it is (fallback), and RL runs its honest untrained network.
"""

from __future__ import annotations

import pytest

from snake_arena.agents.decision_manager import DecisionManager
from snake_arena.agents.factory import build_rl_agent
from snake_arena.evaluation.benchmark import AGENT_LABELS, BenchmarkRunner
from snake_arena.evaluation.experiments import (
    benchmark_rows,
    make_run_dir,
    plot_benchmark_charts,
    save_benchmark_results,
)
from snake_arena.game.environment import OBS_SIZE, SnakeEnv
from snake_arena.game.types import Action
from snake_arena.persistence.history import build_game_record, load_recent

from .conftest import make_engine

ALL_MODES = ["rule_based", "astar", "bfs", "rl", "llm_astar", "llm_rl", "hybrid"]


@pytest.fixture
def rl_agent(config):
    return build_rl_agent(config, load_checkpoint=False)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_complete_managed_episode(mode, config, rl_agent):
    """A full episode driven by each decision mode, start to finish."""
    manager = DecisionManager(mode, config, rl_agent=rl_agent)
    engine = make_engine(seed=70 + ALL_MODES.index(mode), max_foods=4)
    manager.reset_episode()
    steps = 0
    while not engine.finished and steps < 400:
        action = manager.decide(engine.state, "ai")
        engine.step({"ai": action})
        steps += 1
    assert steps > 0
    assert manager.stats["decisions"] == steps

    record = build_game_record(
        mode=f"solo_{mode}", agent_name=mode, engine_state=engine.state,
        outcome=engine.outcome, winner_ids=[], manager_stats=dict(manager.stats),
        seed=70,
    )
    assert record["snakes"]["ai"]["score"] == engine.state.snakes["ai"].score
    assert record["ticks"] == steps
    assert record["ai_summary"]["decisions"] == steps
    # LLM-mode records must not claim LLM usage that never happened.
    if mode in ("llm_astar", "llm_rl"):
        assert record["ai_summary"]["llm_calls"] == 0


def test_environment_observation_and_steps(config):
    env = SnakeEnv(width=16, height=12, max_foods=3, max_steps=200, seed=80)
    obs = env.reset(seed=80)
    assert obs.shape == (OBS_SIZE,)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 250:
        obs, events, terminated, truncated, info = env.step(int(Action.RIGHT))
        steps += 1
    assert steps > 0
    assert "score" in info and "view" in info and "events" in info
    # Walking right into the wall terminates the episode.
    assert terminated or truncated


def test_end_to_end_benchmark_artifacts(config, tmp_path, monkeypatch):
    """Runner -> saved JSON/CSV -> charts -> reload; numbers stay identical."""
    from snake_arena.utils import paths as paths_mod

    monkeypatch.setattr(paths_mod, "benchmark_dir", lambda: tmp_path / "benchmarks")
    config.benchmark.base_seed = 900
    runner = BenchmarkRunner(config, agents=["astar", "bfs"], episodes=2,
                             llm_episodes=0, max_ticks=150)
    results = runner.run()
    run_dir = make_run_dir("it")
    save_benchmark_results(run_dir, results)
    charts = plot_benchmark_charts(run_dir, results)
    assert (run_dir / "results.json").exists()
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "records.csv").exists()
    assert len(charts) >= 2

    from snake_arena.evaluation.experiments import load_run

    reloaded = load_run(run_dir)
    assert reloaded is not None
    rows_before = benchmark_rows(results)
    rows_after = benchmark_rows(reloaded)
    assert rows_before == rows_after  # artifacts faithfully round-trip
    for row in rows_after:
        assert row["agent"] in {AGENT_LABELS["astar"], AGENT_LABELS["bfs"]}


def test_history_receives_played_games(config, tmp_path, monkeypatch):
    from snake_arena.persistence import history as history_mod
    from snake_arena.utils import paths as paths_mod

    monkeypatch.setattr(history_mod, "history_path", lambda: tmp_path / "games.jsonl")
    manager = DecisionManager("astar", config, rl_agent=rl_agent)
    engine = make_engine(seed=90, max_foods=3)
    while not engine.finished and engine.state.tick < 300:
        action = manager.decide(engine.state, "ai")
        engine.step({"ai": action})
    record = build_game_record(mode="vs_ai", agent_name="astar",
                               engine_state=engine.state, outcome=engine.outcome,
                               winner_ids=[], manager_stats=dict(manager.stats))
    history_mod.record_game(record)
    stored = load_recent(5)
    assert len(stored) == 1
    assert stored[0]["agent"] == "astar"
    assert (tmp_path / "games.jsonl").exists()
