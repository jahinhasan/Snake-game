"""Benchmark runner + metrics: identical seeds, honest flags, real summaries."""

from __future__ import annotations

import pytest

from snake_arena.evaluation.benchmark import BenchmarkRunner
from snake_arena.evaluation.metrics import EpisodeRecord, summarize

from .conftest import make_engine


@pytest.fixture
def runner(config):
    config.benchmark.base_seed = 500
    return BenchmarkRunner(config, agents=["rule_based", "astar"], episodes=2,
                           llm_episodes=0, max_ticks=250)


def test_solo_benchmark_structure_and_shared_seeds(runner):
    results = runner.run()
    assert not results.get("cancelled", False)
    assert set(results["agents"].keys()) == {"rule_based", "astar"}
    assert results["meta"]["episodes_per_agent"] == 2
    assert results["meta"]["llm_available"] is False
    assert "records" in results and len(results["records"]) == 4

    seeds_per_agent: dict[str, list[int]] = {}
    for record in results["records"]:
        seeds_per_agent.setdefault(record["agent"], []).append(record["seed"])
    # Identical per-index seeds: differences come only from the policy.
    assert seeds_per_agent["rule_based"] == seeds_per_agent["astar"] == [500, 501]

    for name, summary in results["agents"].items():
        assert summary["episodes"] == 2
        assert "mean_score" in summary and "std_score" in summary
        assert 0.0 <= summary["survival_rate"] <= 1.0


def test_llm_agent_runs_fallback_and_flags_it(config):
    runner = BenchmarkRunner(config, agents=["llm_astar"], episodes=1,
                             llm_episodes=1, max_ticks=120)
    results = runner.run()
    record = results["records"][0]
    assert record["agent"] == "llm_astar"
    assert record["llm_calls"] == 0  # no Ollama in tests: nothing fabricated
    assert record["llm_available"] is False


def test_match_reports_winner_fields(config):
    runner = BenchmarkRunner(config, agents=["rule_based", "astar"], episodes=1,
                             max_ticks=300)
    match = runner.run_match("rule_based", "astar", seed=77)
    assert match["agent_a"] == "rule_based"
    assert match["agent_b"] == "astar"
    assert set(match["scores"].keys()) == {"a", "b"}
    # "ongoing" is a legitimate truncated match (tick cap hit without a kill).
    assert match["outcome"] in ("survivor_win", "timeout", "draw", "ongoing")
    if match["outcome"] == "survivor_win":
        assert match["winner"] in ("a", "b")
        assert match["winner_agent"] in ("rule_based", "astar")
    else:
        assert match["winner_agent"] is None


def test_summarize_computes_real_statistics():
    records = [
        EpisodeRecord(agent="astar", seed=1, score=10, steps=50, outcome="death",
                      death_reason="wall", length=5, survived=False),
        EpisodeRecord(agent="astar", seed=2, score=30, steps=100, outcome="timeout",
                      death_reason=None, length=7, survived=True),
        EpisodeRecord(agent="astar", seed=3, score=20, steps=80, outcome="death",
                      death_reason="self", length=6, survived=False),
    ]
    summary = summarize(records, agent="astar").to_dict()
    assert summary["episodes"] == 3
    assert summary["mean_score"] == pytest.approx(20.0)
    assert summary["median_score"] == pytest.approx(20.0)
    assert summary["min_score"] == 10 and summary["max_score"] == 30
    assert summary["survival_rate"] == pytest.approx(1 / 3, abs=1e-3)  # summary rounds to 3 decimals
    assert summary["death_reasons"] == {"wall": 1, "self": 1, "timeout": 1}


def test_episode_runs_to_capped_ticks(config):
    """max_ticks bounds even an unkillable agent (timeout semantics)."""
    runner = BenchmarkRunner(config, agents=["rule_based"], episodes=1, max_ticks=60)
    results = runner.run()
    record = results["records"][0]
    assert record["steps"] <= 60


def test_cancel_flag_stops_mid_run(config):
    runner = BenchmarkRunner(config, agents=["rule_based", "astar"], episodes=5,
                             max_ticks=100, cancel_check=lambda: True)
    results = runner.run()
    assert results.get("cancelled") is True
    assert results["agents"] == {} or len(results["agents"]) < 2


def test_engine_seeded_environment_matches():
    """The benchmark's engine factory honours seeds (reproducible arenas)."""
    e1 = make_engine(seed=99, max_foods=3)
    e2 = make_engine(seed=99, max_foods=3)
    foods1 = sorted(f.pos for f in e1.state.foods)
    foods2 = sorted(f.pos for f in e2.state.foods)
    assert foods1 == foods2
