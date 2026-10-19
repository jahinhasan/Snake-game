#!/usr/bin/env python3
"""Headless agent benchmark: solo episodes + optional tournament, real metrics.

    python scripts/benchmark.py --episodes 20 --agents astar,bfs,rl
    python main.py --benchmark 20

Artifacts land in ``data/benchmarks/<run_id>/``: results.json, summary.csv,
records.csv and PNG charts. When Ollama is unavailable the LLM-agent rows run
on the deterministic fallback strategy and the run metadata says so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VALID_AGENTS = ("rule_based", "astar", "bfs", "rl", "llm_astar", "llm_rl", "hybrid")


def run_benchmark(config, episodes: int, agents: list[str] | None = None) -> int:
    from snake_arena.agents.factory import build_llm_client
    from snake_arena.evaluation.benchmark import AGENT_LABELS, BenchmarkRunner
    from snake_arena.evaluation.experiments import (
        benchmark_rows,
        make_run_dir,
        plot_benchmark_charts,
        save_benchmark_results,
    )

    agents = [a for a in (agents or config.benchmark.agents) if a in VALID_AGENTS]
    if not agents:
        print("No valid agents selected.")
        return 1
    llm_client = build_llm_client(config) if config.llm.enabled else None

    runner = BenchmarkRunner(
        config,
        agents=agents,
        episodes=episodes,
        llm_client=llm_client,
        progress_cb=lambda p: print(f"  {p.message or p.agent}", flush=True),
    )
    print(f"Benchmark: {len(agents)} agents x {episodes} episodes "
          f"({config.board.width}x{config.board.height} board, identical seeds per index)")
    results = runner.run()

    run_dir = make_run_dir("cli")
    paths_map = save_benchmark_results(run_dir, results)
    charts = plot_benchmark_charts(run_dir, results)

    meta = results.get("meta", {})
    if meta.get("llm_available") is False:
        print(f"NOTE: {meta.get('llm_note', 'LLM unavailable; LLM agents used the fallback strategy.')}")
    print(f"\nResults ({' + '.join(AGENT_LABELS.get(a, a) for a in agents)}):")
    header = ("agent", "eps", "mean", "median", "std", "surv", "trap/1k", "ms")
    print(f"  {'agent':<14}{'eps':>5}{'mean':>8}{'median':>8}{'std':>7}{'surv':>7}{'trap/1k':>9}{'ms':>7}")
    for row in benchmark_rows(results):
        print(f"  {row['agent']:<14}{row['episodes']:>5}{row['mean_score']:>8.1f}"
              f"{row['median_score']:>8.1f}{row['std']:>7.1f}{row['survival']:>7}"
              f"{row['traps/1k']:>9.2f}{row['decision_ms']:>7.1f}")
    print(f"\nsaved: {run_dir}")
    for label, path in paths_map.items():
        print(f"  {label}: {path.name}")
    for chart in charts:
        print(f"  chart: {chart.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent benchmark (headless).")
    parser.add_argument("--episodes", type=int, default=None, help="episodes per agent (default: config)")
    parser.add_argument("--agents", type=str, default=None,
                        help=f"comma-separated subset of {','.join(VALID_AGENTS)}")
    parser.add_argument("--tournament", action="store_true",
                        help="also run an AI-vs-AI round-robin tournament")
    parser.add_argument("--config", type=Path, default=None, help="config JSON path")
    args = parser.parse_args()

    import logging
    import time

    logging.basicConfig(level=logging.WARNING)
    from snake_arena.config import load_app_config
    from snake_arena.evaluation.benchmark import AGENT_LABELS, BenchmarkRunner
    from snake_arena.evaluation.experiments import make_run_dir, save_benchmark_results

    config = load_app_config(args.config)
    agents = [a.strip() for a in args.agents.split(",")] if args.agents else None
    code = run_benchmark(config, args.episodes or config.benchmark.episodes_per_agent, agents)
    if code == 0 and args.tournament:
        from snake_arena.agents.factory import build_llm_client

        llm_client = build_llm_client(config) if config.llm.enabled else None
        runner = BenchmarkRunner(config, agents=agents, llm_client=llm_client)
        tournament = runner.run_tournament(agents=agents, rounds=2)
        run_dir = make_run_dir("tournament")
        save_benchmark_results(run_dir, {"run_id": None, "timestamp": time.time(),
                                         "meta": {"kind": "tournament", "agents": agents or []},
                                         "agents": {}, "records": [], "tournament": tournament})
        print(f"\nTournament ({tournament['matches']} matches, {tournament['draws']} draws):")
        for agent, wins in sorted(tournament["win_rate"].items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {AGENT_LABELS.get(agent, agent):<14} wins {tournament['wins'][agent]:>3}  "
                  f"win rate {wins * 100:.0f}%")
        print(f"saved: {run_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
