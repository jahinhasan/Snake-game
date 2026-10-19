#!/usr/bin/env python3
"""Entry point for the LLM-Enhanced Intelligent Snake Arena.

Usage:
    python main.py                 # launch the graphical app (default)
    python main.py --check         # environment self-test (no window)
    python main.py --train 200     # train the DQN headless for N episodes
    python main.py --benchmark 20  # run the agent benchmark headless
    python main.py --no-llm        # force the LLM planner off for this run

Windows: use ``run_windows.bat``; Linux/macOS: ``./run_linux.sh``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow ``python main.py`` from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from snake_arena import PROJECT_NAME, __version__  # noqa: E402
from snake_arena.config import load_app_config  # noqa: E402
from snake_arena.utils import paths  # noqa: E402


def _setup_logging(verbose: bool) -> None:
    paths.ensure_runtime_dirs()
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.logs_dir() / "app.log", encoding="utf-8"),
        ],
    )


def _cmd_check(config) -> int:
    from scripts.check_environment import run_checks

    return run_checks(config)


def _cmd_train(config, episodes: int) -> int:
    from scripts.train import run_training

    return run_training(config, episodes)


def _cmd_benchmark(config, episodes: int, agents: list[str] | None = None) -> int:
    from scripts.benchmark import run_benchmark

    return run_benchmark(config, episodes, agents=agents)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="snake-arena",
        description=f"{PROJECT_NAME} v{__version__}",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="path to a JSON config file (default: config/default.json)")
    parser.add_argument("--check", action="store_true",
                        help="run the environment self-test and exit")
    parser.add_argument("--train", type=int, default=None, metavar="EPISODES",
                        help="train the DQN agent headless for N episodes, then exit")
    parser.add_argument("--benchmark", type=int, default=None, metavar="EPISODES",
                        help="run the agent benchmark headless (N episodes per agent), then exit")
    parser.add_argument("--agents", type=str, default=None,
                        help="comma-separated agents for --benchmark "
                             "(rule_based,astar,bfs,rl,llm_astar,llm_rl,hybrid)")
    parser.add_argument("--no-llm", action="store_true",
                        help="disable the LLM planner for this run")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    if args.version:
        print(f"{PROJECT_NAME} v{__version__}")
        return 0

    _setup_logging(args.verbose)
    config = load_app_config(args.config)
    if args.no_llm:
        config.llm.enabled = False

    if args.check:
        return _cmd_check(config)
    if args.train is not None:
        return _cmd_train(config, args.train)
    if args.benchmark is not None:
        agent_list = args.agents.split(",") if args.agents else None
        return _cmd_benchmark(config, args.benchmark, agent_list)

    # Default: the graphical application.
    from snake_arena.ui.app import run_app

    return run_app(config)


if __name__ == "__main__":
    raise SystemExit(main())
