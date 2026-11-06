"""Console-script entry point (the ``snake-arena`` command).

Deliberately thin. The full CLI (``--check`` / ``--train`` / ``--benchmark``)
lives in the repository's ``main.py`` because those subcommands import the
``scripts`` package, which ships with the source checkout rather than the
installed wheel. After ``pip install .`` this entry point launches the
graphical app with the default configuration; use the source checkout's
``python main.py`` for the complete command set.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="snake-arena",
        description="LLM-Enhanced Intelligent Snake Arena (graphical app)",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="path to a JSON config file (default: bundled defaults)")
    parser.add_argument("--no-llm", action="store_true",
                        help="disable the LLM planner for this run")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    if args.version:
        from snake_arena import PROJECT_NAME, __version__

        print(f"{PROJECT_NAME} v{__version__}")
        return 0

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    from snake_arena.config import load_app_config
    from snake_arena.utils import paths

    paths.ensure_runtime_dirs()
    config = load_app_config(args.config)
    if args.no_llm:
        config.llm.enabled = False

    from snake_arena.ui.app import run_app

    return run_app(config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
