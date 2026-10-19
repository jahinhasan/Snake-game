#!/usr/bin/env python3
"""Environment self-test: verifies every subsystem and reports honestly.

Runs without a window and without Ollama; exit code 0 means the project is
usable. Missing optional pieces (Ollama, a trained checkpoint) are reported
as such -- they degrade gracefully and are not failures.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from snake_arena import PROJECT_NAME, __version__  # noqa: E402


def _line(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def run_checks(config) -> int:
    print(f"{PROJECT_NAME} v{__version__} - environment check")
    print(f"python: {sys.version.split()[0]} ({sys.platform})")
    all_ok = True

    print("\nPaths")
    from snake_arena.utils import paths

    paths.ensure_runtime_dirs()
    all_ok &= _line("runtime directories", True, str(paths.data_dir()))
    for name, path in (("config", paths.default_config_path()),
                       ("foods", paths.default_foods_path()),
                       ("models dir", paths.models_dir())):
        all_ok &= _line(name, path.exists() or path.is_dir(), str(path))

    print("\nDependencies")
    import numpy
    import pygame

    all_ok &= _line("pygame", True, pygame.version.ver)
    all_ok &= _line("numpy", True, numpy.__version__)
    try:
        import torch

        all_ok &= _line("torch", True, f"{torch.__version__} ({torch.get_num_threads()} threads)")
    except Exception as exc:  # noqa: BLE001
        all_ok &= _line("torch", False, f"import failed: {exc}")
    try:
        import matplotlib

        all_ok &= _line("matplotlib", True, matplotlib.__version__)
    except Exception as exc:  # noqa: BLE001
        all_ok &= _line("matplotlib", False, f"import failed: {exc} (charts disabled)")

    print("\nConfiguration")
    config.validate()
    all_ok &= _line("config loads and validates", True,
                    f"board {config.board.width}x{config.board.height}, "
                    f"{config.game.moves_per_second} moves/s, LLM {'on' if config.llm.enabled else 'off'}")
    from snake_arena.game.food import FoodCatalog

    try:
        catalog = FoodCatalog.load(paths.default_foods_path())
        all_ok &= _line("food catalog", True,
                        ", ".join(f"{t.name} {t.value:+d}" for t in catalog.types.values()))
    except Exception as exc:  # noqa: BLE001
        all_ok &= _line("food catalog", False, str(exc))

    print("\nEngine smoke test")
    from snake_arena.game.engine import GameEngine, SnakeSpec
    from snake_arena.game.types import Action

    try:
        engine = GameEngine(
            width=config.board.width, height=config.board.height,
            moves_per_second=config.game.moves_per_second,
            max_foods=config.game.max_foods, seed=42, time_limit_seconds=60,
            snake_specs=[SnakeSpec("ai", "AI", False, (3, 3), Action.RIGHT)],
        )
        for _ in range(30):
            engine.step({"ai": Action.RIGHT})
        all_ok &= _line("engine steps", True,
                        f"tick {engine.state.tick}, outcome {engine.outcome}, foods {len(engine.state.foods)}")
    except Exception as exc:  # noqa: BLE001
        all_ok &= _line("engine steps", False, str(exc))

    print("\nRL agent")
    try:
        from snake_arena.agents.factory import build_rl_agent
        from snake_arena.rl.checkpoints import describe_checkpoint

        agent = build_rl_agent(config)
        if agent.trained:
            path = agent.checkpoint_path if hasattr(agent, "checkpoint_path") else None
            detail = f"checkpoint loaded from {path.name if path else '?'}"
        else:
            detail = "no usable checkpoint - running with a random-init network (safety validator active)"
        all_ok &= _line("DQN builds", True, detail)
    except Exception as exc:  # noqa: BLE001
        all_ok &= _line("DQN builds", False, str(exc))

    print("\nLLM planner (optional)")
    if not config.llm.enabled:
        _line("status", True, "disabled in config (the game runs fully without it)")
    else:
        from snake_arena.llm.ollama_client import OllamaClient

        client = OllamaClient(config.llm)
        started = time.time()
        ok, detail = client.is_available()
        if ok:
            has_model, model_note = client.has_model()
            if has_model:
                _line("Ollama", True, f"{model_note} ({time.time() - started:.1f}s)")
            else:
                _line("Ollama server", True, f"reachable, but {model_note} - LLM agents will use the fallback strategy")
        else:
            _line("Ollama", True, f"not reachable ({detail}) - expected if not installed; fallback strategy is used")

    print("\nResult: " + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED (see above)"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    from snake_arena.config import load_app_config

    raise SystemExit(run_checks(load_app_config()))
