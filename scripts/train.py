#!/usr/bin/env python3
"""Headless DQN training: real episodes, autosave, greedy evaluation, curves.

    python scripts/train.py --episodes 300
    python main.py --train 300        # equivalent

Every printed number comes from the trainer's recorded history; the script
never extrapolates or fabricates results.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run_training(config, episodes: int) -> int:
    from snake_arena.agents.factory import resolve_checkpoint_path
    from snake_arena.evaluation.experiments import plot_training_curves
    from snake_arena.game.environment import SnakeEnv
    from snake_arena.rl.checkpoints import describe_checkpoint
    from snake_arena.rl.trainer import DQNTrainer, TrainingControl, training_worker
    from snake_arena.utils import paths

    paths.ensure_runtime_dirs()
    tcfg = config.training
    board = (tcfg.board_width, tcfg.board_height)
    checkpoint = resolve_checkpoint_path(config, tcfg.checkpoint_name)
    print(f"Training DQN for {episodes} episodes on a {board[0]}x{board[1]} board "
          f"(device: CPU-first, checkpoint: {checkpoint})")

    def env_factory():
        return SnakeEnv(width=board[0], height=board[1],
                        max_foods=config.game.max_foods, max_steps=config.rl.max_episode_steps)

    trainer = DQNTrainer(config.rl, seed=0)
    if checkpoint.exists():
        try:
            payload = trainer.load(checkpoint)
            print(f"resumed from checkpoint: {describe_checkpoint(payload)}")
        except Exception as exc:  # noqa: BLE001
            print(f"could not load checkpoint ({exc}); starting fresh")

    started = time.perf_counter()

    def on_update(stats) -> None:
        if stats.episode % max(1, episodes // 20) == 0 or stats.episode == 1 or stats.episode == episodes:
            loss = f"{stats.loss:.4f}" if stats.loss is not None else "-"
            elapsed = time.perf_counter() - started
            print(f"  episode {stats.episode:>5}/{episodes}  score {stats.score:>3}  "
                  f"reward {stats.reward:>8.1f}  eps {stats.epsilon:.3f}  loss {loss}  "
                  f"[{elapsed:.0f}s]")

    def on_eval(metrics: dict) -> None:
        print(f"  [eval @ episode {metrics.get('episode', '?')}] mean score "
              f"{metrics.get('mean_score', 0.0):.2f} over {metrics.get('episodes', 0)} greedy episodes")

    training_worker(
        trainer,
        train_env_factory=env_factory,
        eval_env_factory=env_factory,
        episodes=episodes,
        control=TrainingControl(),
        on_update=on_update,
        on_eval=on_eval,
        autosave_path=checkpoint,
        autosave_every=tcfg.autosave_every_episodes,
        eval_every=tcfg.eval_every_episodes,
        eval_episodes=tcfg.eval_episodes,
    )

    final = trainer.save(checkpoint)
    print(f"\ncheckpoint saved: {final}")
    metrics = trainer.evaluate(env_factory, episodes=tcfg.eval_episodes)
    print(f"greedy evaluation ({metrics['episodes']} episodes):")
    print(f"  mean score   {metrics['mean_score']:.2f}")
    print(f"  median score {metrics['median_score']:.2f}")
    print(f"  mean steps   {metrics['mean_steps']:.0f}")
    print(f"  endings      {metrics['deaths']}")

    hist = trainer.history.to_dict()
    curves = plot_training_curves(hist, paths.training_dir())
    if curves:
        print(f"training curves: {curves}")
    print("\nNote: RL quality depends strongly on episode count; report these numbers as measured.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the DQN snake agent (headless).")
    parser.add_argument("--episodes", type=int, default=None, help="episodes to train (default: config)")
    parser.add_argument("--config", type=Path, default=None, help="config JSON path")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    from snake_arena.config import load_app_config

    config = load_app_config(args.config)
    episodes = args.episodes if args.episodes is not None else config.training.episodes
    return run_training(config, episodes)


if __name__ == "__main__":
    raise SystemExit(main())
