"""Atomic checkpoint persistence for DQN training state.

A checkpoint bundles network weights, optimizer state, architecture metadata,
hyperparameters, and the full training history so training can be resumed
exactly and models remain interpretable later. Saves are atomic
(write-temp-then-rename) so an interrupted save can never corrupt the last
valid checkpoint.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .. import __version__

log = logging.getLogger("snake_arena.rl.checkpoints")

CHECKPOINT_SCHEMA_VERSION = 2


class CheckpointError(Exception):
    pass


@dataclass
class TrainingHistory:
    """Everything needed to plot real learning curves (no fabricated data)."""

    episodes: list[dict] = field(default_factory=list)  # reward, score, length, epsilon, loss, steps, ts
    evaluations: list[dict] = field(default_factory=list)  # episode, mean_score, mean_reward, mean_length
    total_steps: int = 0
    best_score: float = float("-inf")
    best_episode: int = -1

    def add_episode(self, **kwargs) -> None:
        kwargs["ts"] = time.time()
        self.episodes.append(kwargs)
        score = float(kwargs.get("score", 0.0))
        if score > self.best_score:
            self.best_score = score
            self.best_episode = int(kwargs.get("episode", len(self.episodes)))

    def add_evaluation(self, **kwargs) -> None:
        self.evaluations.append(kwargs)

    def recent(self, n: int) -> list[dict]:
        return self.episodes[-n:]

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "evaluations": self.evaluations,
            "total_steps": self.total_steps,
            "best_score": self.best_score if self.best_score != float("-inf") else None,
            "best_episode": self.best_episode,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "TrainingHistory":
        if not data:
            return cls()
        hist = cls(
            episodes=list(data.get("episodes", [])),
            evaluations=list(data.get("evaluations", [])),
            total_steps=int(data.get("total_steps", 0)),
            best_episode=int(data.get("best_episode", -1)),
        )
        best = data.get("best_score")
        hist.best_score = float("-inf") if best is None else float(best)
        return hist


def build_checkpoint(
    *,
    online: torch.nn.Module,
    target: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_size: int,
    hidden_sizes: list[int],
    n_actions: int,
    hyperparams: dict,
    history: TrainingHistory,
    epsilon: float,
    steps: int,
    episodes: int,
    trained: bool,
) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "app_version": __version__,
        "created_at": time.time(),
        "obs_size": int(obs_size),
        "hidden_sizes": list(hidden_sizes),
        "n_actions": int(n_actions),
        "online_state": online.state_dict(),
        "target_state": target.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "hyperparams": dict(hyperparams),
        "history": history.to_dict(),
        "epsilon": float(epsilon),
        "steps": int(steps),
        "episodes": int(episodes),
        "trained": bool(trained),
    }


def save_checkpoint(path: Path, payload: dict) -> Path:
    """Atomically write a checkpoint (tmp file then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    log.info("Checkpoint saved: %s", path)
    return path


def load_checkpoint(path: Path, expected_obs_size: int | None = None) -> dict:
    """Load and validate a checkpoint; raises CheckpointError on problems."""
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - corrupt/partial files must not crash
        raise CheckpointError(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise CheckpointError(f"File is not a snake-arena checkpoint: {path}")
    if int(payload["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"Checkpoint schema v{payload['schema_version']} != supported v{CHECKPOINT_SCHEMA_VERSION}: {path}"
        )
    if expected_obs_size is not None and int(payload.get("obs_size", -1)) != int(expected_obs_size):
        raise CheckpointError(
            f"Checkpoint obs_size {payload.get('obs_size')} != expected {expected_obs_size} "
            "(observation definition changed; retrain or use a matching checkpoint)"
        )
    for key in ("online_state", "target_state"):
        if key not in payload:
            raise CheckpointError(f"Checkpoint missing '{key}': {path}")
    return payload


def describe_checkpoint(payload: dict) -> str:
    """Human-readable one-liner used by the UI and logs."""
    trained = payload.get("trained", False)
    episodes = payload.get("episodes", 0)
    best = payload.get("history", {}).get("best_score")
    state = "trained" if trained and episodes > 0 else "untrained"
    best_txt = f", best score {best:g}" if isinstance(best, (int, float)) else ""
    return f"{state} ({episodes} episodes{best_txt})"
