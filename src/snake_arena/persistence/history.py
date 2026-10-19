"""Game history: append-only JSONL of played episodes plus aggregates."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ..utils import paths
from .storage import append_jsonl, read_jsonl, save_json_atomic

log = logging.getLogger("snake_arena.history")

HISTORY_SCHEMA_VERSION = 1


def history_path() -> Path:
    return paths.history_dir() / "games.jsonl"


def build_game_record(
    *,
    mode: str,
    agent_name: str,
    engine_state,
    outcome: str,
    winner_ids: list[str],
    manager_stats: dict | None = None,
    strategy_summary: str = "",
    llm_info: dict | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Structured record of one finished game (measured facts only)."""
    snakes: dict[str, Any] = {}
    for sid, s in engine_state.snakes.items():
        snakes[sid] = {
            "name": s.name,
            "is_human": s.is_human,
            "score": s.score,
            "length": s.length,
            "foods_eaten": dict(s.foods_eaten),
            "death_reason": s.death_reason,
            "survived": s.alive,
            "rejected_moves": s.rejected_moves,
        }
    record = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "game_id": engine_state.game_id,
        "timestamp": time.time(),
        "mode": mode,
        "agent": agent_name,
        "board": {"width": engine_state.width, "height": engine_state.height},
        "moves_per_second": engine_state.moves_per_second,
        "seed": seed,
        "ticks": engine_state.tick,
        "seconds": round(engine_state.seconds_elapsed, 2),
        "outcome": outcome,
        "winner_ids": list(winner_ids),
        "snakes": snakes,
        "ai_summary": dict(manager_stats or {}),
        "strategy_summary": strategy_summary,
        "llm": dict(llm_info or {}),
    }
    return record


def record_game(record: dict[str, Any]) -> Path:
    return append_jsonl(history_path(), record)


def load_recent(n: int = 50) -> list[dict]:
    return read_jsonl(history_path(), limit=n)


def aggregate() -> dict[str, Any]:
    """Aggregate statistics across the whole history (real data only)."""
    games = read_jsonl(history_path())
    if not games:
        return {"games": 0}
    total_score = 0
    top = None
    by_agent: dict[str, list[int]] = {}
    for game in games:
        for sid, s in game.get("snakes", {}).items():
            if s.get("is_human"):
                continue
            by_agent.setdefault(game.get("agent", sid), []).append(s.get("score", 0))
            total_score += s.get("score", 0)
            if top is None or s.get("score", 0) > top["score"]:
                top = {"score": s.get("score", 0), "agent": game.get("agent", sid),
                       "when": game.get("timestamp", 0)}
    agent_stats = {}
    for agent, scores in sorted(by_agent.items()):
        agent_stats[agent] = {
            "games": len(scores),
            "mean_score": round(sum(scores) / len(scores), 1),
            "max_score": max(scores),
        }
    return {
        "games": len(games),
        "total_ai_score": total_score,
        "top_game": top,
        "by_agent": agent_stats,
    }


def clear_history() -> int:
    """Delete history; returns how many records were removed."""
    path = history_path()
    count = len(read_jsonl(path))
    if path.exists():
        path.unlink()
    log.info("Cleared %s history records", count)
    return count


def save_settings_backup(config_dict: dict) -> Path:
    return save_json_atomic(paths.data_dir() / "settings_snapshot.json", config_dict)
