"""Persistence: atomic saves, corrupt recovery, JSONL repair, history."""

from __future__ import annotations

import json

import pytest

from snake_arena.config import AppConfig
from snake_arena.persistence.history import (
    aggregate,
    build_game_record,
    clear_history,
    load_recent,
    record_game,
)
from snake_arena.persistence.storage import (
    append_jsonl,
    load_json,
    read_jsonl,
    run_id,
    save_json_atomic,
    write_csv,
)

from .conftest import make_engine


def test_atomic_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "data.json"
    save_json_atomic(path, {"a": 1, "b": [1, 2, 3]})
    assert load_json(path) == {"a": 1, "b": [1, 2, 3]}
    assert not path.with_suffix(path.suffix + ".tmp").exists()  # temp cleaned by rename


def test_load_json_defaults_and_quarantine(tmp_path):
    assert load_json(tmp_path / "missing.json", default=[]) == []
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{definitely not json", encoding="utf-8")
    assert load_json(corrupt, default={"ok": True}) == {"ok": True}
    assert corrupt.with_suffix(corrupt.suffix + ".corrupt").exists()  # moved aside


def test_jsonl_roundtrip_and_damaged_line_repair(tmp_path):
    path = tmp_path / "log.jsonl"
    append_jsonl(path, {"i": 1})
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{broken line\n")  # simulate a torn write
    append_jsonl(path, {"i": 2})
    items = read_jsonl(path)
    assert [item["i"] for item in items] == [1, 2]  # damaged line skipped
    assert read_jsonl(path, limit=1) == [{"i": 2}]


def test_run_id_format():
    rid = run_id("bench")
    assert rid.startswith("bench-")
    parts = rid.split("-")
    assert len(parts[-1]) == 4  # random suffix


def test_write_csv(tmp_path):
    path = tmp_path / "s.csv"
    write_csv(path, [{"a": 1, "b": "x"}, {"a": 2, "b": "y", "c": {"k": 1}}])
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "a,b,c"
    assert len(lines) == 3
    write_csv(tmp_path / "empty.csv", [])
    assert (tmp_path / "empty.csv").read_text(encoding="utf-8") == ""


def test_config_roundtrip_and_validation_clamps(tmp_path):
    cfg = AppConfig()
    cfg.game.moves_per_second = 999  # out of range
    path = tmp_path / "config.json"
    cfg.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    loaded = AppConfig.load(path)
    assert raw["game"]["moves_per_second"] == 999  # saved as written...
    assert loaded.game.moves_per_second == 30  # ...but clamped on validate


def test_config_corrupt_file_recovers_with_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("}}}nope", encoding="utf-8")
    cfg = AppConfig.load(path)
    assert cfg.game.moves_per_second == 10  # defaults
    assert path.with_suffix(path.suffix + ".corrupt").exists()


def test_partial_config_merges_tolerantly(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"game": {"moves_per_second": 22, "bogus_key": 1},
                                "no_such_section": {}}), encoding="utf-8")
    cfg = AppConfig.load(path)
    assert cfg.game.moves_per_second == 22
    assert cfg.board.width == 24  # untouched default
    assert not hasattr(cfg, "no_such_section")


def test_history_record_build_and_aggregate(tmp_path, monkeypatch):
    from snake_arena.persistence import history as history_mod

    fake_path = tmp_path / "games.jsonl"
    monkeypatch.setattr(history_mod, "history_path", lambda: fake_path)

    engine = make_engine(seed=60, max_foods=3)
    for _ in range(30):
        if engine.finished:
            break
        engine.step({"ai": engine.state.snakes["ai"].direction})  # keep moving
    record = build_game_record(
        mode="vs_ai", agent_name="astar", engine_state=engine.state,
        outcome=engine.outcome, winner_ids=[],
        manager_stats={"decisions": 30, "safety_interventions": 0},
        seed=60,
    )
    assert record["mode"] == "vs_ai"
    assert record["snakes"]["ai"]["score"] == engine.state.snakes["ai"].score
    record_game(record)
    record_game(record)
    assert len(load_recent(10)) == 2

    agg = aggregate()
    assert agg["games"] == 2
    assert "astar" in agg["by_agent"]
    assert agg["by_agent"]["astar"]["games"] == 2

    assert clear_history() == 2
    assert aggregate()["games"] == 0


def test_history_aggregate_skips_human_scores(tmp_path, monkeypatch):
    from snake_arena.persistence import history as history_mod

    fake_path = tmp_path / "games.jsonl"
    monkeypatch.setattr(history_mod, "history_path", lambda: fake_path)
    engine = make_engine(seed=61, max_foods=2)
    snakes = engine.state.snakes
    snakes["ai"].is_human = False
    record = build_game_record(mode="ai_vs_ai", agent_name="astar",
                               engine_state=engine.state, outcome="ongoing",
                               winner_ids=[])
    record_game(record)
    agg = aggregate()
    assert agg["games"] == 1
    assert agg["by_agent"]["astar"]["games"] == 1


@pytest.fixture(autouse=True)
def _isolate_real_history(tmp_path, monkeypatch):
    """Never touch the developer's real history file from tests."""
    from snake_arena.persistence import history as history_mod

    monkeypatch.setattr(history_mod, "history_path",
                        lambda: tmp_path / "unused.jsonl", raising=False)
    yield
