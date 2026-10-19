"""LLM integration with a *fake* transport: no network, no Ollama needed.

Covers: JSON-mode chat, <think> stripping, strict schema validation (bad
output rejected, never clamped), retries, availability probing, and the
deterministic fallback strategy.
"""

from __future__ import annotations

import json

import pytest

from snake_arena.config import LLMConfig
from snake_arena.llm.ollama_client import (
    OllamaBadResponse,
    OllamaClient,
    OllamaUnavailable,
)
from snake_arena.llm.schemas import (
    STRATEGIES,
    StrategyValidationError,
    extract_json_object,
    parse_strategy,
)
from snake_arena.llm.strategist import (
    build_snapshot,
    fallback_strategy,
    format_messages,
    snapshot_hash,
)

from .conftest import make_engine


class FakeTransport:
    """Programmable stand-in for the HTTP transport."""

    def __init__(self, responses: list | None = None, fail_first: int = 0):
        self.calls: list[tuple[str, bytes | None, float]] = []
        self.responses = list(responses or [])
        self.fail_first = fail_first

    def __call__(self, path: str, payload: bytes | None, timeout: float):
        self.calls.append((path, payload, timeout))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise OllamaUnavailable("connection refused (fake)")
        if not self.responses:
            raise AssertionError("unexpected extra transport call")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


VALID_STRATEGY = {
    "strategy": "defensive",
    "food_priority": 0.7,
    "survival_priority": 0.9,
    "risk_tolerance": 0.2,
    "target_food_id": None,
    "reason": "Opponent near the golden food; play safe.",
}


def make_client(transport) -> OllamaClient:
    return OllamaClient(host="http://127.0.0.1:11434", model="qwen3:4b",
                        timeout=2.0, connect_timeout=1.0, max_retries=1,
                        transport=transport)


def test_is_available_true_and_models_listed():
    tags = (200, json.dumps({"models": [{"name": "qwen3:4b"}]}).encode())
    transport = FakeTransport([tags, tags])  # one probe + one list_models call
    client = make_client(transport)
    ok, detail = client.is_available()
    assert ok and "qwen3:4b" in detail
    has, name = client.has_model()
    assert has and name == "qwen3:4b"


def test_is_available_reports_unreachable():
    client = make_client(FakeTransport(fail_first=99))
    ok, detail = client.is_available()
    assert not ok
    assert "refused" in detail
    assert client.last_error


def test_chat_json_returns_parsed_object():
    transport = FakeTransport([
        (200, json.dumps({"message": {"content": json.dumps(VALID_STRATEGY)}}).encode()),
    ])
    client = make_client(transport)
    obj = client.chat_json("system prompt", "user prompt")
    assert obj["strategy"] == "defensive"
    path, payload, _timeout = transport.calls[0]
    assert path == "/api/chat"
    body = json.loads(payload.decode())
    assert body["format"] == "json" and body["stream"] is False


def test_chat_strips_think_block():
    content = "<think>deliberating...</think>" + json.dumps(VALID_STRATEGY)
    transport = FakeTransport([(200, json.dumps({"message": {"content": content}}).encode())])
    obj = make_client(transport).chat_json("s", "u")
    assert obj["strategy"] == "defensive"


def test_chat_retries_transport_failure_then_succeeds():
    ok_response = (200, json.dumps({"message": {"content": json.dumps(VALID_STRATEGY)}}).encode())
    transport = FakeTransport(responses=[ok_response], fail_first=1)
    obj = make_client(transport).chat_json("s", "u")
    assert obj["strategy"] == "defensive"
    assert len(transport.calls) == 2  # one failure + one retry


def test_chat_retries_garbage_output_then_raises():
    transport = FakeTransport([
        (200, b"this is not json at all"),
        (200, b"still not json"),
    ])
    client = make_client(transport)
    with pytest.raises(OllamaBadResponse):
        client.chat_json("s", "u")
    assert len(transport.calls) == 2  # one retry (max_retries=1), then honest failure


def test_extract_json_object_variants():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('noise {"a": {"b": 2}} trailing') == {"a": {"b": 2}}
    with pytest.raises(Exception):
        extract_json_object("no braces here")


def test_parse_strategy_strict_validation():
    strategy = parse_strategy(dict(VALID_STRATEGY))
    assert strategy.strategy == "defensive"
    assert strategy.food_priority == 0.7

    # Out-of-range values are REJECTED, not clamped.
    bad = dict(VALID_STRATEGY, food_priority=7.0)
    with pytest.raises(StrategyValidationError):
        parse_strategy(bad)

    bad = dict(VALID_STRATEGY, strategy="chaotic")
    with pytest.raises(StrategyValidationError):
        parse_strategy(bad)

    bad = dict(VALID_STRATEGY, reason=12345)
    with pytest.raises(StrategyValidationError):
        parse_strategy(bad)


def test_parse_strategy_allows_valid_names():
    assert "defensive" in STRATEGIES and "balanced" in STRATEGIES
    for name in STRATEGIES:
        assert parse_strategy(dict(VALID_STRATEGY, strategy=name)).strategy == name


def _snapshot() -> dict:
    engine = make_engine(seed=30, max_foods=3)
    from snake_arena.game.environment import build_state_view

    view = build_state_view(engine.state, "ai")
    return build_snapshot(engine.state, "ai", view, {})


def test_fallback_strategy_is_deterministic_and_valid():
    snap = _snapshot()
    s1, s2 = fallback_strategy(snap), fallback_strategy(snap)
    assert s1 == s2
    assert s1.strategy in STRATEGIES
    assert s1.source == "fallback"
    # It round-trips through the strict parser.
    parse_strategy(s1.to_dict())


def test_snapshot_hash_stable_and_sensitive():
    snap = _snapshot()
    assert snapshot_hash(snap) == snapshot_hash(dict(snap))
    changed = dict(snap)
    changed["tick"] = snap["tick"] + 1
    assert snapshot_hash(changed) != snapshot_hash(snap)


def test_format_messages_contains_schema():
    system, user = format_messages(_snapshot())
    assert '"strategy"' in system
    assert "JSON" in system
    assert "tick" in user
