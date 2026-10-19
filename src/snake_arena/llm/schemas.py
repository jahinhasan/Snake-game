"""Strict validation for LLM strategic responses (pydantic-free equivalent).

The LLM must answer with a JSON object; this module enforces exact types,
allowed values, and ranges. Invalid responses are *rejected* (never clamped
silently) so the caller can fall back to the deterministic strategy and the
UI can show the rejection reason. Deliberately dependency-free: validation
logic is small, explicit, and fully unit-tested with mocked LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass

STRATEGIES: tuple[str, ...] = (
    "defensive",
    "balanced",
    "aggressive",
    "greedy",
    "survival",
)

STRATEGY_LABELS: dict[str, str] = {
    "defensive": "Defensive",
    "balanced": "Balanced",
    "aggressive": "Aggressive",
    "greedy": "High-value pursuit",
    "survival": "Survival / escape",
}


class StrategyValidationError(ValueError):
    pass


@dataclass
class Strategy:
    """A validated strategic posture (what the DecisionManager consumes)."""

    strategy: str = "balanced"
    food_priority: float = 0.5
    survival_priority: float = 0.5
    risk_tolerance: float = 0.5
    target_food_id: str | None = None
    reason: str = ""
    source: str = "default"
    latency_ms: float = 0.0
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "food_priority": self.food_priority,
            "survival_priority": self.survival_priority,
            "risk_tolerance": self.risk_tolerance,
            "target_food_id": self.target_food_id,
            "reason": self.reason,
            "source": self.source,
        }


DEFAULT_STRATEGY = Strategy()

_EXAMPLE = {
    "strategy": "defensive",
    "food_priority": 0.65,
    "survival_priority": 0.90,
    "risk_tolerance": 0.25,
    "target_food_id": None,
    "reason": "The opponent is close and the high-value food has a dangerous approach.",
}


def parse_strategy(obj: object) -> Strategy:
    """Validate a decoded JSON object into a Strategy; raise on any violation."""
    if not isinstance(obj, dict):
        raise StrategyValidationError("response is not a JSON object")

    unknown = set(obj) - set(_EXAMPLE)
    # Unknown keys are tolerated (models add commentary fields) but required
    # ones are checked strictly below.

    strategy = obj.get("strategy")
    if strategy not in STRATEGIES:
        raise StrategyValidationError(
            f"strategy must be one of {list(STRATEGIES)}, got {strategy!r}"
        )

    def _range(key: str) -> float:
        value = obj.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StrategyValidationError(f"{key} must be a number, got {value!r}")
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise StrategyValidationError(f"{key}={value} outside [0, 1]")
        return value

    food_priority = _range("food_priority")
    survival_priority = _range("survival_priority")
    risk_tolerance = _range("risk_tolerance")

    target = obj.get("target_food_id", None)
    if target is not None and not isinstance(target, str):
        raise StrategyValidationError(f"target_food_id must be null or a string, got {target!r}")
    if isinstance(target, str) and not target.strip():
        target = None

    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        raise StrategyValidationError(f"reason must be a string, got {reason!r}")
    reason = reason.strip()[:300]

    return Strategy(
        strategy=strategy,
        food_priority=food_priority,
        survival_priority=survival_priority,
        risk_tolerance=risk_tolerance,
        target_food_id=target,
        reason=reason,
        source="llm",
    )


def extract_json_object(text: str) -> dict:
    """Pull the first balanced JSON object out of raw model text.

    Handles Qwen3 ``<think>...</think>`` blocks and stray commentary by
    scanning for the first '{' and its matching brace.
    """
    import json

    if "<think>" in text:
        end = text.find("</think>")
        text = text[end + len("</think>"):] if end != -1 else text.split("</think>")[-1]
    start = text.find("{")
    if start == -1:
        raise StrategyValidationError("no JSON object found in model output")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    parsed = json.loads(snippet)
                except json.JSONDecodeError as exc:
                    raise StrategyValidationError(f"invalid JSON: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise StrategyValidationError("decoded JSON is not an object")
                return parsed
    raise StrategyValidationError("unbalanced JSON object in model output")
