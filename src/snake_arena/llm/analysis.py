"""Post-game analysis: deterministic summary + optional LLM narrative.

``summarize_game`` builds the structured summary from the actual game record
(measured facts). ``build_analysis_messages`` renders it for the LLM, whose
output is validated (length, no control characters) and always presented in
the UI as model-generated interpretation, never as measured data.
"""

from __future__ import annotations

from typing import Any

ANALYSIS_SYSTEM_PROMPT = (
    "You are an AI coach analyzing one finished Snake Arena game. You receive the measured "
    "game record as JSON. Write a concise analysis (4-7 sentences) covering: what the AI did "
    "well, what went wrong (deaths, traps, missed food), which risky decisions stand out, and "
    "one concrete improvement suggestion. Refer only to facts present in the record. "
    "Do not invent statistics."
)


def summarize_game(record: dict[str, Any]) -> dict[str, Any]:
    """Structured non-LLM summary from the real game record."""
    snakes = record.get("snakes", {})
    winner_ids = record.get("winner_ids", [])
    summary: dict[str, Any] = {
        "duration_ticks": record.get("ticks", 0),
        "duration_seconds": record.get("seconds", 0.0),
        "outcome": record.get("outcome", "unknown"),
        "winner": [snakes[w]["name"] for w in winner_ids if w in snakes] or None,
        "snakes": {},
    }
    for sid, s in snakes.items():
        summary["snakes"][s["name"]] = {
            "score": s["score"],
            "final_length": s["length"],
            "food_by_type": dict(s.get("foods_eaten", {})),
            "traps_hit": s.get("foods_eaten", {}).get("trap", 0),
            "survived": s.get("survived", False),
            "death_reason": s.get("death_reason"),
            "is_human": s.get("is_human", False),
        }
    ai = record.get("ai_summary")
    if ai:
        summary["ai_summary"] = ai
    return summary


def build_analysis_messages(summary: dict[str, Any]) -> tuple[str, str]:
    user = "Game record (measured facts):\n" + json_compact(summary) + "\n\nWrite your analysis now."
    return ANALYSIS_SYSTEM_PROMPT, user


def parse_analysis(text: str) -> str:
    """Validate and clean LLM analysis text; raises ValueError when unusable."""
    if "<think>" in text:
        end = text.find("</think>")
        text = text[end + len("</think>"):] if end != -1 else text.split("</think>")[-1]
    cleaned = " ".join(text.split())
    if len(cleaned) < 30:
        raise ValueError("analysis too short to be useful")
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    return cleaned[:4000]


def json_compact(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
