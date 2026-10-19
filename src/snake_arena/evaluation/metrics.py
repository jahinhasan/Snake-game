"""Metric definitions for agent evaluation.

Precise definitions (used identically by the benchmark, dashboard, and docs):

* ``score``           final snake score (sum of collected food values, traps negative)
* ``mean/median/std`` statistics over the agent's N episodes (sample counts always reported)
* ``survival_rate``   fraction of episodes the snake was still alive at episode end
* ``death_reasons``   histogram of terminal causes: wall | self | opponent | head_on | timeout
* ``food_by_type``    total food items collected per configured type (traps included)
* ``trap_rate``       trap foods collected per 1000 ticks survived
* ``mean_decision_ms``mean agent think-time per tick (excludes engine step cost)
* ``llm_*``           calls, failures, and mean inference latency; 0/None when the
                      LLM was not part of that agent's pipeline
* ``win_rate``        NOT DEFINED for solo episodes (no adversary); it is reported
                      only for AI-vs-AI matches, where it is wins / matches played.

Variability is reported as std-dev and per-episode records are kept; we do
not claim statistical significance without an explicit test in EXPERIMENTS.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EpisodeRecord:
    agent: str
    seed: int
    score: int
    steps: int
    outcome: str
    death_reason: str | None
    length: int
    survived: bool
    foods: dict[str, int] = field(default_factory=dict)
    safety_interventions: int = 0
    mean_decision_ms: float = 0.0
    llm_calls: int = 0
    llm_failures: int = 0
    llm_latency_ms_mean: float = 0.0
    llm_available: bool = False
    winner: bool = False  # only meaningful in matches

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentSummary:
    agent: str
    episodes: int
    mean_score: float
    median_score: float
    std_score: float
    min_score: int
    max_score: int
    mean_steps: float
    survival_rate: float
    death_reasons: dict[str, int]
    food_by_type: dict[str, int]
    mean_length: float
    trap_rate_per_1k_ticks: float
    mean_decision_ms: float
    llm_calls: int
    llm_failures: float
    llm_latency_ms_mean: float
    llm_available: bool
    mean_path_length: float = 0.0
    win_rate: float | None = None  # matches only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def summarize(records: list[EpisodeRecord], agent: str = "", matches: bool = False) -> AgentSummary:
    if not records:
        raise ValueError("summarize needs at least one episode record")
    scores = [float(r.score) for r in records]
    steps = [float(r.steps) for r in records]
    lengths = [float(r.length) for r in records]
    var = sum((x - _mean(scores)) ** 2 for x in scores) / len(scores)
    death_hist: dict[str, int] = {}
    for r in records:
        key = r.death_reason or ("timeout" if r.survived else "unknown")
        death_hist[key] = death_hist.get(key, 0) + 1
    foods: dict[str, int] = {}
    for r in records:
        for ftype, count in r.foods.items():
            foods[ftype] = foods.get(ftype, 0) + count
    total_ticks = sum(r.steps for r in records)
    traps = foods.get("trap", 0)
    llm_records = [r for r in records if r.llm_calls or r.llm_failures]
    latencies = [r.llm_latency_ms_mean for r in llm_records if r.llm_latency_ms_mean > 0]
    win_rate = None
    if matches:
        win_rate = sum(1 for r in records if r.winner) / len(records)
    return AgentSummary(
        agent=agent or records[0].agent,
        episodes=len(records),
        mean_score=round(_mean(scores), 2),
        median_score=round(_median(scores), 2),
        std_score=round(var ** 0.5, 2),
        min_score=int(min(scores)),
        max_score=int(max(scores)),
        mean_steps=round(_mean(steps), 1),
        survival_rate=round(sum(1 for r in records if r.survived) / len(records), 3),
        death_reasons=death_hist,
        food_by_type=foods,
        mean_length=round(_mean(lengths), 2),
        trap_rate_per_1k_ticks=round(traps * 1000.0 / max(1, total_ticks), 3),
        mean_decision_ms=round(_mean([r.mean_decision_ms for r in records]), 3),
        llm_calls=sum(r.llm_calls for r in records),
        llm_failures=sum(r.llm_failures for r in records),
        llm_latency_ms_mean=round(_mean(latencies), 1),
        llm_available=any(r.llm_available for r in records),
        win_rate=win_rate,
    )


def summary_rows(summaries: list[AgentSummary]) -> list[dict[str, Any]]:
    rows = []
    for s in summaries:
        row = s.to_dict()
        row["death_reasons"] = ";".join(f"{k}:{v}" for k, v in sorted(s.death_reasons.items()))
        row["food_by_type"] = ";".join(f"{k}:{v}" for k, v in sorted(s.food_by_type.items()))
        rows.append(row)
    return rows
