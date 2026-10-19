"""LLM strategic planner: snapshot building, prompting, background worker.

Design rules enforced here:
    * the LLM never sees or controls per-frame movement -- it only sets
      strategic parameters consumed by the DecisionManager;
    * inference runs on a daemon thread with a latest-request slot (older
      requests are coalesced away), rate-limited by ``min_interval_seconds``;
    * identical snapshots are not re-sent (hash cache);
    * any failure (missing Ollama, missing model, timeout, invalid JSON,
      schema violation) yields a deterministic, heuristic fallback strategy;
    * the thread is bounded: one worker, bounded queue, explicit stop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from ..agents.base import StrategyHints
from .ollama_client import OllamaClient, OllamaError
from .schemas import DEFAULT_STRATEGY, STRATEGIES, Strategy, parse_strategy

if TYPE_CHECKING:
    from ..game.environment import StateView
    from ..game.state import GameState

log = logging.getLogger("snake_arena.llm.strategist")

SYSTEM_PROMPT = """You are the strategic planner for an AI snake in a competitive Snake Arena. \
You do NOT control movement directly. Each turn you receive a compact JSON snapshot of the game \
and must choose a strategic posture that the movement system will follow.

Respond with ONLY a JSON object, no prose, exactly this schema:
{
  "strategy": "defensive" | "balanced" | "aggressive" | "greedy" | "survival",
  "food_priority": number in [0,1],
  "survival_priority": number in [0,1],
  "risk_tolerance": number in [0,1],
  "target_food_id": string id from the foods list, or null,
  "reason": "one short sentence"
}

Guidance: "greedy" chases high-value food, "aggressive" plays fast and risky, "defensive" avoids \
the opponent and tight corridors, "survival" prioritizes escaping danger now. Set risk_tolerance \
low when safe routes are scarce. Pick target_food_id only from the provided food ids."""


# -- snapshot ----------------------------------------------------------------


def build_snapshot(
    state: "GameState",
    snake_id: str,
    view: "StateView",
    safe_routes: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Compact, relevant game snapshot (never full history)."""
    me = view.me
    head = me.head
    foods = []
    ranked = sorted(
        state.foods,
        key=lambda f: (f.value if f.value > 0 else -1000) / (1 + abs(f.pos[0] - head[0]) + abs(f.pos[1] - head[1])),
        reverse=True,
    )
    for food in ranked[:6]:
        dist = abs(food.pos[0] - head[0]) + abs(food.pos[1] - head[1])
        contested = any(
            abs(opp.head[0] - food.pos[0]) + abs(opp.head[1] - food.pos[1]) <= dist
            for opp in view.opponents
        )
        foods.append(
            {
                "id": food.id,
                "type": food.type_name,
                "value": food.value,
                "risk": food.risk,
                "dist": dist,
                "remaining_ticks": max(0, food.lifetime_ticks - (state.tick - food.spawned_tick)),
                "contested": contested,
            }
        )

    snapshot: dict[str, Any] = {
        "board": {"w": state.width, "h": state.height},
        "tick": state.tick,
        "time_left_seconds": max(0, round((state.max_ticks - state.tick) / max(1, state.moves_per_second), 1)),
        "you": {
            "id": snake_id,
            "score": me.score,
            "length": me.length,
            "steps_since_food": me.steps_since_food,
        },
        "foods": foods,
    }
    if view.opponents:
        opp = min(view.opponents, key=lambda o: abs(o.head[0] - head[0]) + abs(o.head[1] - head[1]))
        opp_dist = abs(opp.head[0] - head[0]) + abs(opp.head[1] - head[1])
        snapshot["opponent"] = {
            "id": opp.id,
            "score": opp.score,
            "length": opp.length,
            "head_distance": opp_dist,
        }
        if opp_dist <= 4:
            danger = "high"
        elif opp_dist <= 9:
            danger = "medium"
        else:
            danger = "low"
        snapshot["danger"] = danger
    else:
        snapshot["danger"] = "low"
    if safe_routes:
        snapshot["safe_routes"] = {k: bool(v) for k, v in list(safe_routes.items())[:8]}
    return snapshot


def snapshot_hash(snapshot: dict) -> str:
    """Stable hash of the strategically-relevant snapshot content.

    Position-quantized (we round nothing here because all fields are already
    small ints or coarse floats) so unchanged states skip redundant calls.
    """
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def format_messages(snapshot: dict) -> tuple[str, str]:
    user = (
        "Game snapshot:\n"
        + json.dumps(snapshot, separators=(",", ":"))
        + "\n\nReply with ONLY the JSON strategy object now."
    )
    return SYSTEM_PROMPT, user


# -- deterministic fallback -----------------------------------------------------


def fallback_strategy(snapshot: dict) -> Strategy:
    """Heuristic strategy used whenever the LLM is unavailable/invalid.

    Rules (documented in AI_ALGORITHMS.md):
        high danger (opponent within 4 and not shorter)  -> defensive
        a very valuable food within 7 ticks              -> greedy
        hungry for a long time                           -> greedy
        otherwise                                        -> balanced
    """
    danger = snapshot.get("danger", "low")
    opponent = snapshot.get("opponent")
    you = snapshot.get("you", {})
    foods = snapshot.get("foods", [])

    if danger == "high" and opponent and opponent.get("length", 0) >= you.get("length", 0):
        return Strategy("defensive", 0.45, 0.85, 0.20, None,
                        "Fallback: opponent nearby and not smaller; play defensively.",
                        source="fallback")
    for food in foods:
        if food.get("value", 0) >= 50 and food.get("dist", 99) <= 7:
            return Strategy(
                "greedy", 0.9, 0.5, 0.6, food.get("id"),
                f"Fallback: valuable {food.get('type')} food is close; pursue it.",
                source="fallback",
            )
    if you.get("steps_since_food", 0) > 100:
        return Strategy("greedy", 0.85, 0.6, 0.5, None,
                        "Fallback: hungry; prioritize food.", source="fallback")
    return Strategy("balanced", 0.6, 0.6, 0.35, None,
                    "Fallback: default balanced posture.", source="fallback")


# -- background worker -----------------------------------------------------------


@dataclass
class StrategyResult:
    hints: StrategyHints
    source: str = "fallback"      # llm | fallback
    error: str | None = None
    latency_ms: float = 0.0
    model: str = ""
    snapshot_hash: str = ""
    timestamp: float = field(default_factory=time.time)


class StrategyWorker(threading.Thread):
    """Single daemon thread performing rate-limited LLM strategy calls."""

    def __init__(
        self,
        client: OllamaClient,
        min_interval: float = 3.0,
        on_result: Callable[[StrategyResult], None] | None = None,
        availability_probe_every: float = 15.0,
    ) -> None:
        super().__init__(daemon=True, name="llm-strategist")
        self.client = client
        self.min_interval = max(0.5, min_interval)
        self.on_result = on_result
        self.availability_probe_every = availability_probe_every

        self._cv = threading.Condition()
        self._pending: dict | None = None
        self._stop_flag = False

        self._lock = threading.Lock()
        self._last_result: StrategyResult | None = None
        self._last_hash: str = ""
        self._last_call_time = 0.0
        self._available: bool | None = None
        self._last_probe = 0.0
        self.status = "idle"           # idle|waiting|calling|ok|error|disabled
        self.calls_made = 0
        self.calls_failed = 0

    # -- public API (thread-safe) -------------------------------------------------

    def submit(self, snapshot: dict) -> None:
        """Queue the latest snapshot; older unprocessed ones are replaced."""
        h = snapshot_hash(snapshot)
        with self._lock:
            if h == self._last_hash:
                return  # strategically unchanged: skip redundant inference
        with self._cv:
            if self._stop_flag:
                return
            self._pending = snapshot
            self._cv.notify()

    def probe_availability(self, force: bool = False) -> bool:
        now = time.time()
        with self._lock:
            if not force and self._available is not None and now - self._last_probe < self.availability_probe_every:
                return self._available
        ok, detail = self.client.is_available()
        with self._lock:
            self._available = ok
            self._last_probe = now
            if not ok:
                self.status = "error"
        return ok

    def current_result(self) -> StrategyResult | None:
        with self._lock:
            return self._last_result

    def current_hints(self) -> StrategyHints | None:
        result = self.current_result()
        return result.hints if result else None

    def stop(self) -> None:
        with self._cv:
            self._stop_flag = True
            self._cv.notify()

    # -- worker loop --------------------------------------------------------------

    def run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop_flag:
                    self.status = "idle" if self.status != "error" else "error"
                    self._cv.wait(timeout=1.0)
                if self._stop_flag:
                    return
                snapshot = self._pending
                self._pending = None

            h = snapshot_hash(snapshot)
            with self._lock:
                if h == self._last_hash:
                    continue
                wait = self._last_call_time + self.min_interval - time.time()
            if wait > 0:
                self.status = "waiting"
                time.sleep(wait)

            self.status = "calling"
            result = self._call_llm(snapshot, h)
            with self._lock:
                self._last_call_time = time.time()
                self._last_result = result
                self._last_hash = h
            self.status = "ok" if result.error is None else "error"
            if self.on_result is not None:
                try:
                    self.on_result(result)
                except Exception:  # noqa: BLE001 - UI must not kill the worker
                    log.exception("strategy callback failed")

    def _call_llm(self, snapshot: dict, h: str) -> StrategyResult:
        system, user = format_messages(snapshot)
        t0 = time.perf_counter()
        try:
            obj = self.client.chat_json(system, user)
            strategy = parse_strategy(obj)
            # Validate a pinned target against the foods actually on board.
            valid_ids = {f["id"] for f in snapshot.get("foods", [])}
            if strategy.target_food_id and strategy.target_food_id not in valid_ids:
                strategy.target_food_id = None
                strategy.reason = (strategy.reason + " [pinned target not on board]").strip()[:300]
        except OllamaError as exc:
            self.calls_failed += 1
            return StrategyResult(
                hints=_fallback_hints(fallback_strategy(snapshot)),
                source="fallback",
                error=f"Ollama: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                model=self.client.model,
                snapshot_hash=h,
            )
        except Exception as exc:  # schema or parse errors
            self.calls_failed += 1
            return StrategyResult(
                hints=_fallback_hints(fallback_strategy(snapshot)),
                source="fallback",
                error=f"invalid LLM response: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                model=self.client.model,
                snapshot_hash=h,
            )
        self.calls_made += 1
        hints = StrategyHints(
            strategy=strategy.strategy,
            food_priority=strategy.food_priority,
            survival_priority=strategy.survival_priority,
            risk_tolerance=strategy.risk_tolerance,
            target_food_id=strategy.target_food_id,
            reason=strategy.reason,
            source="llm",
            latency_ms=self.client.last_latency_ms,
        )
        return StrategyResult(
            hints=hints,
            source="llm",
            error=None,
            latency_ms=self.client.last_latency_ms,
            model=self.client.model,
            snapshot_hash=h,
        )


def _fallback_hints(strategy: Strategy) -> StrategyHints:
    return StrategyHints(
        strategy=strategy.strategy,
        food_priority=strategy.food_priority,
        survival_priority=strategy.survival_priority,
        risk_tolerance=strategy.risk_tolerance,
        target_food_id=strategy.target_food_id,
        reason=strategy.reason,
        source="fallback",
    )
