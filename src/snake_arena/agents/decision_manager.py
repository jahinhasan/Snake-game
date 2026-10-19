"""Hybrid Decision Manager: the arbitration pipeline.

Pipeline (identical for every mode):

    Game state
      -> Observation (StateView, built once per tick)
      -> Safety pre-check of all four actions ( SafetyValidator )
      -> Strategy  (LLM strategist or deterministic fallback -> StrategyHints)
      -> Policy    (A* / BFS / rule-based / DQN / hybrid scoring)
      -> Candidate action
      -> Safety validation + fallback selection
      -> Final action -> Game engine
      -> Decision log (diagnostics + metrics)

Documented, deterministic arbitration policy
--------------------------------------------
* The LLM (when available) only produces *strategy parameters*; it can never
  emit a move. Those parameters weight food utilities (A*), gate risky DQN
  actions (RL), and re-weight the hybrid score.
* RL risk gating (modes ``rl``, ``llm_rl``): actions are ranked by Q-value.
  Walking the ranking, the first action whose reachable free area is at least
      min_area = snake_length * (0.8 + 0.9 * survival_priority)
                             * (1 - 0.5 * risk_tolerance)
  is chosen (aggressive/greedy strategies multiply min_area by 0.7). If no
  action meets the bar, the safest action wins. The safety validator's hard
  rules always apply on top.
* Hybrid mode scores each *safe* action with a convex combination of three
  comparable, normalized signals:
      score(a) = w_rl   * q_norm(a)      (min-max of Q across actions)
               + w_path * path_norm(a)   (1 - d(a, target) / d_max)
               + w_space * space_norm(a) (free area, min-max)
  where (w_rl, w_path, w_space) are config weights re-weighted by the
  strategy: survival_priority scales space, food_priority scales path.
  Q-values and distances are normalized before mixing precisely so we never
  average incompatible raw quantities.
* If the LLM is unavailable, every LLM mode degrades to its non-LLM policy
  driven by the deterministic fallback strategy; play never stops.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..config import AppConfig
from ..game.environment import StateView, build_state_view
from ..game.types import ALL_ACTIONS, DIR_VECTORS, Action, Cell, manhattan
from ..llm.schemas import DEFAULT_STRATEGY, Strategy
from ..llm.strategist import (
    StrategyResult,
    StrategyWorker,
    _fallback_hints,
    build_snapshot,
    fallback_strategy,
)
from ..safety.validator import SafetyReport, SafetyValidator
from .astar import AStarAgent
from .base import AgentDecision, BaseAgent, StrategyHints
from .bfs import BFSAgent
from .rule_based import RuleBasedAgent
from .rl_agent import RLAgent

if TYPE_CHECKING:
    from ..game.state import GameState
    from ..llm.ollama_client import OllamaClient

log = logging.getLogger("snake_arena.decision")


@dataclass
class DecisionLog:
    """One tick's full decision trace for the diagnostics panel and metrics."""

    tick: int
    mode: str
    proposed_action: str
    final_action: str
    safety_intervened: bool
    safety_rejected: list[tuple[str, str]] = field(default_factory=list)
    strategy_source: str = "default"
    strategy_name: str = "balanced"
    strategy_reason: str = ""
    llm_error: str | None = None
    target_food_id: str | None = None
    path_length: int = 0
    q_values: dict[str, float] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    computation_ms: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class DecisionManager:
    """Coordinates strategy, policy, and safety into one validated action."""

    def __init__(
        self,
        mode: str,
        config: AppConfig,
        rl_agent: RLAgent | None = None,
        validator: SafetyValidator | None = None,
        strategist_worker: StrategyWorker | None = None,
        llm_client: "OllamaClient | None" = None,
    ) -> None:
        self.mode = mode
        self.config = config
        self.validator = validator or SafetyValidator(config.safety)
        self.strategist_worker = strategist_worker
        self.llm_client = llm_client

        self.rule_agent = RuleBasedAgent(flood_limit=config.safety.flood_fill_limit)
        self.astar_agent = AStarAgent(
            turn_penalty=config.decision.astar.turn_penalty,
            risk_aversion=config.decision.astar.risk_aversion,
            value_weight=config.decision.astar.value_weight,
            flood_limit=config.safety.flood_fill_limit,
            survival_escape_ratio=config.decision.astar.survival_escape_ratio,
        )
        self.bfs_agent = BFSAgent(flood_limit=config.safety.flood_fill_limit)
        self.rl_agent = rl_agent

        self.diagnostics: deque[DecisionLog] = deque(maxlen=60)
        self.current_log: DecisionLog | None = None
        # Sync-LLM state (benchmark path): simple hash + interval throttle.
        self._sync_last_hash: str = ""
        self._sync_last_time: float = 0.0
        self._sync_last_result: StrategyResult | None = None
        # Aggregate stats for metrics.
        self.stats = {
            "decisions": 0,
            "safety_interventions": 0,
            "llm_calls": 0,
            "llm_failures": 0,
            "llm_latency_ms_total": 0.0,
            "gating_activations": 0,
        }

    # -- lifecycle ------------------------------------------------------------

    def reset_episode(self) -> None:
        self.diagnostics.clear()
        self.current_log = None
        for agent in (self.rule_agent, self.astar_agent, self.bfs_agent):
            agent.reset()

    @property
    def uses_llm(self) -> bool:
        return self.mode in ("llm_astar", "llm_rl", "hybrid")

    # -- strategy ---------------------------------------------------------------

    def _snapshot(self, state: "GameState", snake_id: str, view: StateView, safe_routes: dict[str, bool] | None) -> dict:
        return build_snapshot(state, snake_id, view, safe_routes)

    def _hints(self, state: "GameState", snake_id: str, view: StateView, safe_routes: dict[str, bool] | None) -> StrategyHints:
        """Strategy from the worker (UI), sync client (benchmark), or fallback."""
        if not self.uses_llm:
            hints = _default_hints()
            return hints

        snapshot = self._snapshot(state, snake_id, view, safe_routes)

        if self.strategist_worker is not None:
            self.strategist_worker.submit(snapshot)
            result = self.strategist_worker.current_result()
            if result is not None:
                if result.error:
                    self.stats["llm_failures"] += 1
                else:
                    self.stats["llm_calls"] += 1
                    self.stats["llm_latency_ms_total"] += result.latency_ms
                return result.hints
            return _fallback_hints(fallback_strategy(snapshot))

        if self.llm_client is not None:
            return self._hints_sync(snapshot)

        # No LLM anywhere: deterministic fallback keeps the mode functional.
        return _fallback_hints(fallback_strategy(snapshot))

    def _hints_sync(self, snapshot: dict) -> StrategyHints:
        from ..llm.strategist import snapshot_hash

        h = snapshot_hash(snapshot)
        now = time.time()
        if h == self._sync_last_hash and self._sync_last_result is not None:
            return self._sync_last_result.hints
        if now - self._sync_last_time < self.config.llm.min_interval_seconds and self._sync_last_result is not None:
            return self._sync_last_result.hints
        from ..llm.strategist import format_messages

        system, user = format_messages(snapshot)
        t0 = time.perf_counter()
        try:
            obj = self.llm_client.chat_json(system, user)
            strategy = parse_and_validate(obj, snapshot)
            hints = _strategy_hints(strategy, self.llm_client.last_latency_ms)
            error = None
            self.stats["llm_calls"] += 1
            self.stats["llm_latency_ms_total"] += self.llm_client.last_latency_ms
        except Exception as exc:  # noqa: BLE001 - every failure degrades gracefully
            strategy = fallback_strategy(snapshot)
            hints = _fallback_hints(strategy)
            error = str(exc)
            self.stats["llm_failures"] += 1
        latency = (time.perf_counter() - t0) * 1000.0
        self._sync_last_hash = h
        self._sync_last_time = now
        self._sync_last_result = StrategyResult(
            hints=hints,
            source=hints.source,
            error=error,
            latency_ms=latency,
            model=self.llm_client.model if self.llm_client else "",
            snapshot_hash=h,
        )
        return hints

    # -- decision ----------------------------------------------------------------

    def decide(self, state: "GameState", snake_id: str) -> Action:
        t0 = time.perf_counter()
        view = build_state_view(state, snake_id)
        reports = self.validator.evaluate_all(state, snake_id, view)
        safe_actions = [a for a, r in reports.items() if r.safe]

        # Optional route-availability info for the LLM snapshot.
        safe_routes: dict[str, bool] = {}
        if self.uses_llm:
            for food in view.foods_sorted[:6]:
                if food.value <= 0:
                    continue
                reachable = any(
                    _step_towards(state.snakes[snake_id].head, food.pos) == a for a in safe_actions
                )
                safe_routes[food.id] = reachable

        hints = self._hints(state, snake_id, view, safe_routes)
        proposed = self._policy(state, snake_id, view, hints, reports, safe_actions)
        fallback_rank = proposed_ranking(proposed, reports)

        final_action, report = self.validator.select_safe_action(
            state,
            snake_id,
            proposed.action,
            fallback_order=fallback_rank,
            view=view,
            target=proposed.target_pos,
        )
        if report.intervention:
            self.stats["safety_interventions"] += 1
            state.snakes[snake_id].rejected_moves += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log_entry = DecisionLog(
            tick=state.tick,
            mode=self.mode,
            proposed_action=proposed.action.name.lower(),
            final_action=final_action.name.lower(),
            safety_intervened=report.intervention,
            safety_rejected=list(report.rejected),
            strategy_source=hints.source,
            strategy_name=hints.strategy,
            strategy_reason=hints.reason,
            llm_error=self._last_llm_error(),
            target_food_id=proposed.target_food_id,
            path_length=proposed.path_length,
            q_values={a.name.lower(): round(v, 3) for a, v in (proposed.q_values or {}).items()},
            components=proposed.diagnostics.get("hybrid_components", {}),
            notes=[*proposed.notes],
            computation_ms=elapsed_ms,
            latency_ms=hints.latency_ms,
        )
        if report.intervention:
            for name, reason in report.rejected:
                log_entry.notes.append(f"safety rejected {name}: {reason}")
        self.stats["decisions"] += 1
        self.diagnostics.append(log_entry)
        self.current_log = log_entry
        return final_action

    # -- policies ------------------------------------------------------------------

    def _policy(
        self,
        state: "GameState",
        snake_id: str,
        view: StateView,
        hints: StrategyHints,
        reports: dict[Action, SafetyReport],
        safe_actions: list[Action],
    ) -> AgentDecision:
        if self.mode == "rule_based":
            return self.rule_agent.decide(state, snake_id, view, hints)
        if self.mode in ("astar", "llm_astar"):
            return self.astar_agent.decide(state, snake_id, view, hints)
        if self.mode == "bfs":
            return self.bfs_agent.decide(state, snake_id, view, hints)
        if self.mode in ("rl", "llm_rl"):
            assert self.rl_agent is not None, "RL modes need a configured RL agent"
            decision = self.rl_agent.decide(state, snake_id, view, hints)
            return self._gate_rl(state, snake_id, view, hints, reports, decision)
        if self.mode == "hybrid":
            return self._hybrid(state, snake_id, view, hints, reports, safe_actions)
        raise ValueError(f"unknown decision mode: {self.mode}")

    def _gate_rl(
        self,
        state: "GameState",
        snake_id: str,
        view: StateView,
        hints: StrategyHints,
        reports: dict[Action, SafetyReport],
        decision: AgentDecision,
    ) -> AgentDecision:
        """Re-rank the DQN proposal by strategy-dependent risk appetite."""
        me = view.me
        ranking = [Action[name.upper()] for name in decision.diagnostics.get("q_ranking", [])]
        if not ranking:
            ranking = [decision.action]
        min_area = me.length * (0.8 + 0.9 * hints.survival_priority)
        min_area *= 1.0 - 0.5 * hints.risk_tolerance
        if hints.is_aggressive:
            min_area *= 0.7
        gated: Action | None = None
        for action in ranking:
            report = reports.get(action)
            if report is None or not report.safe:
                continue
            if report.free_area >= min_area:
                gated = action
                break
        chosen = gated if gated is not None else ranking[0]
        if gated is not None and gated != decision.action:
            self.stats["gating_activations"] += 1
            decision.notes.append(
                f"risk gate: Q-best {decision.action.name.lower()} too tight "
                f"(area<{min_area:.0f}); chose {gated.name.lower()}"
            )
        elif gated is None:
            decision.notes.append("risk gate: no action met the area bar; using Q-best")
        decision.action = chosen
        decision.diagnostics["min_area"] = round(min_area, 1)
        return decision

    def _hybrid(
        self,
        state: "GameState",
        snake_id: str,
        view: StateView,
        hints: StrategyHints,
        reports: dict[Action, SafetyReport],
        safe_actions: list[Action],
    ) -> AgentDecision:
        """Full hybrid: normalized RL Q + A* path quality + free space."""
        assert self.rl_agent is not None, "hybrid mode needs a configured RL agent"
        rl_decision = self.rl_agent.decide(state, snake_id, view, hints)
        astar_decision = self.astar_agent.decide(state, snake_id, view, hints)

        me = view.me
        w = self.config.decision.hybrid
        w_rl, w_path, w_space = w.w_rl, w.w_path, w.w_space
        # Strategy modulation (documented in the module header).
        w_path *= 0.4 + 1.2 * hints.food_priority
        w_space *= 0.5 + 1.0 * hints.survival_priority
        total = w_rl + w_path + w_space
        w_rl, w_path, w_space = w_rl / total, w_path / total, w_space / total

        target = astar_decision.target_pos or (view.target.pos if view.target else None)
        q = rl_decision.q_values or {}
        safe = safe_actions or [rl_decision.action]

        def q_norm(a: Action) -> float:
            if len(q) < 2:
                return 0.0
            vals = [q[x] for x in ALL_ACTIONS if x in q]
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-9:
                return 0.5
            return (q[a] - lo) / (hi - lo)

        def dist(a: Action) -> float:
            if target is None:
                return 0.0
            dx, dy = DIR_VECTORS[a]
            nxt = (me.head[0] + dx, me.head[1] + dy)
            return float(manhattan(nxt, target))

        d_max = max((dist(a) for a in safe), default=1.0) or 1.0
        area_max = max((reports[a].free_area for a in safe), default=1) or 1

        scores: dict[Action, float] = {}
        components: dict[str, float] = {}
        for a in safe:
            path_term = 1.0 - dist(a) / d_max if target is not None else 0.0
            space_term = reports[a].free_area / area_max
            scores[a] = w_rl * q_norm(a) + w_path * path_term + w_space * space_term
            components[a.name.lower()] = round(scores[a], 3)

        best = max(scores, key=lambda a: scores[a])
        notes = [f"hybrid weights: rl={w_rl:.2f} path={w_path:.2f} space={w_space:.2f}"]
        if target is not None:
            notes.append(f"target from A*: {target}")
        return AgentDecision(
            agent="hybrid",
            action=best,
            path=astar_decision.path,
            target_food_id=astar_decision.target_food_id,
            target_pos=target,
            q_values=rl_decision.q_values,
            notes=[*notes, *astar_decision.notes],
            diagnostics={
                "mode": "hybrid",
                "hybrid_components": components,
                "trained": rl_decision.diagnostics.get("trained", False),
            },
            computation_ms=rl_decision.computation_ms + astar_decision.computation_ms,
        )

    # -- misc ---------------------------------------------------------------------

    def _last_llm_error(self) -> str | None:
        if self.strategist_worker is not None:
            result = self.strategist_worker.current_result()
            return result.error if result else None
        if self._sync_last_result is not None:
            return self._sync_last_result.error
        return None

    def describe(self) -> str:
        mode_names = {
            "rule_based": "Rule-Based Agent",
            "astar": "A* Agent",
            "bfs": "BFS Agent",
            "rl": "RL Agent (DQN)",
            "llm_astar": "LLM + A* Agent",
            "llm_rl": "LLM + RL Agent",
            "hybrid": "Full Hybrid Agent",
        }
        return mode_names.get(self.mode, self.mode)


# -- helpers ---------------------------------------------------------------------


def _default_hints() -> StrategyHints:
    return StrategyHints(
        strategy=DEFAULT_STRATEGY.strategy,
        food_priority=DEFAULT_STRATEGY.food_priority,
        survival_priority=DEFAULT_STRATEGY.survival_priority,
        risk_tolerance=DEFAULT_STRATEGY.risk_tolerance,
        target_food_id=None,
        reason="",
        source="default",
    )


def _strategy_hints(strategy: Strategy, latency_ms: float) -> StrategyHints:
    return StrategyHints(
        strategy=strategy.strategy,
        food_priority=strategy.food_priority,
        survival_priority=strategy.survival_priority,
        risk_tolerance=strategy.risk_tolerance,
        target_food_id=strategy.target_food_id,
        reason=strategy.reason,
        source="llm",
        latency_ms=latency,
    )


def parse_and_validate(obj: dict, snapshot: dict) -> Strategy:
    """Parse an LLM strategy object and validate any pinned food target."""
    from ..llm.schemas import parse_strategy

    strategy = parse_strategy(obj)
    valid_ids = {f["id"] for f in snapshot.get("foods", [])}
    if strategy.target_food_id and strategy.target_food_id not in valid_ids:
        strategy.target_food_id = None
        strategy.reason = (strategy.reason + " [pinned target not on board]").strip()[:300]
    return strategy


def proposed_ranking(decision: AgentDecision, reports: dict[Action, SafetyReport]) -> list[Action]:
    """Fallback order for the safety layer: Q ranking, else path heading."""
    ranking: list[Action] = []
    if decision.q_values:
        ranking = sorted(ALL_ACTIONS, key=lambda a: decision.q_values.get(a, float("-inf")), reverse=True)
    elif len(decision.path) > 1:
        nxt = decision.path[1]
        head = decision.path[0]
        ranking = [_step_towards(head, nxt)]
    ranking.append(decision.action)
    seen: set[Action] = set()
    ordered: list[Action] = []
    for a in ranking:
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    ordered.extend(a for a in ALL_ACTIONS if a not in seen)
    return ordered


def _step_towards(a: Cell, b: Cell) -> Action:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0 and dy == 0:
        return Action.RIGHT
    if abs(dx) >= abs(dy):
        return Action.RIGHT if dx > 0 else Action.LEFT
    return Action.DOWN if dy > 0 else Action.UP
