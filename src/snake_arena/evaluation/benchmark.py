"""Automated benchmark runner: many headless episodes, real metrics.

Methodology (see EXPERIMENTS.md):
* every agent plays *solo* episodes (one snake vs the environment) with the
  same board, food config, tick cap, and identical per-episode seeds, so
  agent differences come only from the decision policy;
* LLM-powered agents use a smaller sample size (LLM inference is expensive)
  and their results carry an explicit ``llm_available`` flag: when Ollama is
  missing, they run on the deterministic fallback strategy and are labelled
  as such -- never as real LLM results;
* AI-vs-AI matches/tournaments are supported separately with win rates.

Everything here runs headless and synchronously; the UI wraps ``run`` in a
thread with progress callbacks and a cancel flag.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from ..agents.factory import build_manager
from ..config import AppConfig
from ..game.engine import GameEngine, SnakeSpec
from ..game.types import Action
from ..llm.ollama_client import OllamaClient
from .metrics import EpisodeRecord, summarize, summary_rows

log = logging.getLogger("snake_arena.benchmark")

AGENT_LABELS: dict[str, str] = {
    "rule_based": "Rule-Based",
    "astar": "A*",
    "bfs": "BFS",
    "rl": "RL (DQN)",
    "llm_astar": "LLM + A*",
    "llm_rl": "LLM + RL",
    "hybrid": "Full Hybrid",
}

LLM_MODES = {"llm_astar", "llm_rl", "hybrid"}


@dataclass
class BenchmarkProgress:
    agent: str = ""
    agent_index: int = 0
    agent_count: int = 0
    episode: int = 0
    episodes_for_agent: int = 0
    total_done: int = 0
    total_planned: int = 0
    message: str = ""


class BenchmarkRunner:
    def __init__(
        self,
        config: AppConfig,
        agents: list[str] | None = None,
        episodes: int | None = None,
        llm_episodes: int | None = None,
        base_seed: int | None = None,
        max_ticks: int | None = None,
        llm_client: OllamaClient | None = None,
        rl_agent=None,
        progress_cb: Callable[[BenchmarkProgress], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        bench = config.benchmark
        self.agents = agents or list(bench.agents)
        self.episodes = episodes if episodes is not None else bench.episodes_per_agent
        self.llm_episodes = llm_episodes if llm_episodes is not None else bench.llm_episodes_per_agent
        self.base_seed = base_seed if base_seed is not None else bench.base_seed
        self.max_ticks = max_ticks if max_ticks is not None else bench.max_ticks_per_episode
        self.llm_client = llm_client
        self.rl_agent = rl_agent
        self.progress_cb = progress_cb
        self.cancel_check = cancel_check or (lambda: False)

    # -- solo benchmark ----------------------------------------------------------

    def _make_engine(self, seed: int) -> GameEngine:
        return GameEngine(
            width=self.config.board.width,
            height=self.config.board.height,
            moves_per_second=self.config.game.moves_per_second,
            max_foods=self.config.game.max_foods,
            seed=seed,
            time_limit_seconds=self.config.game.time_limit_seconds,
            snake_specs=[SnakeSpec("ai", "AI", False, (max(2, self.config.board.width // 4), self.config.board.height // 2), Action.RIGHT)],
        )

    def run(self) -> dict:
        total = 0
        for agent in self.agents:
            total += self._episodes_for(agent)
        progress = BenchmarkProgress(agent_count=len(self.agents), total_planned=total)

        results: dict = {
            "run_id": None,  # filled by the caller (experiments.make_run_dir)
            "timestamp": time.time(),
            "meta": {
                "agents": list(self.agents),
                "episodes_per_agent": self.episodes,
                "llm_episodes_per_agent": self.llm_episodes,
                "base_seed": self.base_seed,
                "max_ticks_per_episode": self.max_ticks,
                "board": {"width": self.config.board.width, "height": self.config.board.height},
                "moves_per_second": self.config.game.moves_per_second,
                "max_foods": self.config.game.max_foods,
                "llm_configured": bool(self.llm_client),
            },
            "agents": {},
            "records": [],
        }

        llm_probe: tuple[bool, str] | None = None
        if self.llm_client is not None:
            llm_probe = self.llm_client.is_available()
        results["meta"]["llm_available"] = bool(llm_probe and llm_probe[0])
        if llm_probe and not llm_probe[0]:
            results["meta"]["llm_note"] = f"Ollama unavailable ({llm_probe[1]}); LLM agents ran on fallback strategy."

        done_total = 0
        needs_rl = any(a in ("rl", "llm_rl", "hybrid") for a in self.agents)
        if needs_rl and self.rl_agent is None:
            # One shared RL agent for all episodes: load the checkpoint once.
            from ..agents.factory import build_rl_agent

            self.rl_agent = build_rl_agent(self.config)
        for index, agent in enumerate(self.agents):
            if self.cancel_check():
                results["cancelled"] = True
                break
            n = self._episodes_for(agent)
            progress.agent = agent
            progress.agent_index = index
            progress.episodes_for_agent = n
            records: list[EpisodeRecord] = []
            for ep in range(n):
                if self.cancel_check():
                    results["cancelled"] = True
                    break
                seed = self.base_seed + ep
                record, decisions = self._run_episode(agent, seed)
                records.append(record)
                done_total += 1
                progress.episode = ep + 1
                progress.total_done = done_total
                progress.message = (
                    f"{AGENT_LABELS.get(agent, agent)}: episode {ep + 1}/{n} "
                    f"(score {record.score})"
                )
                if self.progress_cb:
                    self.progress_cb(progress)
            if records:
                summary = summarize(records, agent=agent)
                results["agents"][agent] = summary.to_dict()
                results["records"].extend(r.to_dict() for r in records)
        return results

    def _episodes_for(self, agent: str) -> int:
        # LLM inference is expensive: llm_astar / llm_rl get a configurable,
        # smaller sample size. The hybrid agent plays the full sample.
        if agent in ("llm_astar", "llm_rl") and self.llm_episodes > 0:
            return self.llm_episodes
        return self.episodes

    def _run_episode(self, agent: str, seed: int) -> tuple[EpisodeRecord, int]:
        engine = self._make_engine(seed)
        manager = build_manager(
            agent,
            self.config,
            rl_agent=self.rl_agent,
            llm_client=self.llm_client if agent in LLM_MODES else None,
        )
        manager.reset_episode()
        snake_id = "ai"
        steps = 0
        while not engine.finished and steps < self.max_ticks:
            action = manager.decide(engine.state, snake_id)
            engine.step({snake_id: action})
            steps += 1
        snake = engine.state.snakes[snake_id]
        survived = snake.alive
        mean_decision = (
            sum(entry.computation_ms for entry in manager.diagnostics) / len(manager.diagnostics)
            if manager.diagnostics
            else 0.0
        )
        llm_calls = manager.stats["llm_calls"]
        llm_fail = manager.stats["llm_failures"]
        llm_lat = (
            manager.stats["llm_latency_ms_total"] / llm_calls if llm_calls else 0.0
        )
        record = EpisodeRecord(
            agent=agent,
            seed=seed,
            score=snake.score,
            steps=steps,
            outcome=engine.outcome,
            death_reason=snake.death_reason,
            length=snake.length,
            survived=survived,
            foods=dict(snake.foods_eaten),
            safety_interventions=manager.stats["safety_interventions"],
            mean_decision_ms=round(mean_decision, 3),
            llm_calls=llm_calls,
            llm_failures=llm_fail,
            llm_latency_ms_mean=round(llm_lat, 1),
            llm_available=(llm_calls > 0 and llm_fail == 0),
        )
        return record, manager.stats["decisions"]

    # -- AI vs AI ----------------------------------------------------------------

    def run_match(self, agent_a: str, agent_b: str, seed: int) -> dict:
        """One competitive match between two AI snakes (symmetric starts)."""
        w, h = self.config.board.width, self.config.board.height
        engine = GameEngine(
            width=w,
            height=h,
            moves_per_second=self.config.game.moves_per_second,
            max_foods=self.config.game.max_foods,
            seed=seed,
            time_limit_seconds=self.config.game.time_limit_seconds,
            snake_specs=[
                SnakeSpec("a", AGENT_LABELS.get(agent_a, agent_a), False, (w // 4, h // 2), Action.RIGHT),
                SnakeSpec("b", AGENT_LABELS.get(agent_b, agent_b), False, (3 * w // 4, h // 2), Action.LEFT),
            ],
        )
        mgr_a = build_manager(agent_a, self.config, rl_agent=self.rl_agent,
                              llm_client=self.llm_client if agent_a in LLM_MODES else None)
        mgr_b = build_manager(agent_b, self.config, rl_agent=self.rl_agent,
                              llm_client=self.llm_client if agent_b in LLM_MODES else None)
        managers = {"a": mgr_a, "b": mgr_b}
        for mgr in managers.values():
            mgr.reset_episode()
        steps = 0
        while not engine.finished and steps < self.max_ticks:
            actions = {sid: mgr.decide(engine.state, sid) for sid, mgr in managers.items()}
            engine.step(actions)
            steps += 1
        sa, sb = engine.state.snakes["a"], engine.state.snakes["b"]
        winner = None
        if engine.outcome == "survivor_win" and engine.winner_ids:
            winner = engine.winner_ids[0]
        elif engine.outcome == "timeout" and len(engine.winner_ids) == 1:
            winner = engine.winner_ids[0]
        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "seed": seed,
            "steps": steps,
            "outcome": engine.outcome,
            "winner": winner,
            "winner_agent": (agent_a if winner == "a" else agent_b) if winner else None,
            "scores": {"a": sa.score, "b": sb.score},
            "deaths": {"a": sa.death_reason, "b": sb.death_reason},
            "lengths": {"a": sa.length, "b": sb.length},
            "safety_interventions": {sid: mgr.stats["safety_interventions"] for sid, mgr in managers.items()},
        }

    def run_tournament(self, agents: list[str] | None = None, rounds: int = 2) -> dict:
        """Round-robin: every ordered pair meets ``rounds`` times (new seeds)."""
        roster = agents or [a for a in self.agents if a not in ("hybrid",)] or list(self.agents)
        matches: list[dict] = []
        wins: dict[str, int] = {a: 0 for a in roster}
        draws = 0
        seed = self.base_seed + 50_000
        for i, a in enumerate(roster):
            for b in roster[i + 1 :]:
                for r in range(rounds):
                    seed += 1
                    match = self.run_match(a, b, seed)
                    matches.append(match)
                    if match["winner_agent"] is None:
                        draws += 1
                    else:
                        wins[match["winner_agent"]] = wins.get(match["winner_agent"], 0) + 1
                    if self.progress_cb:
                        progress = BenchmarkProgress(
                            agent=f"{a} vs {b}",
                            message=f"match {len(matches)}: {match['outcome']}",
                            total_done=len(matches),
                            total_planned=len(roster) * (len(roster) - 1) // 2 * rounds,
                        )
                        self.progress_cb(progress)
        total_matches = len(matches)
        return {
            "rounds": rounds,
            "matches": total_matches,
            "draws": draws,
            "wins": wins,
            "win_rate": {a: round(w / total_matches, 3) if total_matches else 0.0 for a, w in wins.items()},
            "match_records": matches,
        }
