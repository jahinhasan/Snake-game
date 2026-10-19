"""Agent benchmark dashboard: automated solo comparisons + AI-vs-AI tournaments.

Runs ``BenchmarkRunner`` in a background thread (the UI stays responsive),
shows live progress, and renders the *same* summary rows/JSON/CSV/PNG
artifacts the CLI produces -- one source of truth for every number shown.
"""

from __future__ import annotations

import threading
import time

import pygame

from ..evaluation.benchmark import AGENT_LABELS, BenchmarkProgress, BenchmarkRunner
from ..evaluation.experiments import (
    benchmark_rows,
    list_runs,
    load_run,
    make_run_dir,
    plot_benchmark_charts,
    save_benchmark_results,
)
from .app import App, Screen
from .components import (
    BarChart,
    Button,
    DataTable,
    Dropdown,
    Label,
    ProgressBar,
    Slider,
    Toggle,
    draw_widgets,
    send_event,
)
from .theme import COLORS, rgb, text

ALL_AGENTS = ["rule_based", "astar", "bfs", "rl", "llm_astar", "llm_rl", "hybrid"]
TABLE_KEYS = ("agent", "episodes", "mean_score", "median_score", "std",
              "survival", "mean_steps", "traps/1k", "decision_ms")
BAR_COLORS = [COLORS.accent, COLORS.green, COLORS.gold, COLORS.purple,
              COLORS.red, COLORS.blue, COLORS.orange]


class BenchmarkDashboard(Screen):
    def __init__(self, app: App) -> None:
        super().__init__(app)
        self.episodes = float(app.config.benchmark.episodes_per_agent)
        self.llm_episodes = float(app.config.benchmark.llm_episodes_per_agent)
        self.selected = list(app.config.benchmark.agents)
        self.results: dict | None = None
        self.tournament: dict | None = None
        self.progress = BenchmarkProgress()
        self.running = False
        self.cancel_requested = False
        self.worker: threading.Thread | None = None
        self.saved_note = ""
        self.previous_runs: list = []

    def on_enter(self) -> None:
        self.refresh_previous_runs()
        self.layout()

    def refresh_previous_runs(self) -> None:
        self.previous_runs = list_runs(limit=12)

    # -- run control -----------------------------------------------------------

    def _toggle_agent(self, agent: str, value: bool) -> None:
        if value and agent not in self.selected:
            self.selected.append(agent)
        elif not value and agent in self.selected:
            self.selected.remove(agent)

    def _run_benchmark(self) -> None:
        self._launch("solo")

    def _run_tournament(self) -> None:
        if len(self.selected) < 2:
            self.saved_note = "select at least 2 agents for a tournament"
            return
        self._launch("tournament")

    def _launch(self, mode: str) -> None:
        if self.running or not self.selected:
            return
        self.running = True
        self.cancel_requested = False
        self.saved_note = ""
        self.results = None
        self.tournament = None
        app = self.app
        app.config.benchmark.episodes_per_agent = int(self.episodes)
        app.config.benchmark.llm_episodes_per_agent = int(self.llm_episodes)
        app.config.benchmark.agents = list(self.selected)
        client = app.llm_client() if app.config.llm.enabled else None
        rl_agent = app.rl_agent() if any(a in ("rl", "llm_rl", "hybrid") for a in self.selected) else None
        agents = list(self.selected)

        def work() -> None:
            runner = BenchmarkRunner(
                app.config,
                agents=agents,
                episodes=int(self.episodes),
                llm_episodes=int(self.llm_episodes),
                llm_client=client,
                rl_agent=rl_agent,
                progress_cb=self._on_progress,
                cancel_check=lambda: self.cancel_requested,
            )
            if mode == "tournament":
                tournament = runner.run_tournament(agents=agents, rounds=2)
                self.tournament = tournament
                results = {
                    "run_id": None,
                    "timestamp": time.time(),
                    "meta": {
                        "kind": "tournament",
                        "agents": agents,
                        "llm_configured": bool(client),
                    },
                    "agents": {},
                    "records": [],
                    "tournament": tournament,
                }
            else:
                results = runner.run()
            self.results = results
            if not self.cancel_requested:
                run_dir = make_run_dir("ui")
                save_benchmark_results(run_dir, results)
                charts = plot_benchmark_charts(run_dir, results)
                self.saved_note = f"saved: data/benchmarks/{run_dir.name} ({len(charts)} chart files)"
            self.running = False

        self.worker = threading.Thread(target=work, daemon=True, name=f"benchmark-{mode}")
        self.worker.start()

    def _cancel(self) -> None:
        if self.running:
            self.cancel_requested = True

    def _on_progress(self, progress: BenchmarkProgress) -> None:
        self.progress = progress

    def _load_selected_run(self, value) -> None:
        for path, _meta in self.previous_runs:
            if path.name == value:
                data = load_run(path)
                if data:
                    self.results = data
                    self.tournament = data.get("tournament")
                    self.saved_note = f"loaded previous run: {path.name}"
                break

    def _back(self) -> None:
        self.cancel_requested = True
        self.app.pop_screen()

    # -- layout ----------------------------------------------------------------

    def layout(self) -> None:
        w, _h = self.app.screen.get_size()
        self.widgets = []
        x0, y = 28, 84
        self.widgets.append(Slider(
            "Episodes per agent", self.episodes, 1, 100,
            lambda v: setattr(self, "episodes", v), (x0, y), width=280))
        y += 58
        self.widgets.append(Slider(
            "Episodes for LLM agents", self.llm_episodes, 0, 50,
            lambda v: setattr(self, "llm_episodes", v), (x0, y), width=280))
        y += 62
        self.widgets.append(Label("Agents (identical seeds for every agent)", (x0, y),
                                  font=None, color=COLORS.text_dim))
        y += 28
        per_row = 2
        for i, agent in enumerate(ALL_AGENTS):
            col, row = i % per_row, i // per_row
            self.widgets.append(Toggle(
                AGENT_LABELS[agent], agent in self.selected,
                lambda v, a=agent: self._toggle_agent(a, v),
                (x0 + col * 190, y + row * 30), width=180))
        y += ((len(ALL_AGENTS) + 1) // per_row) * 30 + 16
        self.widgets.append(Label("Previous runs", (x0, y), font=None, color=COLORS.text_dim))
        y += 22
        self.widgets.append(Dropdown(
            [(p.name, p.name) for p, _m in self.previous_runs] or [("", "no previous runs")],
            self.previous_runs[0][0].name if self.previous_runs else "",
            self._load_selected_run, (x0, y), width=280))
        y += 46
        self.widgets.append(Button("Run Benchmark", self._run_benchmark,
                                   pygame.Rect(x0, y, 200, 46), style="primary"))
        self.widgets.append(Button("Run Tournament", self._run_tournament,
                                   pygame.Rect(x0 + 214, y, 180, 46), style="primary"))
        self.widgets.append(Button("Cancel", self._cancel,
                                   pygame.Rect(x0 + 408, y, 110, 46), style="danger"))
        self.widgets.append(Button("Back", self._back,
                                   pygame.Rect(x0 + 532, y, 100, 46), style="secondary"))

        # Right column: progress, results table, bar chart.
        self.rx = min(x0 + 660, max(560, w - 640))
        self.table_w = min(620, w - self.rx - 30)
        self.progress_bar = ProgressBar((self.rx, 84), width=self.table_w)
        self.widgets.append(self.progress_bar)
        self.table = DataTable(
            list(TABLE_KEYS), (self.rx, 150), (self.table_w, 190),
            col_ratios=[0.16, 0.07, 0.11, 0.12, 0.10, 0.10, 0.11, 0.12, 0.11],
        )
        self.widgets.append(self.table)
        self.bar_chart = BarChart("Mean score by agent",
                                  (self.rx, 356), (self.table_w, 250))
        self.widgets.append(self.bar_chart)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._back()
            return
        send_event(self.widgets, event)

    # -- drawing ---------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        surface.fill(rgb(COLORS.bg))
        pygame.draw.rect(surface, rgb(COLORS.panel), (12, 12, w - 24, h - 24), border_radius=14)
        pygame.draw.rect(surface, rgb(COLORS.border), (12, 12, w - 24, h - 24), width=1, border_radius=14)
        text(surface, "AGENT BENCHMARK", (32, 26), font=self.app.fonts.h2, color=COLORS.accent)
        text(surface, "Solo episodes: identical seeds per agent. Tournament: round-robin AI-vs-AI matches.",
             (32, 62), font=self.app.fonts.small, color=COLORS.text_dim)

        self._draw_progress(surface)
        self._draw_results(surface)
        draw_widgets(surface, self.widgets, self.app.fonts)

    def _draw_progress(self, surface: pygame.Surface) -> None:
        if self.running:
            planned = max(1, self.progress.total_planned)
            self.progress_bar.set(self.progress.total_done / planned)
            note = f"{self.progress.message or 'starting...'}  ({self.progress.total_done}/{planned})"
        elif self.results is not None:
            self.progress_bar.set(1.0)
            cancelled = " (cancelled)" if self.results.get("cancelled") else ""
            kind = self.results.get("meta", {}).get("kind", "solo")
            note = f"done{cancelled} - {kind} run"
        else:
            self.progress_bar.set(0.0)
            note = "idle - configure and press Run"
        self.progress_bar.draw(surface, self.app.fonts)
        text(surface, note, (self.rx, 106), font=self.app.fonts.small, color=COLORS.text_dim)
        if self.saved_note:
            text(surface, self.saved_note, (self.rx, 126), font=self.app.fonts.small, color=COLORS.success)

    def _draw_results(self, surface: pygame.Surface) -> None:
        if self.tournament:
            self.table.headers = ["agent", "wins", "win rate", "draws"]
            total = max(1, self.tournament.get("matches", 1))
            self.table.set_rows([
                [AGENT_LABELS.get(a, a), str(wins), f"{wins / total * 100:.0f}%"]
                for a, wins in sorted(self.tournament.get("wins", {}).items(),
                                      key=lambda kv: kv[1], reverse=True)
            ] + [["(draws)", str(self.tournament.get("draws", 0)),
                  f"{self.tournament.get('draws', 0) / total * 100:.0f}%"]])
            self.bar_chart.set_bars([
                (AGENT_LABELS.get(a, a), float(wins), BAR_COLORS[i % len(BAR_COLORS)])
                for i, (a, wins) in enumerate(sorted(self.tournament.get("wins", {}).items(),
                                                     key=lambda kv: kv[1], reverse=True))
            ])
            self.bar_chart.title = "Tournament wins by agent"
            return

        self.bar_chart.title = "Mean score by agent"
        self.table.headers = list(TABLE_KEYS)
        if self.results is None:
            self.table.set_rows([])
            self.bar_chart.set_bars([])
            return
        rows = benchmark_rows(self.results)
        self.table.set_rows([[str(r.get(key, "-")) for key in TABLE_KEYS] for r in rows])
        self.bar_chart.set_bars([
            (r["agent"], float(r["mean_score"]), BAR_COLORS[i % len(BAR_COLORS)])
            for i, r in enumerate(rows)
        ])
        meta = self.results.get("meta", {})
        if meta.get("llm_available") is False and any(a in ("llm_astar", "llm_rl") for a in meta.get("agents", [])):
            text(surface, "LLM was unavailable: LLM-agent rows used the deterministic fallback strategy.",
                 (self.rx, self.table.rect.bottom + 4), font=self.app.fonts.small, color=COLORS.gold)
