"""Post-game results screen: measured summary + optional LLM analysis."""

from __future__ import annotations

import threading
import time
from typing import Callable

import pygame

from ..llm.analysis import build_analysis_messages, parse_analysis, summarize_game
from ..persistence.storage import run_id, save_json_atomic
from ..utils import paths
from .app import App, Screen
from .components import Button, Label, draw_widgets
from .theme import COLORS, rgb, text, wrap_text


class ResultsScreen(Screen):
    def __init__(self, app: App, record: dict | None, rematch: Callable[[], None]) -> None:
        super().__init__(app)
        self.record = record or {}
        self.rematch_cb = rematch
        self.analysis_lines: list[str] = []
        self.analysis_status = "idle"
        self.analysis_error: str | None = None
        self._analysis_thread: threading.Thread | None = None
        self.export_note_until = 0.0

    def on_enter(self) -> None:
        self.layout()
        self._maybe_start_analysis()

    # -- LLM analysis ----------------------------------------------------------

    def _maybe_start_analysis(self) -> None:
        cfg = self.app.config
        if not cfg.llm.enabled:
            self.analysis_status = "disabled"
            return
        client = self.app.llm_client()
        if client is None:
            self.analysis_status = "disabled"
            return

        def work() -> None:
            self.analysis_status = "waiting"
            ok, _detail = client.is_available()
            if not ok:
                self.analysis_status = "unavailable"
                return
            self.analysis_status = "generating"
            summary = summarize_game(self.record)
            system, user = build_analysis_messages(summary)
            try:
                raw = client.chat_raw(system, user)
                text_out = parse_analysis(raw)
                self.analysis_lines = wrap_text(text_out, self.app.fonts.body, 760)
                self.analysis_status = "done"
            except Exception as exc:  # noqa: BLE001 - analysis is optional
                self.analysis_error = str(exc)[:120]
                self.analysis_status = "failed"

        self._analysis_thread = threading.Thread(target=work, daemon=True, name="llm-analysis")
        self._analysis_thread.start()

    # -- actions ---------------------------------------------------------------

    def _export(self) -> None:
        if not self.record:
            return
        out_dir = paths.history_dir() / "exports"
        path = save_json_atomic(out_dir / f"game-{run_id('g')}.json", self.record)
        self.export_note_until = time.time() + 3.0
        self._export_path = str(path)
        self.layout()

    def _menu(self) -> None:
        self.go_menu()

    def layout(self) -> None:
        w, h = self.app.screen.get_size()
        panel = pygame.Rect(w // 2 - 430, h // 2 - 300, 860, 600)
        self.panel = panel
        y = panel.bottom - 62
        self.widgets = [
            Button("Rematch", self._rematch, pygame.Rect(panel.x + 24, y, 180, 46), style="primary"),
            Button("Export JSON", self._export, pygame.Rect(panel.x + 224, y, 180, 46), style="secondary"),
            Button("Main Menu", self._menu, pygame.Rect(panel.x + 424, y, 180, 46), style="secondary"),
        ]

    def _rematch(self) -> None:
        self.app.pop_screen()
        self.rematch_cb()

    # -- drawing ---------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        surface.fill(rgb(COLORS.bg))
        panel = self.panel
        pygame.draw.rect(surface, rgb(COLORS.panel), panel, border_radius=14)
        pygame.draw.rect(surface, rgb(COLORS.border), panel, width=1, border_radius=14)
        x0 = panel.x + 26
        y = panel.y + 20

        summary = summarize_game(self.record)
        winner = summary.get("winner")
        outcome = summary.get("outcome", "?")
        if outcome == "draw":
            banner, color = "DRAW", COLORS.gold
        elif winner:
            banner, color = f"{winner[0]} WINS", COLORS.success
        else:
            banner, color = "GAME OVER", COLORS.text
        text(surface, banner, (x0, y), font=self.app.fonts.h1, color=color)
        y += 44
        text(surface, f"mode: {self.record.get('mode', '?')}   duration: {summary.get('duration_seconds', 0):.1f}s "
                     f"({summary.get('duration_ticks', 0)} ticks)   outcome: {outcome}",
             (x0, y), font=self.app.fonts.small, color=COLORS.text_dim)
        y += 30

        # Per-snake stats table.
        rows = []
        for name, s in summary.get("snakes", {}).items():
            foods = ", ".join(f"{k}:{v}" for k, v in sorted(s.get("food_by_type", {}).items())) or "-"
            status = "survived" if s.get("survived") else f"died ({s.get('death_reason')})"
            tag = " [human]" if s.get("is_human") else ""
            rows.append(f"{name}{tag}  -  score {s.get('score', 0)}  -  length {s.get('final_length', 0)}  -  "
                        f"food [{foods}]  -  {status}")
        for row in rows:
            text(surface, row, (x0, y), font=self.app.fonts.mono, color=COLORS.text)
            y += 22
        y += 8

        ai = self.record.get("ai_summary", {}) or {}
        if ai:
            text(surface,
                 f"AI: {ai.get('decisions', 0)} decisions, {ai.get('safety_interventions', 0)} safety overrides, "
                 f"strategy: {self.record.get('strategy_summary') or 'default'}",
                 (x0, y), font=self.app.fonts.small, color=COLORS.text_dim)
            y += 20
            if ai.get("gating_activations"):
                text(surface, f"RL risk-gating activations: {ai.get('gating_activations')}",
                     (x0, y), font=self.app.fonts.small, color=COLORS.text_dim)
                y += 20
        llm_meta = self.record.get("llm", {}) or {}
        y += 6
        pygame.draw.line(surface, rgb(COLORS.border), (x0, y), (panel.right - 26, y))
        y += 12

        # LLM analysis box (clearly marked as model-generated).
        if self.analysis_status == "done":
            text(surface, "AI-generated analysis (Qwen3 via Ollama - interpretation, not measurement):",
                 (x0, y), font=self.app.fonts.small, color=COLORS.purple)
            y += 22
            for line in self.analysis_lines:
                if y > panel.bottom - 110:
                    text(surface, "...", (x0, y), font=self.app.fonts.body, color=COLORS.text_faint)
                    break
                text(surface, line, (x0, y), font=self.app.fonts.body, color=COLORS.text)
                y += 21
        else:
            status_text = {
                "idle": "",
                "disabled": "LLM analysis disabled - offline summary shown above.",
                "waiting": "Connecting to Ollama...",
                "unavailable": "Ollama unavailable - offline summary shown above.",
                "generating": "Generating AI analysis...",
                "failed": f"LLM analysis failed ({self.analysis_error}) - offline summary shown above.",
            }.get(self.analysis_status, "")
            if status_text:
                text(surface, status_text, (x0, y), font=self.app.fonts.small, color=COLORS.text_faint)
                y += 22

        draw_widgets(surface, self.widgets, self.app.fonts)
        if time.time() < self.export_note_until:
            path = getattr(self, "_export_path", "")
            Label(f"saved: {path}", (panel.x + 620, panel.bottom - 50),
                  font=self.app.fonts.small, color=COLORS.success).draw(surface, self.app.fonts)
