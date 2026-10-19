"""Statistics screen: aggregates and recent games from the persisted history.

Every number comes from ``data/history/games.jsonl`` -- records written when
games actually finish. With no history the screen says so plainly instead of
showing placeholders.
"""

from __future__ import annotations

import datetime

import pygame

from ..evaluation.benchmark import AGENT_LABELS
from ..persistence.history import aggregate, clear_history, load_recent
from .app import App, Screen
from .components import Button, DataTable, Label, draw_widgets, send_event
from .theme import COLORS, rgb, text


class StatisticsScreen(Screen):
    def __init__(self, app: App) -> None:
        super().__init__(app)
        self.note = ""
        self.stats: dict = {}
        self.recent: list[dict] = []

    def on_enter(self) -> None:
        self.refresh()
        self.layout()

    def refresh(self) -> None:
        self.stats = aggregate()
        self.recent = load_recent(30)

    def _refresh_clicked(self) -> None:
        self.refresh()
        self.note = "reloaded from history"
        self.layout()

    def _clear_clicked(self) -> None:
        count = clear_history()
        self.note = f"deleted {count} history records"
        self.refresh()
        self.layout()

    def _back(self) -> None:
        self.app.pop_screen()

    def layout(self) -> None:
        w, h = self.app.screen.get_size()
        self.widgets = []
        bx, fy = 32, h - 66
        self.widgets.append(Button("Reload", self._refresh_clicked,
                                   pygame.Rect(bx, fy, 140, 44), style="secondary"))
        self.widgets.append(Button("Clear History", self._clear_clicked,
                                   pygame.Rect(bx + 154, fy, 170, 44), style="danger"))
        self.widgets.append(Button("Back", self._back,
                                   pygame.Rect(bx + 338, fy, 120, 44), style="secondary"))
        if self.note:
            self.widgets.append(Label(self.note, (bx + 480, fy + 12), font=None, color=COLORS.gold))

        self.table = DataTable(
            ["when", "mode", "opponent", "ticks", "outcome", "human", "AI score"],
            (32, 300), (min(720, w - 64), min(280, h - 390)),
            col_ratios=[0.17, 0.15, 0.16, 0.09, 0.15, 0.13, 0.15],
        )
        self.widgets.append(self.table)
        self.table.set_rows([self._row_for(game) for game in self.recent])

    @staticmethod
    def _row_for(game: dict) -> list[str]:
        when = datetime.datetime.fromtimestamp(game.get("timestamp", 0)).strftime("%m-%d %H:%M")
        humans = {sid: s for sid, s in game.get("snakes", {}).items() if s.get("is_human")}
        agents = {sid: s for sid, s in game.get("snakes", {}).items() if not s.get("is_human")}
        human_cell = "-"
        if humans:
            h = next(iter(humans.values()))
            human_cell = f"{h.get('score', 0)} ({'alive' if h.get('survived') else h.get('death_reason', '?')})"
        agent_cell = "-"
        if agents:
            a = next(iter(agents.values()))
            agent_cell = f"{a.get('score', 0)} ({'alive' if a.get('survived') else a.get('death_reason', '?')})"
        opponent = AGENT_LABELS.get(game.get("agent", ""), game.get("agent", "-"))
        return [when, str(game.get("mode", "?")), opponent, str(game.get("ticks", 0)),
                str(game.get("outcome", "?")), human_cell, agent_cell]

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._back()
            return
        send_event(self.widgets, event)

    def draw(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        surface.fill(rgb(COLORS.bg))
        pygame.draw.rect(surface, rgb(COLORS.panel), (12, 12, w - 24, h - 24), border_radius=14)
        pygame.draw.rect(surface, rgb(COLORS.border), (12, 12, w - 24, h - 24), width=1, border_radius=14)
        text(surface, "STATISTICS", (32, 26), font=self.app.fonts.h2, color=COLORS.accent)
        text(surface, "recorded from completed games in data/history/games.jsonl",
             (260, 36), font=self.app.fonts.small, color=COLORS.text_dim)

        if self.stats.get("games", 0) == 0:
            text(surface, "No games recorded yet.", (32, 100), font=self.app.fonts.h3, color=COLORS.text_dim)
            text(surface, "Play a game or run a benchmark with history recording enabled - statistics appear here.",
                 (32, 136), font=self.app.fonts.body, color=COLORS.text_faint)
        else:
            y = 92
            top = self.stats.get("top_game")
            lines = [
                f"games recorded: {self.stats.get('games', 0)}",
                f"total AI score across history: {self.stats.get('total_ai_score', 0)}",
                f"best AI game: {top['score'] if top else '-'}"
                + (f" by {AGENT_LABELS.get(top['agent'], top['agent'])}" if top else ""),
            ]
            for line in lines:
                text(surface, line, (32, y), font=self.app.fonts.h3, color=COLORS.text)
                y += 34

            y += 8
            text(surface, "Per-agent history", (32, y), font=self.app.fonts.h3, color=COLORS.accent)
            y += 32
            by_agent = self.stats.get("by_agent", {})
            for agent, s in by_agent.items():
                label = AGENT_LABELS.get(agent, agent)
                text(surface, f"{label:<14} games {s['games']:>4}   mean {s['mean_score']:>7.1f}   "
                              f"best {s['max_score']:>4}",
                     (32, y), font=self.app.fonts.mono, color=COLORS.text_dim)
                y += 24
            if not by_agent:
                text(surface, "(no AI-agent games recorded)", (32, y), font=self.app.fonts.body,
                     color=COLORS.text_faint)
                y += 24

            y += 14
            text(surface, f"Recent games ({len(self.recent)} most recent)", (32, y),
                 font=self.app.fonts.h3, color=COLORS.accent)

        draw_widgets(surface, self.widgets, self.app.fonts)
