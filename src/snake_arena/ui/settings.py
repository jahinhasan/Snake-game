"""Settings screen: edit the live AppConfig and persist it atomically.

Settings mutate the in-memory dataclasses immediately (games launched
afterwards pick them up); "Save" validates and writes ``config/default.json``.
LLM-dependent cached services are invalidated when LLM settings change.
"""

from __future__ import annotations

import pygame

from ..config import AppConfig, load_app_config
from ..evaluation.benchmark import AGENT_LABELS
from .app import App, Screen
from .components import (
    Button,
    Dropdown,
    Label,
    Slider,
    TextInput,
    Toggle,
    draw_widgets,
    send_event,
)
from .theme import COLORS, rgb, text

AGENT_OPTIONS = [(t, AGENT_LABELS.get(t, t)) for t in
                 ("rule_based", "astar", "bfs", "rl", "llm_astar", "llm_rl", "hybrid")]
BOARD_PRESETS = [((24, 18), "Standard 24x18"), ((16, 12), "Small 16x12"),
                 ((32, 24), "Large 32x24"), ((40, 30), "Huge 40x30")]
RESOLUTIONS = [((1280, 800), "1280 x 800"), ((1440, 900), "1440 x 900"),
               ((1920, 1080), "1920 x 1080"), ((1024, 680), "1024 x 680")]
SECTIONS = ["Game", "Display & Audio", "Agents", "LLM Planner", "RL & Safety"]


class SettingsScreen(Screen):
    def __init__(self, app: App) -> None:
        super().__init__(app)
        self.section = "Game"
        self.note = ""
        self.note_color = COLORS.text_dim

    def on_enter(self) -> None:
        self.layout()

    # -- helpers ---------------------------------------------------------------

    def _invalidate_llm_cache(self) -> None:
        # The App caches one client/worker; drop it so new settings take effect.
        self.app._llm_client = None
        if self.app._llm_worker is not None:
            self.app._llm_worker.stop()
            self.app._llm_worker = None

    def _flash(self, message: str, color: str = COLORS.text_dim) -> None:
        self.note = message
        self.note_color = color

    def _volume_changed(self, value: float) -> None:
        self.app.config.audio.volume = value
        try:
            self.app.audio.apply_volume(self.app.config.audio)
        except Exception:  # noqa: BLE001 - audio is optional
            pass

    # -- save / restore ----------------------------------------------------------

    def _save(self) -> None:
        cfg = self.app.config
        cfg.validate()
        path = cfg.save()
        self._flash(f"saved: {path}", COLORS.success)

    def _restore(self) -> None:
        self.app.config = load_app_config()
        self._invalidate_llm_cache()
        self._flash("restored saved settings from disk", COLORS.gold)
        self.layout()

    def _back(self) -> None:
        self.app.pop_screen()

    # -- layout ----------------------------------------------------------------

    def layout(self) -> None:
        w, h = self.app.screen.get_size()
        self.widgets = []
        x0, y0 = 40, 120
        cfg = self.app.config

        section_dd = Dropdown(
            [(s, s) for s in SECTIONS], self.section,
            lambda v: self._switch_section(v), (x0, 76), width=260)
        section_dd.font = self.app.fonts.h3
        self.widgets.append(section_dd)

        y = y0
        section = self.section
        if section == "Game":
            self.widgets.append(Slider(
                "Game speed", float(cfg.game.moves_per_second), 2, 30,
                lambda v: setattr(cfg.game, "moves_per_second", int(v)),
                (x0, y), width=320, fmt=lambda v: f"{int(v)} moves/s"))
            y += 58
            self.widgets.append(Slider(
                "Max foods on board", float(cfg.game.max_foods), 1, 12,
                lambda v: setattr(cfg.game, "max_foods", int(v)),
                (x0, y), width=320, fmt=lambda v: f"{int(v)}"))
            y += 58
            self.widgets.append(Slider(
                "Round time limit", float(cfg.game.time_limit_seconds), 30, 900,
                lambda v: setattr(cfg.game, "time_limit_seconds", int(v)),
                (x0, y), width=320, fmt=lambda v: f"{int(v)} s"))
            y += 58
            self.widgets.append(Label("Board size (next game)", (x0, y), font=None, color=COLORS.text_dim))
            self.widgets.append(Dropdown(
                BOARD_PRESETS, (cfg.board.width, cfg.board.height),
                lambda v: (setattr(cfg.board, "width", v[0]), setattr(cfg.board, "height", v[1])),
                (x0 + 200, y - 4), width=200))
            y += 48
            self.widgets.append(Label("Food types: normal +10, bonus +25, golden +50, special +100, trap -20",
                                      (x0, y), font=None, color=COLORS.text_faint))

        elif section == "Display & Audio":
            self.widgets.append(Dropdown(
                RESOLUTIONS, (cfg.window.width, cfg.window.height),
                lambda v: (setattr(cfg.window, "width", v[0]), setattr(cfg.window, "height", v[1])),
                (x0, y), width=200))
            self.widgets.append(Label("Window size (applies on next launch)", (x0 + 220, y + 8),
                                      font=None, color=COLORS.text_dim))
            y += 46
            self.widgets.append(Toggle("Fullscreen (next launch)", cfg.window.fullscreen,
                                       lambda v: setattr(cfg.window, "fullscreen", v), (x0, y)))
            self.widgets.append(Toggle("Show FPS counter", cfg.window.fps_counter,
                                       lambda v: setattr(cfg.window, "fps_counter", v), (x0 + 320, y)))
            y += 40
            self.widgets.append(Toggle("Sound effects", cfg.audio.enabled,
                                       lambda v: setattr(cfg.audio, "enabled", v), (x0, y)))
            y += 40
            self.widgets.append(Slider(
                "Volume", cfg.audio.volume, 0.0, 1.0, self._volume_changed,
                (x0, y), width=320, step=0.05, fmt=lambda v: f"{int(v * 100)}%"))
            y += 62
            self.widgets.append(Label("AI visualization overlays", (x0, y), font=None, color=COLORS.text_dim))
            y += 28
            overlay_rows = [
                ("show_grid", "Grid"), ("show_path", "Planned path"),
                ("show_danger", "Danger map"), ("show_target", "Target ring"),
                ("show_rejected", "Rejected moves"), ("show_qvalues", "Q-values"),
            ]
            for i, (key, label) in enumerate(overlay_rows):
                col, row = i % 3, i // 3
                self.widgets.append(Toggle(
                    label, getattr(cfg.visual, key),
                    lambda v, k=key: setattr(cfg.visual, k, v),
                    (x0 + col * 220, y + row * 32), width=210))

        elif section == "Agents":
            self.widgets.append(Label("Default agent", (x0, y), font=None, color=COLORS.text_dim))
            self.widgets.append(Dropdown(
                AGENT_OPTIONS, cfg.agent.type,
                lambda v: setattr(cfg.agent, "type", v), (x0 + 160, y - 4), width=190))
            self.widgets.append(Label("Default opponent", (x0 + 380, y), font=None, color=COLORS.text_dim))
            self.widgets.append(Dropdown(
                AGENT_OPTIONS, cfg.agent.opponent_type,
                lambda v: setattr(cfg.agent, "opponent_type", v), (x0 + 530, y - 4), width=190))
            y += 48
            self.widgets.append(TextInput(
                cfg.agent.rl_checkpoint,
                lambda v: setattr(cfg.agent, "rl_checkpoint", v),
                (x0, y), width=280, placeholder="models/dqn_default.pt"))
            self.widgets.append(Label("RL checkpoint path (relative to project)", (x0 + 296, y + 8),
                                      font=None, color=COLORS.text_dim))
            y += 60
            self.widgets.append(Slider(
                "Hybrid weight: RL", cfg.decision.hybrid.w_rl, 0.0, 1.0,
                lambda v: setattr(cfg.decision.hybrid, "w_rl", v),
                (x0, y), width=320, step=0.05, fmt=lambda v: f"{v:.2f}"))
            y += 56
            self.widgets.append(Slider(
                "Hybrid weight: path (A*)", cfg.decision.hybrid.w_path, 0.0, 1.0,
                lambda v: setattr(cfg.decision.hybrid, "w_path", v),
                (x0, y), width=320, step=0.05, fmt=lambda v: f"{v:.2f}"))
            y += 56
            self.widgets.append(Slider(
                "Hybrid weight: free space", cfg.decision.hybrid.w_space, 0.0, 1.0,
                lambda v: setattr(cfg.decision.hybrid, "w_space", v),
                (x0, y), width=320, step=0.05, fmt=lambda v: f"{v:.2f}"))
            y += 52
            self.widgets.append(Label("Weights are normalized to sum to 1 on save.", (x0, y),
                                      font=None, color=COLORS.text_faint))

        elif section == "LLM Planner":
            self.widgets.append(Toggle("Enable LLM strategic planner (optional; game works without it)",
                                       cfg.llm.enabled,
                                       lambda v: (setattr(cfg.llm, "enabled", v), self._invalidate_llm_cache()),
                                       (x0, y), width=560))
            y += 42
            self.widgets.append(TextInput(
                cfg.llm.host, lambda v: (setattr(cfg.llm, "host", v.strip()), self._invalidate_llm_cache()),
                (x0, y), width=280, placeholder="http://127.0.0.1:11434"))
            self.widgets.append(Label("Ollama host", (x0 + 296, y + 8), font=None, color=COLORS.text_dim))
            y += 44
            self.widgets.append(TextInput(
                cfg.llm.model, lambda v: (setattr(cfg.llm, "model", v.strip()), self._invalidate_llm_cache()),
                (x0, y), width=280, placeholder="qwen3:4b"))
            self.widgets.append(Label("Model tag (ollama pull <tag>)", (x0 + 296, y + 8),
                                      font=None, color=COLORS.text_dim))
            y += 52
            self.widgets.append(Slider(
                "Request timeout", cfg.llm.timeout_seconds, 1.0, 30.0,
                lambda v: setattr(cfg.llm, "timeout_seconds", v),
                (x0, y), width=320, step=0.5, fmt=lambda v: f"{v:.1f} s"))
            y += 56
            self.widgets.append(Slider(
                "Min seconds between strategy calls", cfg.llm.min_interval_seconds, 0.5, 30.0,
                lambda v: (setattr(cfg.llm, "min_interval_seconds", v), self._invalidate_llm_cache()),
                (x0, y), width=320, step=0.5, fmt=lambda v: f"{v:.1f} s"))
            y += 56
            self.widgets.append(Slider(
                "Sampling temperature", cfg.llm.temperature, 0.0, 1.5,
                lambda v: setattr(cfg.llm, "temperature", v),
                (x0, y), width=320, step=0.05, fmt=lambda v: f"{v:.2f}"))
            y += 56
            self.widgets.append(Slider(
                "Max response tokens", float(cfg.llm.max_tokens), 40, 1024,
                lambda v: setattr(cfg.llm, "max_tokens", int(v)),
                (x0, y), width=320, fmt=lambda v: f"{int(v)}"))

        elif section == "RL & Safety":
            self.widgets.append(Slider(
                "Discount factor (gamma)", cfg.rl.gamma, 0.10, 0.999,
                lambda v: setattr(cfg.rl, "gamma", v),
                (x0, y), width=320, step=0.01, fmt=lambda v: f"{v:.2f}"))
            y += 56
            self.widgets.append(Slider(
                "Learning rate", cfg.rl.learning_rate, 0.0001, 0.01,
                lambda v: setattr(cfg.rl, "learning_rate", v),
                (x0, y), width=320, step=0.0005, fmt=lambda v: f"{v:.4f}"))
            y += 56
            self.widgets.append(Slider(
                "Epsilon decay steps", float(cfg.rl.eps_decay_steps), 1000, 100000,
                lambda v: setattr(cfg.rl, "eps_decay_steps", int(v)),
                (x0, y), width=320, fmt=lambda v: f"{int(v)}"))
            y += 62
            self.widgets.append(Label("Safety validator", (x0, y), font=None, color=COLORS.text_dim))
            y += 28
            self.widgets.append(Toggle("Dead-end check (flood-fill escape area)",
                                       cfg.safety.dead_end_check,
                                       lambda v: setattr(cfg.safety, "dead_end_check", v), (x0, y)))
            y += 40
            self.widgets.append(Slider(
                "Min escape area x snake length", cfg.safety.min_escape_area_factor, 0.0, 3.0,
                lambda v: setattr(cfg.safety, "min_escape_area_factor", v),
                (x0, y), width=320, step=0.1, fmt=lambda v: f"{v:.1f}"))
            y += 56
            self.widgets.append(Slider(
                "Flood-fill cell limit", float(cfg.safety.flood_fill_limit), 100, 5000,
                lambda v: setattr(cfg.safety, "flood_fill_limit", int(v)),
                (x0, y), width=320, fmt=lambda v: f"{int(v)}"))

        # Footer (always visible).
        fy = h - 70
        self.widgets.append(Button("Save Settings", self._save,
                                   pygame.Rect(x0, fy, 190, 46), style="primary"))
        self.widgets.append(Button("Restore Saved", self._restore,
                                   pygame.Rect(x0 + 204, fy, 170, 46), style="secondary"))
        self.widgets.append(Button("Back", self._back,
                                   pygame.Rect(x0 + 388, fy, 120, 46), style="secondary"))
        if self.note:
            self.widgets.append(Label(self.note, (x0 + 530, fy + 14), font=None, color=self.note_color))

    def _switch_section(self, value: object) -> None:
        if isinstance(value, str) and value in SECTIONS:
            self.section = value
            self.layout()

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
        text(surface, "SETTINGS", (40, 26), font=self.app.fonts.h2, color=COLORS.accent)
        text(surface, f"section: {self.section}  -  changes apply to new games; Save writes config/default.json",
             (260, 36), font=self.app.fonts.small, color=COLORS.text_dim)
        draw_widgets(surface, self.widgets, self.app.fonts)
