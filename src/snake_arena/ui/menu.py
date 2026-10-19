"""Main menu and game-mode selection screens."""

from __future__ import annotations

import threading

import pygame

from .. import PROJECT_NAME, __version__
from ..agents.factory import build_manager
from ..config import AppConfig
from ..game.engine import GameEngine, SnakeSpec
from ..game.types import Action
from .app import App, Screen
from .components import Button, Dropdown, Label, Slider, draw_widgets, send_event
from .theme import COLORS, panel, rgb, text


class AnimatedSnakeBackground:
    """A dimmed autonomous game running behind the menu (real engine + agent)."""

    def __init__(self, config: AppConfig, width: int, height: int) -> None:
        self.engine = GameEngine(
            width=width,
            height=height,
            moves_per_second=9,
            max_foods=4,
            seed=None,
            time_limit_seconds=60,
            snake_specs=[SnakeSpec("bg", "bg", False, (width // 4, height // 2), Action.RIGHT)],
        )
        self.manager = build_manager("rule_based", config)
        self.accumulator = 0.0

    def update(self, dt: float) -> None:
        self.accumulator += dt
        while self.accumulator > 1.0 / 9.0:
            self.accumulator -= 1.0 / 9.0
            if self.engine.finished:
                self.engine.reset(seed=None)
            action = self.manager.decide(self.engine.state, "bg")
            self.engine.step({"bg": action})

    def draw(self, surface: pygame.Surface, board_px: tuple[int, int], offset: tuple[int, int]) -> None:
        cell = min(board_px[0] // self.engine.width, board_px[1] // self.engine.height)
        if cell < 4:
            return
        ox, oy = offset
        w, h = self.engine.width * cell, self.engine.height * cell
        dim = pygame.Surface((w, h), pygame.SRCALPHA)
        dim.fill((13, 17, 23, 170))
        surface.blit(dim, (ox, oy))
        for food in self.engine.state.foods:
            rect = pygame.Rect(ox + food.pos[0] * cell, oy + food.pos[1] * cell, cell, cell)
            pygame.draw.rect(surface, rgb(food.color), rect.inflate(-cell // 2, -cell // 2), border_radius=3)
        for snake in self.engine.state.snakes.values():
            for i, seg in enumerate(snake.body):
                rect = pygame.Rect(ox + seg[0] * cell + 1, oy + seg[1] * cell + 1, cell - 2, cell - 2)
                shade = 255 - min(150, i * 4)
                pygame.draw.rect(surface, (shade // 3, shade // 2, shade), rect, border_radius=max(2, cell // 4))


class MenuScreen(Screen):
    def __init__(self, app: App) -> None:
        super().__init__(app)
        self.background: AnimatedSnakeBackground | None = None
        self.ollama_label = Label("LLM: checking...", (0, 0), font=None, color=COLORS.text_dim)
        self.time = 0.0

    def on_enter(self) -> None:
        self.background = AnimatedSnakeBackground(self.app.config, 40, 24)
        self.layout()
        self._probe_ollama()

    def layout(self) -> None:
        surface = self.app.screen
        w, h = surface.get_size()
        cx = w // 2
        button_w, button_h = 340, 54
        gap = 14
        total = 6 * button_h + 5 * gap
        y0 = h // 2 - total // 2 + 20
        items = [
            ("Play", self._open_play, "primary"),
            ("AI Training", lambda: self._goto("training"), "secondary"),
            ("Agent Benchmark", lambda: self._goto("benchmark"), "secondary"),
            ("Statistics", lambda: self._goto("statistics"), "secondary"),
            ("Settings", lambda: self._goto("settings"), "secondary"),
            ("Exit", self.app.quit, "danger"),
        ]
        self.widgets = []
        for i, (label, cb, style) in enumerate(items):
            rect = pygame.Rect(cx - button_w // 2, y0 + i * (button_h + gap), button_w, button_h)
            self.widgets.append(Button(label, cb, rect, style=style))
        self.ollama_label.rect.topleft = (24, h - 40)

    def _probe_ollama(self) -> None:
        def probe() -> None:
            client = self.app.llm_client()
            if client is None:
                self.ollama_label.set("LLM planner: disabled in settings", COLORS.text_faint)
                return
            ok, detail = client.is_available()
            if ok:
                has, model_note = client.has_model()
                if has:
                    self.ollama_label.set(f"LLM planner: ready ({model_note})", COLORS.success)
                else:
                    self.ollama_label.set(f"Ollama running, but {model_note}", COLORS.gold)
            else:
                self.ollama_label.set("LLM planner: Ollama not reachable (optional - game works without it)", COLORS.text_dim)

        threading.Thread(target=probe, daemon=True).start()

    def _open_play(self) -> None:
        self.app.push_screen(ModeSelectScreen(self.app))

    def _goto(self, name: str) -> None:
        if name == "training":
            from .training_dashboard import TrainingDashboard

            self.app.push_screen(TrainingDashboard(self.app))
        elif name == "benchmark":
            from .benchmark_dashboard import BenchmarkDashboard

            self.app.push_screen(BenchmarkDashboard(self.app))
        elif name == "statistics":
            from .statistics import StatisticsScreen

            self.app.push_screen(StatisticsScreen(self.app))
        elif name == "settings":
            from .settings import SettingsScreen

            self.app.push_screen(SettingsScreen(self.app))

    def update(self, dt: float) -> None:
        self.time += dt
        if self.background:
            self.background.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        surface.fill(rgb(COLORS.bg))
        if self.background:
            self.background.draw(surface, (w, h), (w // 2 - w * 0.3, 0))
        # Title block
        title_y = int(h * 0.16)
        text(surface, "SNAKE ARENA", (w // 2, title_y), font=self.app.fonts.title,
             color=COLORS.accent, align="center", shadow=True)
        text(surface, PROJECT_NAME, (w // 2, title_y + 58), font=self.app.fonts.h3,
             color=COLORS.text_dim, align="center")
        text(surface, "A*  -  BFS  -  Deep Q-Networks  -  Local LLM Strategy  -  Safety Validation",
             (w // 2, title_y + 86), font=self.app.fonts.small, color=COLORS.text_faint, align="center")
        draw_widgets(surface, self.widgets, self.app.fonts)
        self.ollama_label.draw(surface, self.app.fonts)
        text(surface, f"v{__version__}", (w - 16, h - 40), font=self.app.fonts.small,
             color=COLORS.text_faint, align="topright")


class ModeSelectScreen(Screen):
    MODES = [
        ("vs_astar", "Human vs A* Agent"),
        ("vs_bfs", "Human vs BFS Agent"),
        ("vs_rule", "Human vs Rule-Based Agent"),
        ("vs_rl", "Human vs RL Agent (DQN)"),
        ("vs_llm_astar", "Human vs LLM + A* Agent"),
        ("vs_llm_rl", "Human vs LLM + RL Agent"),
        ("vs_hybrid", "Human vs Full Hybrid Agent"),
        ("ai_vs_ai", "AI vs AI (choose both agents)"),
    ]
    BOARD_PRESETS = [
        ((16, 12), "Small  16x12"),
        ((24, 18), "Standard  24x18"),
        ((32, 24), "Large  32x24"),
    ]

    def __init__(self, app: App) -> None:
        super().__init__(app)
        self.mode = "vs_astar"
        self.preset = (24, 18)
        self.speed = float(app.config.game.moves_per_second)
        self.ai_a = "astar"
        self.ai_b = "rule_based"

    def on_enter(self) -> None:
        self.layout()

    def layout(self) -> None:
        w, h = self.app.screen.get_size()
        panel_rect = pygame.Rect(w // 2 - 340, h // 2 - 290, 680, 580)
        self.panel = panel_rect
        x0 = panel_rect.x + 40
        y = panel_rect.y + 24
        self.widgets = [
            Label("Choose a game mode", (x0, y), font=self.app.fonts.h2),
        ]
        y += 44
        self.mode_dd = Dropdown(
            [(value, label) for value, label in self.MODES],
            self.mode,
            self._on_mode,
            (x0, y),
            width=380,
            font=self.app.fonts.h3,
        )
        self.widgets.append(self.mode_dd)
        y += 46
        self.dd_a = Dropdown(
            [(m, self._agent_label(m)) for m in ("rule_based", "astar", "bfs", "rl", "llm_astar", "llm_rl", "hybrid")],
            self.ai_a, lambda v: setattr(self, "ai_a", v), (x0, y), width=250)
        self.dd_b = Dropdown(
            [(m, self._agent_label(m)) for m in ("rule_based", "astar", "bfs", "rl", "llm_astar", "llm_rl", "hybrid")],
            self.ai_b, lambda v: setattr(self, "ai_b", v), (x0 + 270, y), width=250)
        self.widgets += [self.dd_a, self.dd_b]
        y += 50
        self.preset_dd = Dropdown(
            [((wh), label) for wh, label in self.BOARD_PRESETS],
            self.preset, lambda v: setattr(self, "preset", v), (x0, y), width=250, font=self.app.fonts.body)
        self.widgets.append(self.preset_dd)
        y += 50
        self.speed_slider = Slider("Game speed", self.speed, 3, 25, lambda v: setattr(self, "speed", v),
                                   (x0, y), width=300, step=1.0, fmt=lambda v: f"{int(v)} moves/s")
        self.widgets.append(self.speed_slider)
        y += 70
        self.widgets.append(Button(
            "Start Game", self._start,
            pygame.Rect(x0, y, 200, 52), style="primary", font=self.app.fonts.h3))
        self.widgets.append(Button(
            "Back", self.app.pop_screen,
            pygame.Rect(x0 + 220, y, 140, 52), style="secondary"))

    @staticmethod
    def _agent_label(mode: str) -> str:
        from ..evaluation.benchmark import AGENT_LABELS

        return AGENT_LABELS.get(mode, mode.replace("_", " ").title())

    def _on_mode(self, value: str) -> None:
        self.mode = value

    def _start(self) -> None:
        from .arena import ArenaScreen

        config = self.app.config
        config.board.width, config.board.height = self.preset
        config.game.moves_per_second = int(self.speed)
        mode = self.mode
        if mode == "ai_vs_ai":
            self.app.push_screen(ArenaScreen(self.app, mode="ai_vs_ai",
                                             agent_a=self.ai_a, agent_b=self.ai_b))
        else:
            agent = {
                "vs_astar": "astar", "vs_bfs": "bfs", "vs_rule": "rule_based",
                "vs_rl": "rl", "vs_llm_astar": "llm_astar", "vs_llm_rl": "llm_rl",
                "vs_hybrid": "hybrid",
            }[mode]
            self.app.push_screen(ArenaScreen(self.app, mode="vs_ai", agent_a=agent))

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(rgb(COLORS.bg))
        panel(surface, self.panel, fill=COLORS.panel)
        note = (
            "AI vs AI: both snakes are autonomous agents competing for the same food."
            if self.mode == "ai_vs_ai"
            else "You control the white snake. Arrow keys / WASD to move, Space to pause."
        )
        text(surface, note, (self.panel.x + 40, self.panel.bottom - 46),
             font=self.app.fonts.small, color=COLORS.text_dim)
        draw_widgets(surface, self.widgets, self.app.fonts)
