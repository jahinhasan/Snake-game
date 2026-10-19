"""Application shell: window, screen stack, shared services (audio, LLM)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pygame

from .. import PROJECT_NAME, __version__
from ..config import AppConfig
from ..llm.ollama_client import OllamaClient
from ..llm.strategist import StrategyWorker
from .audio import AudioManager
from .components import Widget
from .theme import COLORS, Fonts

if TYPE_CHECKING:
    from ..agents.rl_agent import RLAgent

log = logging.getLogger("snake_arena.app")


class Screen:
    """A full-window view; screens own their widget list and layout."""

    def __init__(self, app: "App") -> None:
        self.app = app
        self.widgets: list[Widget] = []

    def on_enter(self) -> None:
        """Called when the screen becomes active (layout/rebuild here)."""

    def on_exit(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event) -> None:
        from .components import send_event

        send_event(self.widgets, event)

    def update(self, dt: float) -> None:
        pass

    def draw(self, surface: pygame.Surface) -> None:
        from .components import draw_widgets

        draw_widgets(surface, self.widgets, self.app.fonts)

    # Convenience for subclasses
    def layout(self) -> None:
        """Recompute widget rects for the current window size."""

    def go_menu(self) -> None:
        from .menu import MenuScreen

        self.app.replace_screen(MenuScreen(self.app))


class App:
    """Owns the window, main loop, and cross-screen services."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        pygame.init()
        pygame.display.set_caption(PROJECT_NAME)
        flags = pygame.RESIZABLE
        if config.window.fullscreen:
            flags |= pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((config.window.width, config.window.height), flags)
        pygame.display.set_icon(self._make_icon())
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()
        self.audio = AudioManager(config.audio)
        self.running = True
        self.screens: list[Screen] = []
        self._llm_client: OllamaClient | None = None
        self._llm_worker: StrategyWorker | None = None
        self._rl_agent_cache: dict[str, "RLAgent"] = {}

    # -- screen stack ----------------------------------------------------------

    def push_screen(self, screen: Screen) -> None:
        if self.screens:
            self.screens[-1].on_exit()
        self.screens.append(screen)
        screen.on_enter()

    def pop_screen(self) -> None:
        if self.screens:
            self.screens[-1].on_exit()
            self.screens.pop()
        if self.screens:
            self.screens[-1].on_enter()

    def replace_screen(self, screen: Screen) -> None:
        while self.screens:
            self.screens[-1].on_exit()
            self.screens.pop()
        self.push_screen(screen)

    @property
    def active_screen(self) -> Screen | None:
        return self.screens[-1] if self.screens else None

    # -- shared services ----------------------------------------------------------

    def llm_client(self) -> OllamaClient | None:
        """Client built from current config; None when LLM disabled."""
        if self._llm_client is None:
            from ..agents.factory import build_llm_client

            self._llm_client = build_llm_client(self.config)
        return self._llm_client

    def llm_worker(self) -> StrategyWorker | None:
        """Background strategist worker for interactive LLM modes."""
        if self._llm_worker is None:
            client = self.llm_client()
            if client is None:
                return None
            from ..agents.factory import build_strategy_worker

            self._llm_worker = build_strategy_worker(client, self.config)
            self._llm_worker.start()
        return self._llm_worker

    def rl_agent(self, reload: bool = False) -> "RLAgent":
        """Shared RL agent (checkpoint loaded once per app run)."""
        key = self.config.agent.rl_checkpoint
        if reload or key not in self._rl_agent_cache:
            from ..agents.factory import build_rl_agent

            self._rl_agent_cache[key] = build_rl_agent(self.config)
        return self._rl_agent_cache[key]

    # -- lifecycle ----------------------------------------------------------

    def run(self) -> None:
        from .menu import MenuScreen

        self.push_screen(MenuScreen(self))
        while self.running and self.screens:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.WINDOWRESIZED:
                    for screen in self.screens:
                        screen.layout()
                if self.screens:
                    self.screens[-1].handle_event(event)
            if not self.screens:
                break
            screen = self.screens[-1]
            screen.update(dt)
            screen.draw(self.screen)
            pygame.display.flip()
        self.shutdown()

    def shutdown(self) -> None:
        if self._llm_worker is not None:
            self._llm_worker.stop()
        self.audio.shutdown()
        pygame.quit()

    def quit(self) -> None:
        self.running = False

    # -- misc -----------------------------------------------------------------

    @staticmethod
    def _make_icon() -> pygame.Surface:
        icon = pygame.Surface((32, 32), pygame.SRCALPHA)
        icon.fill((13, 17, 23, 255))
        body = [(6, 16), (10, 12), (14, 12), (18, 16), (22, 16), (24, 18), (22, 20), (16, 20), (10, 20), (6, 20)]
        pygame.draw.polygon(icon, (88, 166, 255), [(x * 32 // 32, y) for x, y in [(2, 10), (6, 6), (10, 8), (14, 12), (22, 12), (26, 16), (22, 20), (12, 20), (4, 18), (2, 14)]])
        pygame.draw.circle(icon, (126, 231, 135), (26, 8), 4)
        pygame.draw.circle(icon, (240, 246, 252), (25, 6), 1)
        return icon


def run_app(config: AppConfig) -> int:
    try:
        app = App(config)
    except pygame.error as exc:
        log.error("Could not open a window: %s", exc)
        return 1
    try:
        app.run()
    except Exception:  # noqa: BLE001 - crash reports beat silent death
        log.exception("Unhandled error in main loop")
        return 1
    return 0
