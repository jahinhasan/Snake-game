"""Headless UI smoke test: every screen constructs, lays out, and draws.

Uses SDL's dummy video/audio drivers, so it runs anywhere (CI included)
without a display. It exercises the real widget layout + draw paths.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest  # noqa: E402

pygame = pytest.importorskip("pygame", reason="pygame not installed")

from snake_arena.config import AppConfig  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from snake_arena.ui.app import App

    cfg = AppConfig()
    cfg.llm.enabled = False
    window = App(cfg)
    yield window
    window.shutdown()


def _exercise(screen, app) -> None:
    """Push, update, draw a few frames, send a mouse event, pop."""
    app.push_screen(screen)
    for _ in range(3):
        screen.update(1 / 60)
        screen.draw(app.screen)
    pygame.event.pump()
    screen.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(200, 200), rel=(1, 0)))
    app.pop_screen()


def test_menu_screens_render(app):
    from snake_arena.ui.menu import MenuScreen, ModeSelectScreen

    _exercise(MenuScreen(app), app)
    _exercise(ModeSelectScreen(app), app)


def test_dashboard_screens_render(app):
    from snake_arena.ui.benchmark_dashboard import BenchmarkDashboard
    from snake_arena.ui.settings import SettingsScreen
    from snake_arena.ui.statistics import StatisticsScreen
    from snake_arena.ui.training_dashboard import TrainingDashboard

    _exercise(BenchmarkDashboard(app), app)
    _exercise(TrainingDashboard(app), app)
    _exercise(StatisticsScreen(app), app)
    for section in ("Game", "Display & Audio", "Agents", "LLM Planner", "RL & Safety"):
        settings = SettingsScreen(app)
        settings.section = section
        _exercise(settings, app)


def test_arena_and_results_render(app):
    from snake_arena.ui.arena import ArenaScreen
    from snake_arena.ui.results import ResultsScreen

    arena = ArenaScreen(app, mode="ai_vs_ai", agent_a="rule_based", agent_b="astar")
    _exercise(arena, app)

    record = {
        "mode": "ai_vs_ai",
        "timestamp": 0.0,
        "ticks": 10,
        "seconds": 1.0,
        "outcome": "survivor_win",
        "winner_ids": ["a"],
        "strategy_summary": "",
        "ai_summary": {"decisions": 10, "safety_interventions": 0},
        "snakes": {
            "a": {"name": "A", "is_human": False, "score": 30, "length": 6,
                  "food_by_type": {"normal": 3}, "survived": True, "death_reason": None},
            "b": {"name": "B", "is_human": False, "score": 0, "length": 4,
                  "food_by_type": {}, "survived": False, "death_reason": "wall"},
        },
    }
    _exercise(ResultsScreen(app, record, rematch=lambda: None), app)


def test_settings_roundtrip_through_widgets(app, monkeypatch, tmp_path):
    """Moving a slider and saving writes a valid config file.

    The save target is redirected into tmp_path: this test used to write the
    app fixture's settings (LLM disabled) over the user's real
    config/default.json, silently turning the LLM planner off for them.
    """
    from snake_arena.ui.settings import SettingsScreen
    from snake_arena.utils import paths

    monkeypatch.setattr(paths, "default_config_path", lambda: tmp_path / "default.json")
    screen = SettingsScreen(app)
    app.push_screen(screen)
    screen._volume_changed(0.75)
    assert app.config.audio.volume == pytest.approx(0.75)
    screen._save()
    assert (tmp_path / "default.json").exists(), "save must write the redirected config path"
    app.pop_screen()


def test_arena_game_over_buttons_survive_clicks(app):
    """Regression: the game-over buttons must stay clickable across frames.

    draw() used to rebuild the widget list every frame, wiping a pressed
    button's state between MOUSEBUTTONDOWN and MOUSEBUTTONUP, so clicks on
    the end-of-game (and pause) buttons never fired.
    """
    import time as time_mod

    from snake_arena.ui.arena import ArenaScreen

    arena = ArenaScreen(app, mode="ai_vs_ai", agent_a="rule_based", agent_b="rule_based")
    app.push_screen(arena)
    try:
        # Fast-forward a few ticks so the screen is fully laid out.
        for _ in range(3):
            arena.update(1 / 30)
            arena.draw(app.screen)

        # Force the game-over state (deterministic; no long episode needed).
        arena.state_name = "over"
        arena.over_at = time_mod.time() - 2.0  # past the reveal delay
        arena.update(1 / 60)  # reveals the end-of-game buttons
        arena.draw(app.screen)
        assert arena.widgets, "end-of-game buttons must be revealed"

        rematch = next(w for w in arena.widgets if getattr(w, "label", "") == "Rematch")

        # A real click: down, (frames pass, screen redraws), up.
        down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rematch.rect.center, button=1)
        up = pygame.event.Event(pygame.MOUSEBUTTONUP, pos=rematch.rect.center, button=1)
        arena.handle_event(down)
        arena.draw(app.screen)  # a frame passes while the mouse is held
        arena.handle_event(up)
        assert arena.state_name == "running", "Rematch click must restart the game"

        # Pause buttons must be clickable the same way.
        arena.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        assert arena.state_name == "paused"
        resume = next(w for w in arena.widgets if getattr(w, "label", "") == "Resume")
        arena.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=resume.rect.center, button=1))
        arena.draw(app.screen)
        arena.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=resume.rect.center, button=1))
        assert arena.state_name == "running", "Resume click must resume the game"
    finally:
        app.pop_screen()


def test_human_up_arrow_turns_the_snake(app):
    """Regression: Action.UP == 0 is falsy, so `pending or direction` used to
    silently discard UP presses. Every arrow must actually turn the snake.

    Press order UP → LEFT → DOWN → RIGHT is always perpendicular to the
    current heading, so no press is a reversal (reversals are ignored by
    design, on purpose, at the engine level).
    """
    from snake_arena.game.types import Action
    from snake_arena.ui.arena import ArenaScreen

    arena = ArenaScreen(app, mode="vs_ai", agent_a="astar")
    app.push_screen(arena)
    try:
        for key, expected in (
            (pygame.K_UP, Action.UP),
            (pygame.K_LEFT, Action.LEFT),
            (pygame.K_DOWN, Action.DOWN),
            (pygame.K_RIGHT, Action.RIGHT),
        ):
            head_before = arena.engine.state.snakes["human"].head
            arena.pending_human_action = None
            arena.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
            arena._tick()
            moved = arena.engine.state.snakes["human"].head
            assert moved != head_before, f"{expected.name} press was ignored"
            assert arena.engine.state.snakes["human"].direction == expected
    finally:
        app.pop_screen()


def test_settings_llm_status_label_reports_ollama(app, monkeypatch):
    """The LLM Planner section shows a real reachability status, probed in the
    background so the UI never freezes, with honest text when Ollama is absent.
    """
    import time as time_mod

    from snake_arena.config import LLMConfig
    from snake_arena.llm.ollama_client import OllamaClient, OllamaUnavailable
    from snake_arena.ui.settings import SettingsScreen

    # Deterministic probe result: pretend Ollama is unreachable.
    def unavailable(self, path, payload, timeout):
        raise OllamaUnavailable("cannot reach Ollama (test)")

    monkeypatch.setattr(OllamaClient, "_transport", unavailable, raising=False)

    cfg: AppConfig = app.config
    original = cfg.llm
    cfg.llm = LLMConfig()  # enabled, pointing at the default host
    try:
        screen = SettingsScreen(app)
        screen.section = "LLM Planner"
        app.push_screen(screen)
        label = screen._llm_status_label
        assert label is not None, "LLM status label must exist"
        deadline = time_mod.time() + 5.0
        while "checking" in label.content and time_mod.time() < deadline:
            time_mod.sleep(0.05)
        assert "not reachable" in label.content, label.content
        assert "ollama serve" in label.content, "status must say how to fix it"
        # Re-layout with unchanged settings shows the stored verdict, no re-probe.
        screen.layout()
        assert "not reachable" in screen._llm_status_label.content
    finally:
        cfg.llm = original
        if app.screens and app.screens[-1] is screen:
            app.pop_screen()
