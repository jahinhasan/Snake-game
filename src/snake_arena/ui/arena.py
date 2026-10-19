"""Game arena screen: board rendering, human input, live AI diagnostics.

One engine tick advances the world one cell for every snake simultaneously.
Human input is queued and applied on the next tick; AI snakes ask their
DecisionManager (strategy -> policy -> safety) each tick. The right panel
shows the *real* decision trace: proposed vs final action, rejected moves
with reasons, path, target, Q-values, and LLM status.
"""

from __future__ import annotations

import math
import time

import pygame

from ..agents.factory import build_manager
from ..config import AppConfig
from ..evaluation.benchmark import AGENT_LABELS
from ..game.engine import GameEngine, SnakeSpec
from ..game.types import Action, DIR_VECTORS
from ..persistence import history as history_mod
from ..safety.validator import SafetyValidator
from .app import App, Screen
from .components import Button, draw_widgets
from .theme import COLORS, rgb, text, wrap_text

MODE_INFO = {
    "vs_ai": "Human vs AI",
    "ai_vs_ai": "AI vs AI",
}


class ArenaScreen(Screen):
    def __init__(
        self,
        app: App,
        mode: str = "vs_ai",
        agent_a: str = "astar",
        agent_b: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(app)
        self.mode = mode
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.seed = seed
        self.has_human = mode == "vs_ai"
        self.pending_human_action: Action | None = None
        self.state_name = "running"  # running | paused | over
        self.speed = app.config.game.moves_per_second
        self.accumulator = 0.0
        self.over_at: float | None = None
        self.show_fps = app.config.window.fps_counter
        self.record = None
        self._build_game(seed)

    # -- construction -----------------------------------------------------------

    def _build_game(self, seed: int | None = None) -> None:
        config: AppConfig = self.app.config
        w, h = config.board.width, config.board.height
        specs = []
        if self.has_human:
            specs.append(SnakeSpec("human", "You", True, (w // 4, h // 2), Action.RIGHT))
            specs.append(SnakeSpec("ai", AGENT_LABELS.get(self.agent_a, self.agent_a), False,
                                   (3 * w // 4, h // 2), Action.LEFT))
        else:
            specs.append(SnakeSpec("a", AGENT_LABELS.get(self.agent_a, self.agent_a), False,
                                   (w // 4, h // 2), Action.RIGHT))
            b = self.agent_b or "rule_based"
            specs.append(SnakeSpec("b", AGENT_LABELS.get(b, b), False, (3 * w // 4, h // 2), Action.LEFT))
        self.engine = GameEngine(
            width=w,
            height=h,
            moves_per_second=self.speed,
            max_foods=config.game.max_foods,
            seed=seed,
            time_limit_seconds=config.game.time_limit_seconds,
            snake_specs=specs,
        )
        self.validator = SafetyValidator(config.safety)
        self.ai_snakes: list[str] = ["ai"] if self.has_human else ["a", "b"]
        use_llm_worker = self.app.llm_worker() if config.llm.enabled else None
        self.managers = {}
        if self.has_human:
            self.managers["ai"] = build_manager(
                self.agent_a, config, rl_agent=self.app.rl_agent(),
                strategist_worker=use_llm_worker,
            )
        else:
            self.managers["a"] = build_manager(
                self.agent_a, config, rl_agent=self.app.rl_agent(),
                strategist_worker=use_llm_worker,
            )
            self.managers["b"] = build_manager(
                self.agent_b or "rule_based", config, rl_agent=self.app.rl_agent(),
                strategist_worker=use_llm_worker,
            )
        self.ai_decisions: dict[str, object] = {}
        self.state_name = "running"
        self.pending_human_action = None
        self.record = None
        self.over_at = None
        self.over_widgets_shown = False
        self.accumulator = 0.0
        self.risky_cache: dict[str, set] = {}
        self.risky_tick = -1

    def on_enter(self) -> None:
        self.layout()

    # -- events -----------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        super().handle_event(event)
        if event.type == pygame.KEYDOWN:
            key = event.key
            if key in (pygame.K_ESCAPE,):
                if self.state_name == "running":
                    self._pause()
                elif self.state_name == "paused":
                    self._resume()
                return
            if key == pygame.K_SPACE:
                if self.state_name == "running":
                    self._pause()
                elif self.state_name == "paused":
                    self._resume()
                return
            if self.state_name != "running":
                if self.state_name == "over" and key == pygame.K_RETURN:
                    self._open_results()
                return
            human_action = {
                pygame.K_UP: Action.UP, pygame.K_w: Action.UP,
                pygame.K_DOWN: Action.DOWN, pygame.K_s: Action.DOWN,
                pygame.K_LEFT: Action.LEFT, pygame.K_a: Action.LEFT,
                pygame.K_RIGHT: Action.RIGHT, pygame.K_d: Action.RIGHT,
            }.get(key)
            if human_action is not None:
                self.pending_human_action = human_action
                return
            if key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self._change_speed(+1)
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self._change_speed(-1)
            elif key == pygame.K_1:
                self.app.config.visual.show_path = not self.app.config.visual.show_path
            elif key == pygame.K_2:
                self.app.config.visual.show_danger = not self.app.config.visual.show_danger
            elif key == pygame.K_r:
                self._restart()

    def _pause(self) -> None:
        self.state_name = "paused"
        self.app.audio.play("pause")
        self.layout()

    def _resume(self) -> None:
        self.state_name = "running"
        self.app.audio.play("pause")
        self.layout()

    def _change_speed(self, delta: int) -> None:
        self.speed = max(self.app.config.game.min_moves_per_second,
                         min(self.app.config.game.max_moves_per_second, self.speed + delta))
        self.engine.moves_per_second = self.speed
        self.app.audio.play("tick_speed")

    def _restart(self) -> None:
        self._build_game(seed=None)
        self.layout()

    def _menu(self) -> None:
        self.go_menu()

    def _open_results(self) -> None:
        from .results import ResultsScreen

        self.app.push_screen(ResultsScreen(self.app, self.record, rematch=self._restart))

    # -- simulation ---------------------------------------------------------------

    def update(self, dt: float) -> None:
        if self.state_name == "over":
            # Reveal the end-of-game buttons once, shortly after the banner.
            if (not self.over_widgets_shown and self.over_at is not None
                    and time.time() - self.over_at > 0.9):
                self.over_widgets_shown = True
                self.layout()
            return
        if self.state_name != "running":
            return
        self.accumulator += dt
        step_time = 1.0 / max(1, self.speed)
        guard = 0
        while self.accumulator >= step_time and self.state_name == "running" and guard < 4:
            self.accumulator -= step_time
            guard += 1
            self._tick()

    def _tick(self) -> None:
        actions: dict[str, Action] = {}
        if self.has_human:
            human = self.engine.state.snakes["human"]
            # NB: `Action.UP == 0` is falsy -- must compare with None explicitly.
            action = self.pending_human_action if self.pending_human_action is not None else human.direction
            if human.is_reversal(action):
                action = human.direction
            actions["human"] = action
            self.pending_human_action = None
        for snake_id, manager in self.managers.items():
            if not self.engine.state.snakes[snake_id].alive:
                continue
            chosen = manager.decide(self.engine.state, snake_id)
            self.ai_decisions[snake_id] = chosen
            actions[snake_id] = chosen
        result = self.engine.step(actions)

        for event in result.events:
            if event.kind == "eat":
                self.app.audio.play("golden" if event.value >= 50 else "eat")
            elif event.kind == "trap":
                self.app.audio.play("trap")
            elif event.kind == "death":
                self.app.audio.play("death")

        if result.done:
            self.state_name = "over"
            self.over_at = time.time()
            if self.engine.outcome in ("survivor_win", "timeout"):
                self.app.audio.play("win")
            self._finalize_record()

    def _finalize_record(self) -> None:
        agent_names = ",".join(AGENT_LABELS.get(a, a) for a in ([self.agent_a] if self.has_human else [self.agent_a, self.agent_b or "rule_based"]))
        mgr = next(iter(self.managers.values()))
        llm_info = {
            "enabled": self.app.config.llm.enabled,
            "model": self.app.config.llm.model if self.app.config.llm.enabled else "",
            "available": mgr.stats["llm_calls"] > 0 and mgr.stats["llm_failures"] == 0,
            "calls": mgr.stats["llm_calls"],
            "failures": mgr.stats["llm_failures"],
        }
        strategy_summary = ""
        if mgr.diagnostics:
            strategies = [entry.strategy_name for entry in mgr.diagnostics if entry.strategy_source != "default"]
            if strategies:
                counts: dict[str, int] = {}
                for s in strategies:
                    counts[s] = counts.get(s, 0) + 1
                top = max(counts, key=counts.get)
                strategy_summary = f"{top} ({counts[top]}/{len(strategies)} ticks, source={mgr.diagnostics[-1].strategy_source})"
        self.record = history_mod.build_game_record(
            mode=f"{self.mode}:{agent_names}",
            agent_name=agent_names,
            engine_state=self.engine.state,
            outcome=self.engine.outcome,
            winner_ids=self.engine.winner_ids,
            manager_stats=dict(mgr.stats),
            strategy_summary=strategy_summary,
            llm_info=llm_info,
            seed=self.seed,
        )
        history_mod.record_game(self.record)

    # -- layout ----------------------------------------------------------------

    def layout(self) -> None:
        from .components import Label

        w, h = self.app.screen.get_size()
        self.panel_w = 320
        self.top_h = 46
        self.bottom_h = 30
        self.panel = pygame.Rect(w - self.panel_w - 12, self.top_h + 8, self.panel_w, h - self.top_h - self.bottom_h - 16)
        self.board_area = pygame.Rect(12, self.top_h + 8, w - self.panel_w - 36, h - self.top_h - self.bottom_h - 16)
        cell = max(8, min(self.board_area.w // self.engine.width, self.board_area.h // self.engine.height))
        self.cell = cell
        bw, bh = cell * self.engine.width, cell * self.engine.height
        self.board_rect = pygame.Rect(
            self.board_area.x + (self.board_area.w - bw) // 2,
            self.board_area.y + (self.board_area.h - bh) // 2,
            bw, bh,
        )
        if self.state_name == "paused":
            cx, cy = w // 2, h // 2
            self.widgets = [
                Button("Resume", self._resume, pygame.Rect(cx - 110, cy - 70, 220, 48), style="primary"),
                Button("Restart", self._restart, pygame.Rect(cx - 110, cy - 12, 220, 48), style="secondary"),
                Button("Main Menu", self._menu, pygame.Rect(cx - 110, cy + 46, 220, 48), style="danger"),
            ]
        elif self.state_name == "over":
            cx, cy = w // 2, h // 2
            self.widgets = [
                Button("Results & Analysis", self._open_results, pygame.Rect(cx - 110, cy + 40, 220, 46), style="primary"),
                Button("Rematch", self._restart, pygame.Rect(cx - 110, cy + 96, 220, 46), style="secondary"),
                Button("Main Menu", self._menu, pygame.Rect(cx - 110, cy + 152, 220, 46), style="secondary"),
            ]
        else:
            self.widgets = []

    # -- drawing ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        surface.fill(rgb(COLORS.bg))
        self._draw_board(surface)
        self._draw_right_panel(surface)
        self._draw_top_bar(surface, w)
        self._draw_bottom_bar(surface, w, h)
        if self.state_name == "over" and self.record is not None:
            self._draw_game_over_banner(surface, w, h)
        draw_widgets(surface, self.widgets, self.app.fonts)

    def _draw_board(self, surface: pygame.Surface) -> None:
        cell = self.cell
        rect = self.board_rect
        pygame.draw.rect(surface, rgb("#10161f"), rect, border_radius=12)
        pygame.draw.rect(surface, rgb(COLORS.border), rect, width=1, border_radius=12)
        state = self.engine.state
        ox, oy = rect.x, rect.y
        visual = self.app.config.visual

        if visual.show_grid and cell >= 12:
            for gx in range(1, state.width):
                pygame.draw.line(surface, rgb("#161d29"), (ox + gx * cell, oy), (ox + gx * cell, oy + rect.h))
            for gy in range(1, state.height):
                pygame.draw.line(surface, rgb("#161d29"), (ox, oy + gy * cell), (ox + rect.w, oy + gy * cell))

        # Danger overlay: opponent reach + validator-risky cells for the AI.
        if visual.show_danger and state.tick != self.risky_tick:
            self.risky_cache = {}
            for snake_id in self.ai_snakes:
                if state.snakes[snake_id].alive:
                    self.risky_cache[snake_id] = self.validator.risky_cells(state, snake_id)
            self.risky_tick = state.tick
        if visual.show_danger:
            risky: set = set()
            for cells in self.risky_cache.values():
                risky |= cells
            for cx_, cy_ in risky:
                cell_rect = pygame.Rect(ox + cx_ * cell, oy + cy_ * cell, cell, cell)
                pygame.draw.rect(surface, rgb(COLORS.danger), cell_rect.inflate(-4, -4), width=2, border_radius=6)

        # AI planned path(s).
        if visual.show_path:
            for snake_id, decision in self.ai_decisions.items():
                path = getattr(decision, "path", [])
                color = COLORS.accent if (self.has_human or snake_id == "a") else COLORS.purple
                for i, (px, py) in enumerate(path[1:], start=1):
                    r = pygame.Rect(ox + px * cell, oy + py * cell, cell, cell)
                    pygame.draw.rect(surface, rgb(color), r.inflate(-cell // 2, -cell // 2),
                                     width=2, border_radius=5)

        # Target ring.
        if visual.show_target:
            for snake_id, decision in self.ai_decisions.items():
                tpos = getattr(decision, "target_pos", None)
                if tpos:
                    r = pygame.Rect(ox + tpos[0] * cell, oy + tpos[1] * cell, cell, cell)
                    pygame.draw.rect(surface, rgb(COLORS.gold), r.inflate(-2, -2), width=2, border_radius=6)

        # Food.
        now = pygame.time.get_ticks() / 1000.0
        for food in state.foods:
            fx, fy = food.pos
            center = (ox + fx * cell + cell // 2, oy + fy * cell + cell // 2)
            radius = max(4, int(cell * 0.36))
            remaining = food.remaining_ratio(state.tick)
            color = rgb(food.color)
            if food.value < 0:
                self._draw_skull(surface, center, radius, color)
            elif food.shape == "diamond":
                pygame.draw.polygon(surface, color, [
                    (center[0], center[1] - radius), (center[0] + radius, center[1]),
                    (center[0], center[1] + radius), (center[0] - radius, center[1]),
                ])
            elif food.shape == "star":
                self._draw_star(surface, center, radius, color, points=5)
            elif food.shape == "burst":
                self._draw_star(surface, center, radius, color, points=8)
            else:
                pygame.draw.circle(surface, color, center, radius)
            if remaining < 0.28 and food.value > 0:  # expiring soon: pulse ring
                pulse = 1.0 + 0.35 * math.sin(now * 8.0)
                pygame.draw.circle(surface, rgb(COLORS.gold), center, int(radius * pulse + 3), width=1)

        # Snakes.
        for snake in state.snakes.values():
            if not snake.alive:
                continue
            base = rgb(COLORS.human if snake.is_human else COLORS.ai)
            n = max(1, snake.length)
            for i, seg in enumerate(snake.body):
                factor = 1.0 - 0.55 * (i / n)
                color = tuple(int(c * factor) for c in base)
                r = pygame.Rect(ox + seg[0] * cell, oy + seg[1] * cell, cell, cell)
                pygame.draw.rect(surface, color, r.inflate(-max(1, cell // 10), -max(1, cell // 10)),
                                 border_radius=max(3, cell // 3))
            head = snake.head
            head_center = (ox + head[0] * cell + cell // 2, oy + head[1] * cell + cell // 2)
            dx, dy = DIR_VECTORS[snake.direction]
            eye_off = max(2, cell // 7)
            for side in (-1, 1):
                ex = head_center[0] + dy * side * eye_off + dx * eye_off
                ey = head_center[1] + dx * side * eye_off + dy * eye_off
                pygame.draw.circle(surface, (10, 12, 16), (ex, ey), max(1, cell // 10))

            # Rejected-move arrows.
            if visual.show_rejected and not snake.is_human:
                manager = self.managers.get(snake.id) or self.managers.get("ai")
                log = getattr(manager, "current_log", None)
                if log is not None and log.safety_rejected:
                    for name, _reason in log.safety_rejected[:2]:
                        try:
                            raction = Action[name.upper()]
                        except KeyError:
                            continue
                        rdx, rdy = DIR_VECTORS[raction]
                        start = (head_center[0] + rdx * cell // 2, head_center[1] + rdy * cell // 2)
                        end = (head_center[0] + rdx * cell, head_center[1] + rdy * cell)
                        pygame.draw.line(surface, rgb(COLORS.danger), start, end, 3)
                        pygame.draw.circle(surface, rgb(COLORS.danger), end, 3)

    @staticmethod
    def _draw_star(surface: pygame.Surface, center, radius: int, color, points: int) -> None:
        pts = []
        for i in range(points * 2):
            r = radius if i % 2 == 0 else radius * 0.45
            a = math.pi * i / points - math.pi / 2
            pts.append((center[0] + r * math.cos(a), center[1] + r * math.sin(a)))
        pygame.draw.polygon(surface, color, pts)

    @staticmethod
    def _draw_skull(surface: pygame.Surface, center, radius: int, color) -> None:
        pygame.draw.circle(surface, color, center, radius)
        pygame.draw.rect(surface, color, (center[0] - radius // 2, center[1], radius, radius))
        eye_r = max(2, radius // 4)
        for side in (-1, 1):
            pygame.draw.circle(surface, (13, 17, 23),
                               (center[0] + side * radius // 2, center[1] - radius // 4), eye_r)
        pygame.draw.line(surface, (13, 17, 23), (center[0], center[1] + radius // 5),
                         (center[0], center[1] + radius // 2), 2)

    # -- panels ---------------------------------------------------------------------

    def _draw_top_bar(self, surface: pygame.Surface, w: int) -> None:
        state = self.engine.state
        seconds_left = max(0.0, (state.max_ticks - state.tick) / max(1, state.moves_per_second))
        mm, ss = divmod(int(seconds_left), 60)
        mode_label = MODE_INFO.get(self.mode, self.mode)
        text(surface, mode_label, (14, 8), font=self.app.fonts.h3, color=COLORS.text)
        text(surface, f"Time {mm:02d}:{ss:02d}", (w // 2, 10), font=self.app.fonts.h3,
             color=COLORS.text_dim, align="midleft")
        text(surface, f"Speed: {self.speed}/s  (+/-)", (w - 320, 10), font=self.app.fonts.small,
             color=COLORS.text_dim, align="midleft")
        if self.show_fps:
            text(surface, f"FPS {self.app.clock.get_fps():.0f}", (w - 14, 10),
                 font=self.app.fonts.small, color=COLORS.text_faint, align="topright")

    def _draw_bottom_bar(self, surface: pygame.Surface, w: int, h: int) -> None:
        hint = "Move: WASD / Arrows    Space/Esc: pause    R: restart    1: path    2: danger    +/-: speed"
        if not self.has_human:
            hint = "Space/Esc: pause    R: restart    1: path overlay    2: danger overlay    +/-: speed"
        text(surface, hint, (14, h - 20), font=self.app.fonts.small, color=COLORS.text_faint)

    def _draw_right_panel(self, surface: pygame.Surface) -> None:
        panel = self.panel
        theme_panel_fill = COLORS.panel
        pygame.draw.rect(surface, rgb(theme_panel_fill), panel, border_radius=12)
        pygame.draw.rect(surface, rgb(COLORS.border), panel, width=1, border_radius=12)
        x0 = panel.x + 14
        width = panel.w - 28
        y = panel.y + 12

        # Scoreboard rows.
        for snake in self.engine.state.snakes.values():
            color = COLORS.human if snake.is_human else COLORS.ai
            if not self.has_human:
                color = COLORS.ai if snake.id == "a" else COLORS.purple
            dot = pygame.Rect(x0, y + 4, 12, 12)
            pygame.draw.circle(surface, rgb(color), dot.center, 6)
            text(surface, snake.name, (x0 + 20, y), font=self.app.fonts.h3, color=COLORS.text)
            text(surface, f"{snake.score}", (panel.right - 14, y), font=self.app.fonts.h3,
                 color=COLORS.text, align="topright")
            y += 24
            text(surface, f"length {snake.length}   food {sum(snake.foods_eaten.values())}   traps {snake.foods_eaten.get('trap', 0)}",
                 (x0 + 20, y), font=self.app.fonts.small, color=COLORS.text_dim)
            y += 22
        y += 6
        pygame.draw.line(surface, rgb(COLORS.border), (x0, y), (panel.right - 14, y))
        y += 10

        # AI diagnostics.
        text(surface, "AI DIAGNOSTICS", (x0, y), font=self.app.fonts.small, color=COLORS.accent)
        y += 22
        lines = self._diagnostic_lines()
        font = self.app.fonts.mono_small
        max_y = panel.bottom - 24
        for line in lines:
            for wrapped in wrap_text(line, font, width)[:2]:
                if y > max_y:
                    break
                text(surface, wrapped, (x0, y), font=font, color=COLORS.text_dim)
                y += 17
            if y > max_y:
                text(surface, "...", (x0, y), font=font, color=COLORS.text_faint)
                break

        # Legend pinned to panel bottom.
        legend_y = panel.bottom - 20
        items = [
            (COLORS.human if self.has_human else COLORS.ai, "you/human" if self.has_human else "agent A"),
            (COLORS.ai if self.has_human else COLORS.purple, "AI" if self.has_human else "agent B"),
            (COLORS.accent, "path"), (COLORS.gold, "target"), (COLORS.danger, "risk"),
        ]
        lx = x0
        for color, label in items:
            pygame.draw.circle(surface, rgb(color), (lx + 5, legend_y + 7), 4)
            lx = text(surface, label, (lx + 14, legend_y), font=self.app.fonts.tiny, color=COLORS.text_faint).right + 10

    def _diagnostic_lines(self) -> list[str]:
        lines: list[str] = []
        for snake_id in self.ai_snakes:
            manager = self.managers.get(snake_id)
            if manager is None:
                continue
            decision = self.ai_decisions.get(snake_id)
            label = AGENT_LABELS.get(manager.mode, manager.mode)
            lines.append(f"[{getattr(self.engine.state.snakes[snake_id], 'name', snake_id)}] {label}")
            log = manager.current_log
            if decision is not None and log is not None:
                arrow = {"up": "^", "down": "v", "left": "<", "right": ">"}[log.final_action]
                line = f"  action: {log.proposed_action} -> {log.final_action} {arrow}"
                if log.safety_intervened:
                    line += "  (safety override)"
                lines.append(line)
                if log.target_food_id:
                    lines.append(f"  target: {log.target_food_id}  path: {log.path_length}")
                if log.q_values and self.app.config.visual.show_qvalues:
                    qs = "  ".join(f"{k}:{v:+.2f}" for k, v in log.q_values.items())
                    lines.append(f"  Q {qs}")
                if log.strategy_source != "default":
                    lines.append(f"  strategy: {log.strategy_name} ({log.strategy_source})")
                    if log.strategy_reason:
                        lines.append(f"  reason: {log.strategy_reason}")
                    if log.llm_error:
                        lines.append(f"  llm: {log.llm_error[:60]}")
                    elif log.strategy_source == "llm":
                        lines.append(f"  llm: ok {log.latency_ms:.0f}ms")
                for note in log.notes[-2:]:
                    lines.append(f"  note: {note[:64]}")
            elif decision is None:
                lines.append("  (no decision yet)")
        if not self.app.config.llm.enabled:
            lines.append("LLM planner disabled in settings")
        return lines

    def _draw_game_over_banner(self, surface: pygame.Surface, w: int, h: int) -> None:
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((5, 8, 12, 160))
        surface.blit(overlay, (0, 0))
        outcome = self.engine.outcome
        if self.has_human:
            human = self.engine.state.snakes["human"]
            if not human.alive:
                banner = "YOU DIED"
                sub = f"cause: {human.death_reason}"
                color = COLORS.danger
            else:
                banner = "YOU WIN!"
                sub = f"score {human.score}"
                color = COLORS.success
        else:
            if outcome == "draw":
                banner = "DRAW"
                color = COLORS.gold
            elif outcome == "timeout":
                winner_names = [self.engine.state.snakes[sid].name for sid in self.engine.winner_ids]
                banner = f"TIME - {winner_names[0] if winner_names else 'tie'} leads"
                color = COLORS.gold
            else:
                winner_names = [self.engine.state.snakes[sid].name for sid in self.engine.winner_ids]
                banner = f"{winner_names[0]} WINS" if winner_names else "GAME OVER"
                color = COLORS.success
            sub = "  vs  ".join(
                f"{s.name}: {s.score}" for s in self.engine.state.snakes.values()
            )
        text(surface, banner, (w // 2, h // 2 - 60), font=self.app.fonts.title, color=color, align="center", shadow=True)
        text(surface, sub, (w // 2, h // 2), font=self.app.fonts.h3, color=COLORS.text_dim, align="center")
        text(surface, "Press Enter for results", (w // 2, h // 2 + 34), font=self.app.fonts.small,
             color=COLORS.text_faint, align="center")
