"""RL training dashboard: live curves, worker-thread controls, checkpoints.

Training runs in a background thread (``rl.trainer.training_worker``) with
pause/stop flags; the dashboard renders the trainer's *recorded* history --
never synthetic data. A network with zero episodes is labelled untrained.
"""

from __future__ import annotations

import logging
import threading

import pygame

from ..config import AppConfig
from ..game.environment import SnakeEnv
from ..rl.checkpoints import CheckpointError, describe_checkpoint
from ..rl.trainer import DQNTrainer, TrainingControl, training_worker
from ..utils import paths
from .app import App, Screen
from .components import Button, Chart, Label, TextInput, draw_widgets, send_event
from .theme import COLORS, rgb, text

log = logging.getLogger("snake_arena.ui.training")


def _downsample(values: list, target: int = 400) -> list:
    if len(values) <= target:
        return list(values)
    step = len(values) / target
    return [values[int(i * step)] for i in range(target)]


class TrainingDashboard(Screen):
    def __init__(self, app: App) -> None:
        super().__init__(app)
        self.trainer: DQNTrainer | None = None
        self.control = TrainingControl()
        self.worker_thread: threading.Thread | None = None
        self.eval_thread: threading.Thread | None = None
        self.status = "idle"
        self.eval_note = ""
        self.checkpoint_note = ""
        self.load_error = ""

    # -- lifecycle ------------------------------------------------------------

    def on_enter(self) -> None:
        if self.trainer is None:
            self._new_trainer(load_existing=True)
        self.layout()

    def _checkpoint_path(self) -> object:
        from ..agents.factory import resolve_checkpoint_path

        name = self.checkpoint_input.value if hasattr(self, "checkpoint_input") else self.app.config.training.checkpoint_name
        return resolve_checkpoint_path(self.app.config, name)

    def _new_trainer(self, load_existing: bool = False) -> None:
        self.trainer = DQNTrainer(self.app.config.rl, seed=0)
        if load_existing:
            path = self._checkpoint_path()
            if hasattr(path, "exists") and path.exists():
                try:
                    payload = self.trainer.load(path)
                    self.checkpoint_note = f"loaded: {describe_checkpoint(payload)}"
                except CheckpointError as exc:
                    self.load_error = str(exc)[:100]
                    self.checkpoint_note = f"could not load: {exc}"
            else:
                self.checkpoint_note = "no checkpoint on disk yet"

    # -- worker management ------------------------------------------------------

    def _training_active(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _start_or_resume(self) -> None:
        self.load_error = ""
        if self._training_active():
            self.control.paused = False
            self.status = "running"
            return
        if self.control.stop:  # finished a previous run: fresh flags
            self.control = TrainingControl()
        self.control.paused = False
        self.control.stop = False
        self.status = "running"
        cfg: AppConfig = self.app.config
        board = (cfg.training.board_width, cfg.training.board_height)

        def train_env_factory():
            return SnakeEnv(width=board[0], height=board[1], max_foods=cfg.game.max_foods,
                            max_steps=cfg.rl.max_episode_steps)

        self.worker_thread = threading.Thread(
            target=training_worker,
            kwargs=dict(
                trainer=self.trainer,
                train_env_factory=train_env_factory,
                eval_env_factory=train_env_factory,
                episodes=cfg.training.episodes,
                control=self.control,
                on_update=self._on_episode,
                on_eval=self._on_eval,
                autosave_path=self._checkpoint_path(),
                autosave_every=cfg.training.autosave_every_episodes,
                eval_every=cfg.training.eval_every_episodes,
                eval_episodes=cfg.training.eval_episodes,
            ),
            daemon=True,
            name="rl-training",
        )
        self.worker_thread.start()

    def _on_episode(self, stats) -> None:
        self.status = "running"

    def _on_eval(self, metrics: dict) -> None:
        self.eval_note = (f"eval@{metrics.get('episode', '?')}: "
                          f"mean score {metrics.get('mean_score', 0):.1f} over {metrics.get('episodes', 0)} eps")

    def _pause(self) -> None:
        if self._training_active():
            self.control.paused = True
            self.status = "paused"

    def _stop(self) -> None:
        self.control.stop = True
        self.control.paused = False
        self.status = "stopping"

    def _save(self) -> None:
        try:
            path = self.trainer.save(self._checkpoint_path())
            self.checkpoint_note = f"saved: {path.name} ({self.trainer.episodes_done} episodes)"
        except Exception as exc:  # noqa: BLE001
            self.checkpoint_note = f"save failed: {exc}"

    def _load(self) -> None:
        try:
            payload = self.trainer.load(self._checkpoint_path())
            self.checkpoint_note = f"loaded: {describe_checkpoint(payload)}"
            self.load_error = ""
        except CheckpointError as exc:
            self.load_error = str(exc)[:90]
            self.checkpoint_note = f"load failed: {exc}"

    def _reset(self) -> None:
        self._stop()
        self.eval_note = ""
        self._new_trainer(load_existing=False)
        self.checkpoint_note = "reset: fresh untrained network"

    def _evaluate(self) -> None:
        if self._eval_running() or self._training_active():
            return
        self.eval_note = "evaluating..."
        cfg = self.app.config

        def work() -> None:
            env_factory = lambda: SnakeEnv(  # noqa: E731
                width=cfg.training.board_width, height=cfg.training.board_height,
                max_foods=cfg.game.max_foods, max_steps=cfg.rl.max_episode_steps,
            )
            metrics = self.trainer.evaluate(env_factory, episodes=cfg.training.eval_episodes)
            metrics["episode"] = self.trainer.episodes_done
            self.trainer.history.add_evaluation(**metrics)
            self.eval_note = (f"greedy eval: mean score {metrics['mean_score']:.1f}, "
                              f"median {metrics['median_score']:.1f}, "
                              f"survival steps {metrics['mean_steps']:.0f}")

        self.eval_thread = threading.Thread(target=work, daemon=True, name="rl-eval")
        self.eval_thread.start()

    def _eval_running(self) -> bool:
        return self.eval_thread is not None and self.eval_thread.is_alive()

    def _back(self) -> None:
        self._stop()
        self.app.pop_screen()

    # -- layout ---------------------------------------------------------------

    def layout(self) -> None:
        w, h = self.app.screen.get_size()
        self.widgets = []
        bx, by = 20, 70
        bw, bh = 220, 44
        rows = [
            ("Start / Resume", self._start_or_resume, "primary"),
            ("Pause", self._pause, "secondary"),
            ("Stop", self._stop, "secondary"),
            ("Save Checkpoint", self._save, "secondary"),
            ("Load Checkpoint", self._load, "secondary"),
            ("Evaluate (greedy)", self._evaluate, "secondary"),
            ("Reset Training", self._reset, "danger"),
            ("Back", self._back, "secondary"),
        ]
        for i, (label, cb, style) in enumerate(rows):
            self.widgets.append(Button(label, cb, pygame.Rect(bx, by + i * (bh + 10), bw, bh), style=style))
        cx = bx + bw + 30
        cy = by
        chart_w = min(400, (w - cx - 40 - 20) // 2)
        chart_h = min(190, (h - cy - 60) // 2)
        self.chart_reward = Chart("Episode reward (moving avg)", (cx, cy), (chart_w, chart_h))
        self.chart_score = Chart("Score (with greedy eval)", (cx + chart_w + 16, cy), (chart_w, chart_h))
        self.chart_loss = Chart("Huber loss", (cx, cy + chart_h + 16), (chart_w, chart_h))
        self.chart_eps = Chart("Exploration (epsilon)", (cx + chart_w + 16, cy + chart_h + 16), (chart_w, chart_h))
        right_x = cx + 2 * chart_w + 44
        self.checkpoint_input = TextInput(
            self.app.config.training.checkpoint_name,
            lambda v: setattr(self.app.config.training, "checkpoint_name", v),
            (right_x, cy), width=200,
        )
        self.widgets.append(self.checkpoint_input)
        self.widgets.append(Label("checkpoint file (models/)", (right_x, cy + 34), font=None, color=COLORS.text_faint))

    # -- update / draw ----------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        surface.fill(rgb(COLORS.bg))
        pygame.draw.rect(surface, rgb(COLORS.panel), (12, 12, w - 24, h - 24), border_radius=14)
        pygame.draw.rect(surface, rgb(COLORS.border), (12, 12, w - 24, h - 24), width=1, border_radius=14)
        text(surface, "RL TRAINING - DQN", (32, 26), font=self.app.fonts.h2, color=COLORS.accent)
        self.layout_charts_if_needed()
        self._refresh_charts()
        status_color = {"running": COLORS.success, "paused": COLORS.gold, "stopping": COLORS.gold}.get(self.status, COLORS.text_dim)
        trained = self.trainer.trained
        badge = (f"{self.trainer.episodes_done} episodes trained" if trained
                 else "UNTRAINED network (random init) - press Start/Resume to train")
        text(surface, f"Status: {self.status}", (260, 30), font=self.app.fonts.h3, color=status_color)
        text(surface, badge, (260, 56), font=self.app.fonts.small,
             color=COLORS.success if trained else COLORS.gold)

        hist = self.trainer.history
        recent = hist.recent(1)
        stats_lines = [
            f"episodes: {self.trainer.episodes_done}",
            f"env steps: {self.trainer.steps}",
            f"epsilon: {self.trainer.epsilon():.3f}",
            f"best score: {hist.best_score if hist.best_score != float('-inf') else '-'}",
            f"last reward: {recent[-1]['reward']:.1f}" if recent else "last reward: -",
            f"last loss: {recent[-1].get('loss') if recent and recent[-1].get('loss') is not None else '-'}",
            f"buffer: {len(self.trainer.buffer)}/{self.trainer.buffer.capacity}",
            f"device: {self.trainer.device}",
        ]
        y = 100
        for line in stats_lines:
            text(surface, line, (32, y), font=self.app.fonts.mono, color=COLORS.text_dim)
            y += 24
        if self.checkpoint_note:
            text(surface, self.checkpoint_note, (32, y + 6), font=self.app.fonts.small, color=COLORS.success)
            y += 22
        if self.load_error:
            text(surface, self.load_error, (32, y + 6), font=self.app.fonts.small, color=COLORS.danger)
            y += 22
        if self.eval_note:
            text(surface, self.eval_note, (32, y + 6), font=self.app.fonts.small, color=COLORS.purple)
        draw_widgets(surface, self.widgets, self.app.fonts)

    def layout_charts_if_needed(self) -> None:
        pass  # charts sized in layout(); cheap enough to call layout on resize only

    def _refresh_charts(self) -> None:
        hist = self.trainer.history
        episodes = hist.episodes
        rewards = _downsample([e["reward"] for e in episodes])
        scores = _downsample([e["score"] for e in episodes])
        losses = _downsample([e["loss"] for e in episodes if e.get("loss") is not None])
        epsilons = _downsample([e["epsilon"] for e in episodes])
        from ..evaluation.experiments import moving_average

        self.chart_reward.set_series(("reward (MA)", moving_average(rewards, max(2, len(rewards) // 40 or 2)), COLORS.accent))
        eval_scores = [e.get("mean_score", 0.0) for e in hist.evaluations]
        self.chart_score.set_series(("score", scores, COLORS.green))
        self.chart_loss.set_series(("loss", losses, COLORS.purple))
        self.chart_eps.set_series(("epsilon", epsilons, COLORS.red))
        surface = self.app.screen
        for chart in (self.chart_reward, self.chart_score, self.chart_loss, self.chart_eps):
            chart.draw(surface, self.app.fonts)
        if eval_scores:
            text(surface, f"greedy evals: {[round(s, 1) for s in eval_scores[-6:]]}",
                 (self.chart_score.rect.x + 10, self.chart_score.rect.bottom + 2),
                 font=self.app.fonts.tiny, color=COLORS.gold)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._back()
            return
        send_event(self.widgets, event)
