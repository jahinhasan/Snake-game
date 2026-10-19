"""DQN trainer: epsilon-greedy collection, replay, target network, eval.

Default algorithm: Deep Q-Network (Mnih et al. 2015) with optional Double-DQN
targets (van Hasselt et al. 2016). Everything is CPU-first; a CUDA/MPS device
is used only when present. Loss is Huber (smooth L1) for stability.

The same class serves CLI training (scripts/train.py), in-app background
training (UI worker thread), and greedy evaluation of checkpoints.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..config import RLConfig
from ..game.environment import OBS_SIZE, SnakeEnv
from .checkpoints import TrainingHistory, build_checkpoint, save_checkpoint
from .network import QNetwork, pick_device
from .replay_buffer import ReplayBuffer
from .rewards import RewardCalculator

log = logging.getLogger("snake_arena.rl.trainer")


@dataclass
class EpisodeStats:
    episode: int
    reward: float
    score: int
    length: int
    steps: int
    epsilon: float
    loss: float | None
    terminated: bool
    truncated: bool
    death_reason: str | None
    seconds: float


class DQNTrainer:
    def __init__(
        self,
        rl_cfg: RLConfig,
        obs_size: int = OBS_SIZE,
        n_actions: int = 4,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> None:
        self.cfg = rl_cfg
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.device = device or pick_device()
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.online = QNetwork(obs_size, rl_cfg.hidden_sizes, n_actions).to(self.device)
        self.target = QNetwork(obs_size, rl_cfg.hidden_sizes, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.buffer = ReplayBuffer(rl_cfg.buffer_size, obs_size, self.rng)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=rl_cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.steps = 0
        self.episodes_done = 0
        self.history = TrainingHistory()
        self.recent_losses: deque[float] = deque(maxlen=100)

    # -- exploration ---------------------------------------------------------

    def epsilon(self) -> float:
        cfg = self.cfg
        if cfg.eps_decay_steps <= 0:
            return cfg.eps_end
        frac = min(1.0, self.steps / cfg.eps_decay_steps)
        return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(0, self.n_actions))
        with torch.no_grad():
            tensor = torch.from_numpy(obs.astype(np.float32)).to(self.device)
            return self.online.act_greedy(tensor)

    # -- learning ------------------------------------------------------------

    def learn(self) -> float | None:
        """One gradient step; returns loss or None if the buffer is too small."""
        cfg = self.cfg
        if not self.buffer.can_sample(cfg.batch_size):
            return None
        obs, actions, rewards, next_obs, dones = self.buffer.sample(cfg.batch_size)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)

        q_sa = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if cfg.double_dqn:
                best_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
                q_next = self.target(next_obs).gather(1, best_actions).squeeze(1)
            else:
                q_next = self.target(next_obs).max(dim=1).values
            q_target = rewards + cfg.gamma * (1.0 - dones) * q_next

        loss = self.loss_fn(q_sa, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.recent_losses.append(float(loss.item()))

        self.steps += 1
        if self.steps % cfg.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    # -- episodes ------------------------------------------------------------

    def run_episode(
        self,
        env: SnakeEnv,
        greedy: bool = False,
        seed: int | None = None,
        reward_calc: RewardCalculator | None = None,
        learn: bool = True,
    ) -> EpisodeStats:
        """Run one full episode; optionally learns from every step."""
        reward_calc = reward_calc or RewardCalculator(self.cfg.reward, env.state.width, env.state.height)
        obs = env.reset(seed)
        snake = env.state.snakes[env.snake_id]
        total = 0.0
        last_loss: float | None = None
        t0 = time.perf_counter()
        terminated = truncated = False
        death_reason: str | None = None

        while True:
            prev_view = env._last_view
            prev_dist = RewardCalculator.target_distance(prev_view) if prev_view else None
            action = self.select_action(obs, greedy=greedy)
            next_obs, events, terminated, truncated, info = env.step(action)
            new_view = info["view"]
            new_dist = RewardCalculator.target_distance(new_view)
            step_reward = reward_calc.compute(
                events,
                prev_dist,
                new_dist,
                snake.death_reason,
                snake.steps_since_food,
                terminated,
            )
            if death_reason is None and terminated:
                death_reason = snake.death_reason
            total += step_reward.total

            if learn:
                self.buffer.push(obs, action, step_reward.total, next_obs, terminated)
                if self.steps % self.cfg.train_every_steps == 0:
                    last_loss = self.learn()

            obs = next_obs
            if terminated or truncated:
                break

        self.episodes_done += 1
        stats = EpisodeStats(
            episode=self.episodes_done,
            reward=total,
            score=info["score"],
            length=info["length"],
            steps=info["steps"],
            epsilon=self.epsilon(),
            loss=last_loss,
            terminated=terminated,
            truncated=truncated,
            death_reason=death_reason,
            seconds=time.perf_counter() - t0,
        )
        if learn:
            self.history.total_steps = self.steps
            self.history.add_episode(
                reward=total,
                score=stats.score,
                length=stats.length,
                steps=stats.steps,
                epsilon=stats.epsilon,
                loss=stats.loss,
                terminated=terminated,
            )
        return stats

    def evaluate(
        self,
        env_factory,
        episodes: int = 20,
        seed_base: int = 900_000,
    ) -> dict:
        """Greedy, deterministic evaluation on fresh environments."""
        scores: list[float] = []
        rewards: list[float] = []
        lengths: list[int] = []
        deaths: dict[str, int] = {}
        for i in range(episodes):
            env = env_factory()
            stats = self.run_episode(env, greedy=True, seed=seed_base + i, learn=False)
            scores.append(stats.score)
            rewards.append(stats.reward)
            lengths.append(stats.steps)
            key = stats.death_reason or ("timeout" if stats.truncated else "survived")
            deaths[key] = deaths.get(key, 0) + 1

        def _mean(xs):
            return sum(xs) / len(xs) if xs else 0.0
        return {
            "episodes": episodes,
            "mean_score": _mean(scores),
            "median_score": float(np.median(scores)) if scores else 0.0,
            "mean_reward": _mean(rewards),
            "mean_steps": _mean(lengths),
            "deaths": deaths,
            "timestamp": time.time(),
        }

    # -- persistence ----------------------------------------------------------

    def hyperparams(self) -> dict:
        return {
            "gamma": self.cfg.gamma,
            "learning_rate": self.cfg.learning_rate,
            "buffer_size": self.cfg.buffer_size,
            "batch_size": self.cfg.batch_size,
            "target_update_steps": self.cfg.target_update_steps,
            "train_every_steps": self.cfg.train_every_steps,
            "eps_start": self.cfg.eps_start,
            "eps_end": self.cfg.eps_end,
            "eps_decay_steps": self.cfg.eps_decay_steps,
            "double_dqn": self.cfg.double_dqn,
            "hidden_sizes": list(self.cfg.hidden_sizes),
            "reward": vars(self.cfg.reward),
        }

    @property
    def trained(self) -> bool:
        return self.episodes_done > 0

    def save(self, path) -> dict:
        payload = build_checkpoint(
            online=self.online,
            target=self.target,
            optimizer=self.optimizer,
            obs_size=self.obs_size,
            hidden_sizes=list(self.cfg.hidden_sizes),
            n_actions=self.n_actions,
            hyperparams=self.hyperparams(),
            history=self.history,
            epsilon=self.epsilon(),
            steps=self.steps,
            episodes=self.episodes_done,
            trained=self.trained,
        )
        return save_checkpoint(path, payload)

    def load(self, path) -> dict:
        from .checkpoints import load_checkpoint

        payload = load_checkpoint(path, expected_obs_size=self.obs_size)
        self.online.load_state_dict(payload["online_state"])
        self.target.load_state_dict(payload["target_state"])
        try:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        except Exception as exc:  # noqa: BLE001 - optimizer mismatch is non-fatal
            log.warning("Optimizer state not restored (%s); continuing with fresh optimizer.", exc)
        self.history = TrainingHistory.from_dict(payload.get("history"))
        self.steps = int(payload.get("steps", 0))
        self.episodes_done = int(payload.get("episodes", 0))
        self.history.total_steps = self.steps
        return payload


@dataclass
class TrainingControl:
    """Thread-safe flags shared between the UI and the training worker."""

    stop: bool = False
    paused: bool = False


def training_worker(
    trainer: DQNTrainer,
    train_env_factory,
    eval_env_factory,
    episodes: int,
    control: TrainingControl,
    on_update,
    on_eval=None,
    autosave_path=None,
    autosave_every: int = 50,
    eval_every: int = 50,
    eval_episodes: int = 20,
) -> None:
    """Background-thread loop used by the RL training dashboard.

    ``on_update(stats)`` fires after every episode; ``on_eval(metrics)`` after
    each evaluation block. Both run on the worker thread -- UI consumers must
    copy under their own lock. Honors pause/stop flags every episode.
    """
    eval_env = eval_env_factory()
    for ep in range(1, episodes + 1):
        if control.stop:
            break
        while control.paused and not control.stop:
            time.sleep(0.1)
        if control.stop:
            break
        env = train_env_factory()
        stats = trainer.run_episode(env)
        trainer.history.total_steps = trainer.steps
        try:
            on_update(stats)
        except Exception:  # noqa: BLE001 - UI callback must not kill training
            log.exception("Training UI callback failed")
        if autosave_path is not None and ep % max(1, autosave_every) == 0:
            try:
                trainer.save(autosave_path)
            except Exception:  # noqa: BLE001
                log.exception("Autosave failed")
        if ep % max(1, eval_every) == 0:
            metrics = trainer.evaluate(
                lambda: eval_env, episodes=eval_episodes, seed_base=900_000 + trainer.episodes_done
            )
            metrics["episode"] = trainer.episodes_done
            trainer.history.add_evaluation(**metrics)
            if on_eval is not None:
                try:
                    on_eval(metrics)
                except Exception:  # noqa: BLE001
                    log.exception("Eval callback failed")
    if autosave_path is not None and not control.stop:
        try:
            trainer.save(autosave_path)
        except Exception:  # noqa: BLE001
            log.exception("Final save failed")
