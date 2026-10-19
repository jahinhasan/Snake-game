"""RL components: network, replay buffer, rewards, trainer, checkpoints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snake_arena.config import RLConfig, RLRewardConfig
from snake_arena.game.engine import GameEvent
from snake_arena.rl.network import QNetwork, pick_device
from snake_arena.rl.replay_buffer import ReplayBuffer
from snake_arena.rl.rewards import RewardCalculator
from snake_arena.rl.trainer import DQNTrainer

from .conftest import make_env


@pytest.fixture
def tiny_rl_config() -> RLConfig:
    cfg = RLConfig()
    cfg.hidden_sizes = [32, 32]
    cfg.buffer_size = 2000
    cfg.batch_size = 16
    cfg.target_update_steps = 50
    cfg.eps_decay_steps = 500
    cfg.max_episode_steps = 120
    return cfg


def test_network_shapes():
    net = QNetwork(30, [64, 64], 4)
    out = net(torch.zeros(5, 30))
    assert out.shape == (5, 4)
    idx = net.act_greedy(torch.zeros(30))
    assert isinstance(idx, int) and 0 <= idx < 4
    values = net.q_values(torch.zeros(30))
    assert len(values) == 4 and all(isinstance(v, float) for v in values)
    assert pick_device().type in ("cpu", "cuda", "mps")


def test_replay_buffer_ring_and_sampling():
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=10, obs_size=4, rng=rng)
    assert not buffer.can_sample(8)
    for i in range(25):
        buffer.push(np.full(4, i, dtype=np.float32), i % 4, float(i),
                    np.full(4, i + 1, dtype=np.float32), False)
    assert len(buffer) == 10  # ring: newest kept
    assert buffer.can_sample(8)
    obs, actions, rewards, next_obs, dones = buffer.sample(8)
    assert obs.shape == (8, 4)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert dones.shape == (8,)
    assert float(dones.max()) <= 1.0 and float(dones.min()) >= 0.0
    # Ring overwrote the oldest pushes: only pushes 15..24 remain.
    assert int(obs.min()) >= 15


def _events(kind: str, value: int = 0, reason: str | None = None) -> list:
    return [GameEvent(kind=kind, snake_id="ai", value=value, reason=reason)]


def test_reward_food_trap_death_shaping():
    calc = RewardCalculator(RLRewardConfig(), 24, 18)
    eat = calc.compute(_events("eat", value=50), 10.0, 9.0, None, 0, False)
    assert eat.total > 0.5  # 0.1*50 + shaping
    assert eat.components["food"] == pytest.approx(5.0)

    trap = calc.compute(_events("trap", value=-20), 5.0, 5.0, None, 0, False)
    assert trap.total == pytest.approx(RLRewardConfig().trap_penalty)

    death = calc.compute(_events("death", reason="wall"), None, None, "wall", 0, True)
    assert death.total == pytest.approx(RLRewardConfig().death_wall)

    closer = calc.compute([], 10.0, 9.0, None, 0, False)
    farther = calc.compute([], 9.0, 10.0, None, 0, False)
    assert closer.components["progress"] > 0 > farther.components["progress"]

    stagnating = calc.compute([], None, None, None, steps_since_food=200, terminated=False)
    assert stagnating.components["stagnation"] < 0


def test_trainer_runs_episodes_and_records_history(tiny_rl_config):
    trainer = DQNTrainer(tiny_rl_config, obs_size=30, seed=0)
    env = make_env(seed=5)
    stats = trainer.run_episode(env, seed=101)
    assert trainer.episodes_done == 1
    assert stats.steps > 0
    assert len(trainer.history.episodes) == 1
    entry = trainer.history.episodes[0]
    assert entry["score"] == stats.score
    assert 0.0 <= entry["epsilon"] <= 1.0


def test_trainer_collects_and_learns(tiny_rl_config):
    trainer = DQNTrainer(tiny_rl_config, obs_size=30, seed=1)
    env = make_env(seed=6)
    for _ in range(6):
        trainer.run_episode(env)
    assert trainer.buffer.can_sample(tiny_rl_config.batch_size)
    assert len(trainer.recent_losses) > 0
    assert all(loss >= 0.0 for loss in trainer.recent_losses)


def test_greedy_evaluation_deterministic(tiny_rl_config):
    trainer = DQNTrainer(tiny_rl_config, obs_size=30, seed=2)
    trainer.episodes_done = 5  # pretend trained so history records nothing odd
    factory = lambda: make_env(seed=8)  # noqa: E731
    m1 = trainer.evaluate(factory, episodes=3, seed_base=500)
    m2 = trainer.evaluate(factory, episodes=3, seed_base=500)
    assert m1["episodes"] == 3
    assert m1["mean_score"] == pytest.approx(m2["mean_score"])


def test_checkpoint_roundtrip_preserves_policy(tmp_path, tiny_rl_config):
    trainer = DQNTrainer(tiny_rl_config, obs_size=30, seed=3)
    env = make_env(seed=9)
    for _ in range(3):
        trainer.run_episode(env)
    path = tmp_path / "dqn_test.pt"
    trainer.save(path)

    clone = DQNTrainer(tiny_rl_config, obs_size=30, seed=99)
    payload = clone.load(path)
    assert clone.episodes_done == trainer.episodes_done
    assert payload["trained"] is True
    assert payload["episodes"] == 3

    obs = np.random.default_rng(7).standard_normal(30).astype(np.float32)
    t = torch.from_numpy(obs)
    assert trainer.online.act_greedy(t) == clone.online.act_greedy(t)


def test_bad_checkpoint_raises(tmp_path):
    from snake_arena.rl.checkpoints import CheckpointError, load_checkpoint

    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError):
        load_checkpoint(bad, expected_obs_size=30)
