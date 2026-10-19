# RL Training Guide

How the DQN is trained, what the numbers mean, and how to reproduce every
figure in this project. Everything here runs on CPU.

---

## Setup

| Component | Value (config: `rl`, `training` sections) |
|---|---|
| Algorithm | Double DQN (online net selects, target net evaluates) |
| Network | MLP 30 → 128 → 128 → 4, ReLU |
| γ | 0.9 |
| Optimizer | Adam, lr 5e-4 |
| Loss | Huber (smooth L1) |
| Replay buffer | 50 000 transitions, batch 64, learn every step |
| Target network sync | every 1 000 steps |
| ε-greedy | 1.0 → 0.05 linearly over 25 000 steps |
| Episode cap | 1 200 steps (`max_episode_steps`) |
| Training board | 20×15 (`training.board_width/height`), 4 foods |

## Observation (30 floats — `game/environment.py`)

Documented verbatim in the module docstring; summarised:

1. **Danger** for straight/left/right (wall, body, or opponent) — 3
2. **Post-move free area** per direction, `min(1, area/length)` (dead-end
   signal) — 3
3. **Heading** one-hot — 4
4. **Target food** block: direction signs, normalised Manhattan, |value|/100,
   is-trap — 7
5. **Second-ranked food** direction signs — 4
6. **Opponent** block: presence, head signs, distance, longer-than-us — 7
7. **Length ratio** (capped at 1) and **hunger** (`steps_since_food/150`) — 2

The target food is the best non-trap item by `value/(1+manhattan)`; if only
traps remain, the nearest item of any kind is used so direction features stay
sensible.

## Rewards (`rl/rewards.py`, config `rl.reward`)

| Event | Reward |
|---|---|
| Eating food | `0.1 × food value` (normal +1.0 … special +10.0) |
| Trap | −2.0 |
| Death (wall / self / opponent) | −10.0 |
| Progress shaping | `0.12 × Δ(manhattan to target food)` per step |
| Stagnation | −0.02/step after 120 steps without eating |
| Win (opponent eliminated first in duels) | +5.0 |
| Step cost | 0.0 (shaping carries the opportunity cost) |

Design intent: shaping rewards dense progress, the large death penalty makes
the safety-relevant behaviour learnable, and food rewards are scaled so
hunting a special (+10.0) genuinely outweighs six normal foods (+6.0).

## Training loop (`rl/trainer.py`)

`DQNTrainer.run_episode(env, greedy, seed, reward_calc, learn)`:
reset → act (ε-greedy or greedy) → step → store → sample & learn →
target-sync bookkeeping → terminate on death or truncation. `training_worker`
wraps N episodes with callbacks (`on_episode`, `on_eval`), periodic
evaluation (`eval_every_episodes`, greedy, `eval_episodes` episodes), and
autosave (`autosave_every_episodes`), all cancellable through
`TrainingControl`.

## Running training

```bash
.venv/bin/python main.py --train 400          # config defaults
.venv/bin/python main.py --train 2000         # serious run (hours on CPU)
```

Each run prints per-episode progress, performs greedy evaluations, saves the
checkpoint atomically to `models/<checkpoint_name>` (default
`models/dqn_default.pt`), and writes `data/training/training_curves.png`
(score / loss / ε / eval curves). Training **resumes** from an existing
checkpoint; delete it to start fresh.

Evaluation is deterministic: fixed seeds, pure greedy policy
(`evaluate(env_factory, episodes, seed_base)`), so re-running evaluation on
the same checkpoint reproduces the same numbers exactly.

## What a checkpoint honestly reports

`describe_checkpoint(payload)` returns e.g.
`"trained (440 episodes, best score 190)"` or
`"untrained (random initialization)"`. The factory logs this at load; the
arena and benchmark dashboards display it. A checkpoint from this project
will never claim more training than it contains.

## Measured reference run (2026-10-19, this machine)

A **40-episode** functional run (`python main.py --train 40`, Arch Linux,
CPU, torch 2.14.1+cpu):

- final ε ≈ 0.950 (ε-decay spans 25 000 steps; 40 episodes barely dent it —
  as expected)
- training scores: mostly 0–10, one 125 episode
- greedy evaluation, 20 episodes: mean score **83.75**, median 90.0, mean
  40 steps, endings `{wall: 17, self: 3}`

Interpretation (stated plainly): **40 episodes is a functional smoke test,
not training.** The shipped checkpoint is expected to lose to A\* — which is
exactly what the benchmark shows. The honest claim this project makes about
RL is about *infrastructure correctness* (deterministic eval, resumable
training, truthful labels), not about a strong trained policy; producing the
latter is a `--train 5000` away and the tooling will report whatever actually
happens.

## Tuning notes

- **Slow improvement:** raise `eps_decay_steps` if the agent becomes greedy
  too early; lower `progress_scale` if it orbits food without eating.
- **Suicidal after loading a short run:** the safety validator already
  prevents immediate fatalities; train longer rather than increasing death
  penalties.
- **Board transfer:** checkpoints are board-agnostic (the observation is
  relative), but behaviour trained on 20×15 degrades on 40×30; fine-tune
  instead of expecting zero-shot transfer — and measure before claiming.
