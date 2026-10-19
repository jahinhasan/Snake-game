# Architecture

## Overview

The project is a layered simulation: a deterministic game engine at the core,
agents that observe the engine through a cheap precomputed view, a mandatory
safety layer between any agent and the engine, and UI/evaluation layers that
consume the same building blocks. All AI reasoning is synchronous per tick and
budgeted; the only asynchronous components are the LLM strategist worker and
UI benchmark/training threads.

```
┌────────────────────────────── UI (pygame) ──────────────────────────────┐
│  menu → mode select → arena ⇄ results      training / benchmark /       │
│                                            settings / statistics        │
└──────────────┬───────────────────────────────────────────┬──────────────┘
               │ actions                                   │ threads
┌──────────────▼───────────────────────────────┐   ┌──────▼───────┐
│ DecisionManager (one per AI snake)           │   │ StrategyWorker│
│  1. StateView (precomputed per tick)         │◄──┤ (LLM, bg thd)│
│  2. SafetyValidator.evaluate_all             │   └──────┬───────┘
│  3. hints: StrategyWorker | client | fallback│          │ HTTP
│  4. policy vote (rule/BFS/A*/DQN/hybrid)     │   ┌──────▼───────┐
│  5. select_safe_action (arbitration)         │   │ OllamaClient │
│  6. DecisionLog (per-tick diagnostics)       │   └──────────────┘
└──────────────┬───────────────────────────────┘
┌──────────────▼───────────────────────────────┐
│ GameEngine: deterministic tick, collisions,  │
│ food system (5 types), scores, win/lose      │
└──────────────────────────────────────────────┘
        ▲                                    ▲
   SnakeEnv (RL)                    BenchmarkRunner / history
   (30-float obs, step())           (identical seeds, artifacts)
```

## Modules

| Package | Responsibility |
|---|---|
| `snake_arena.game` | Engine and world model. `GameEngine.step({snake_id: Action})` advances one tick: applies safe moves, resolves collisions (tail-aware; head-on = both die = draw), ages food, maintains `max_foods`. `Snake.commit_move` grows gradually (1 segment per tick via `pending_growth`). `SnakeEnv` wraps the engine as a Gym-style single-snake RL environment with the documented 30-float observation. |
| `snake_arena.game.collision` | `is_deadly_for` (returns `None` for the vacating tail cell), `flood_fill_count`, `planning_blocked_cells`. |
| `snake_arena.agents` | Policies (`rule_based`, `bfs`, `astar`, `rl_agent`), the `DecisionManager` pipeline, and the `factory` (construction policy: checkpoint resolution, LLM wiring, manager building). |
| `snake_arena.safety` | `SafetyValidator` — independent of every policy. Checks wall, reversal, self-collision, opponent-collision, dead-end (flood-fill area vs. snake length). `evaluate_all` reports all four actions; `select_safe_action` enforces the final say. |
| `snake_arena.llm` | `ollama_client` (HTTP with injectable transport for tests, JSON mode, think-tag stripping, retries), `strategist` (snapshot builder, prompt, background `StrategyWorker` with caching + rate limiting, deterministic `fallback_strategy`), `schemas` (strict validation), `analysis` (post-game narrative). |
| `snake_arena.rl` | `network` (DQN MLP + target net), `replay` (ring buffer), `rewards` (`RewardCalculator`), `trainer` (`DQNTrainer`, `TrainingControl` for cancellation, `training_worker`), `checkpoints` (atomic save/load, `describe_checkpoint`, `CheckpointError`). |
| `snake_arena.evaluation` | `BenchmarkRunner` (solo runs with identical per-index seeds; head-to-head matches; round-robin tournaments), `metrics` (per-episode records, `summarize`), `experiments` (run dirs, JSON/CSV persistence, matplotlib charts, run loading). |
| `snake_arena.persistence` | Atomic JSON/JSONL/CSV storage with `.corrupt` quarantine, game history (`record_game`, `load_recent`, `aggregate`). |
| `snake_arena.config` | `AppConfig` with 12 validated sections, clamping, tolerant partial merge, atomic save, corrupt-recovery, env overrides (`SNAKE_ARENA_DISABLE_LLM`, `OLLAMA_HOST`, `OLLAMA_MODEL`). |
| `snake_arena.ui` | `App` (screen stack, caches for LLM client/worker/RL agent), `Screen` base, widget kit (`Button`, `Slider`, `Toggle`, `Dropdown`, `TextInput`, `ProgressBar`, `DataTable`, `BarChart`), theme, and the 7 screens. |

## The decision pipeline (per AI snake, per tick)

1. **View.** `StateView` precomputes blocked cells and ranked food priorities
   once per tick; every consumer reads the same view (consistency + speed).
2. **Safety pre-computation.** `SafetyValidator.evaluate_all` produces a
   `SafetyReport` per action (safe?, reason, free-area, intervention?).
3. **Strategy hints (LLM modes only).** The `DecisionManager` asks, in order:
   the background `StrategyWorker` (cached snapshot hash → fresh call at most
   every `min_interval_seconds`), a synchronous throttled client (headless
   runs), or the deterministic `fallback_strategy`. Hints are *postures* —
   they never move the snake directly.
4. **Policy vote.** The mode's policy proposes an action:
   `rule_based` heuristics, BFS route, A\* route, DQN argmax, or hybrid
   (A\* + DQN votes combined with hint-modulated weights).
5. **Arbitration.** `select_safe_action(preferred)` returns the preferred
   action if safe, otherwise the best safe alternative in a documented
   fallback order, otherwise the preferred action flagged
   `no_safe_action` (the snake is dead either way — reported honestly).
6. **Diagnostics.** Every decision appends a `DecisionLog` entry: proposed vs
   final action, safety reasons, strategy source (`llm`/`fallback`), Q-values,
   chosen target. The arena screen renders these live; records persist
   aggregates (`decisions`, `safety_interventions`, `llm_calls`,
   `llm_failures`, mean latency, `gating_activations`).

## Hybrid weights (documented, deterministic)

`hybrid` combines A\* and DQN votes: `w_astar = 0.6 + 0.2·(1 − risk_tolerance)
`, `w_dqn = 1 − w_astar`, both modulated by the hint posture (`greedy`
raises food weight, `survival`/`defensive` raise safety margins; the RL risk
gate shrinks the minimum free area a move must leave:
`min_area = length · (0.8 + 0.9·survival_priority) · (1 − 0.5·risk_tolerance)`
, ×0.7 for `aggressive`). Ties break toward A\*. This is deliberately simple,
fully inspectable, and identical across runs.

## Threading model

- **Main thread:** pygame loop (input, update at the configured moves/s,
  draw at up to 60 fps). Never touches the network or HTTP.
- **StrategyWorker thread:** one daemon thread per LLM-enabled manager; owns
  its `OllamaClient`; the manager polls a result slot and falls back when the
  slot is empty. Shutdown is cooperative (`stop()`).
- **Benchmark/training threads (UI dashboards):** daemon threads run the
  headless runner/trainer with a `progress_cb` and a cancellation flag; the UI
  stays responsive and can cancel.

## Determinism

- Engine ticks are pure functions of (state, actions, RNG draw).
- Benchmarks pin per-index seeds shared by all agents, so cross-agent
  comparisons vary only by policy.
- With Ollama absent, LLM-mode decisions are fully deterministic (fallback
  strategy), which the test suite relies on.

## Failure containment

| Failure | Behaviour |
|---|---|
| Ollama down/slow/unknown model | Fallback strategy, `strategy_source=fallback`, failures counted |
| Malformed LLM JSON | Retry (≤ `max_retries`), then fallback; never crashes |
| Corrupt checkpoint | Logged, untrained network + safety validator (honest) |
| Corrupt config/history file | Quarantined as `.corrupt`, defaults / empty history |
| RL episode timeout | `truncated=True`, no fake win/loss credit |
