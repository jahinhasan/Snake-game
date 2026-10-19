# AI Algorithms

This document describes every agent, the mandatory safety layer, the LLM
planner protocol, and the arbitration that combines them. All of it is
implemented in `src/snake_arena/` — nothing here is aspirational.

All agents consume the same per-tick `StateView`: blocked cells (walls,
bodies, opponent bodies, live traps for planning), food ranked by
`value / (1 + manhattan_distance)`, and own/opponent geometry.

---

## 1. Rule-based agent (`agents/rule_based.py`)

A weighted-scorer over the four actions:

- **food attraction** — negative weighted manhattan distance to the top-ranked
  food, scaled by the food's value and remaining lifetime (expiring food
  becomes more urgent);
- **trap avoidance** — strong negative weight for actions whose next cell is,
  or neighbours, a trap;
- **danger avoidance** — penalise cells with few free neighbours (pre-dead-end
  signal) and proximity to a longer opponent's head;
- **straight bias** — a small constant for continuing straight (smoother,
  fewer self-trapping wiggles).

The best-scoring action wins; the safety validator still has the final say.
Decision cost is O(1) (~0.1 ms measured).

## 2. BFS agent (`agents/bfs.py`)

Breadth-first search from the head over free cells (blocked set from the
view, minus the vacating tail). It routes to the highest-ranked reachable
food and returns the first step of that route. BFS is optimal in *number of
steps* on unweighted grids — the point of including it next to A\* is that on
this grid they coincide, so differences in play come from tie-breaking,
trap treatment, and dead-end avoidance, not path optimality.

## 3. A\* agent (`agents/astar.py`)

A\* with Manhattan heuristic (admissible and consistent on this grid, so
paths are optimal), same blocked-set rules, plus two explicit safety
refinements:

- **tail-aware blocking** — the snake's own tail cell is passable because it
  vacates;
- **dead-end avoidance** — a candidate next cell must have a flood-fill
  reachable area ≥ snake length (bounded flood fill, `safety.flood_fill_count`
  with the configured `flood_fill_limit`); otherwise the move is treated as
  blocked and the search re-plans.

If no food is reachable, it performs a survival walk toward the largest open
region. Cost ~0.7 ms per decision at 24×18 (measured).

## 4. DQN agent (`rl_agent.py` + `rl/`)

- **Network:** MLP 30 → 128 → 128 → 4 (ReLU). Online + target networks,
  target updated every 1000 steps.
- **Observation:** the 30-float vector documented in
  `game/environment.py` (danger one-hot-ish triple, post-move free-area
  ratios, heading, target-food block, second-food block, opponent block,
  length ratio, hunger). Shared verbatim between training and inference.
- **Algorithm:** Double DQN (online net picks, target net evaluates),
  γ = 0.9, Adam lr = 5e-4, Huber loss, batch 64, replay buffer 50 000,
  learn on every step. ε-greedy 1.0 → 0.05 over 25 000 steps.
- **Honesty rules:** an untrained or unloaded network reports
  `trained=False` and the UI/benchmark label it "untrained";
  `describe_checkpoint` reports what a checkpoint actually contains
  ("trained (N episodes, best score X)"); evaluation is greedy and
  deterministic (`evaluate(env_factory, episodes, seed_base)`).
- Checkpoints are atomic (`tmp` file + rename) and contain weights, config
  snapshot, episode count, best score, and timestamp.

See [RL_TRAINING.md](RL_TRAINING.md) for the training recipe and rewards.

## 5. Safety checker (`safety/validator.py`) — mandatory, independent

Every action proposed by *any* agent is checked before it reaches the engine:

| Check | Reason string |
|---|---|
| Next cell outside the board | `wall` |
| 180° turn (relative to the snake's actual facing, derived from geometry) | `reversal` |
| Next cell in own body and not the vacating tail | `self_collision` |
| Next cell in an opponent's body / head cell | `opponent_collision` |
| Flood-fill free area after the move < snake length | `dead_end` |

`select_safe_action(preferred)` implements the final arbitration:

1. preferred action is safe → play it (`intervention=False`);
2. otherwise pick the first safe action in a fixed order
   (straight → left → right → backward-excluded), `intervention=True`;
3. otherwise return the preferred action with
   `reason="no_safe_action"` — the position is lost; we move honestly and
   the record says so.

Interventions are counted per snake and shown in the arena diagnostics and
post-game record. The validator is deliberately policy-free: it knows nothing
about food or strategy, so it can override every planner.

## 6. LLM strategic planner (`llm/`)

**Role.** The LLM never moves the snake. Every few seconds it receives a
compact JSON snapshot and replies with a *posture*:

```json
{
  "strategy": "defensive|balanced|aggressive|greedy|survival",
  "food_priority": 0.0-1.0,
  "survival_priority": 0.0-1.0,
  "risk_tolerance": 0.0-1.0,
  "target_food_id": "<id from snapshot> | null",
  "reason": "one short sentence"
}
```

**Snapshot** (`strategist.build_snapshot`): board size, tick, time left,
own score/length/steps-since-food, the top-6 foods by value-per-distance
(value, risk, distance, remaining lifetime, contested-by-opponent flag),
nearest opponent (score, length, head distance), a coarse danger level, and
the currently-known safe-route flags. Snapshots are hashed (canonical JSON,
SHA-1) so an unchanged situation reuses the cached posture.

**Transport** (`ollama_client.py`): HTTP to a local Ollama server (default
`http://127.0.0.1:11434`, model `qwen3:4b`), JSON-mode requests
(`format:"json"`), `<think>…</think>` stripping, connect + read timeouts,
retry on transport failure *or* malformed JSON (≤ `max_retries`), then
fallback. The transport function is injectable, which is how the test suite
exercises the full pipeline offline with a programmable fake.

**Validation** (`schemas.py`): the reply is strictly parsed — enum-checked
strategy, floats clamped to [0, 1], `target_food_id` must exist in the
snapshot or is dropped — then converted to a `StrategyHints` dataclass.

**Fallback** (`fallback_strategy`): a small deterministic policy
(balanced posture; food_priority and survival_priority from measured score
gap and length ratio; risk_tolerance low when free area is scarce). With
Ollama absent, this makes LLM-mode agents fully deterministic — which is why
the test suite can assert exact behaviour and benchmarks can flag
`llm_available: false` honestly.

**Guarantees:** a slow/hung LLM can never stall the game (background worker +
cached slot + timeouts); a lying LLM cannot kill the snake (safety validator
overrides); a missing LLM changes behaviour only in documented ways.

## 7. Hybrid decision manager (`decision_manager.py`, mode `hybrid`)

Votes from A\* (reliable, myopic) and DQN (learned, long-horizon-ish) are
combined with hint-modulated weights:

- `w_astar = 0.6 + 0.2·(1 − risk_tolerance)`, `w_dqn = 1 − w_astar`;
- `greedy` posture raises the food-value weight, `survival`/`defensive`
  raise the dead-end margin;
- RL risk gating shrinks the acceptable free area:
  `min_area = length·(0.8 + 0.9·survival_priority)·(1 − 0.5·risk_tolerance)`
  (×0.7 when `aggressive`);
- ties break toward A\*.

The same `DecisionManager` drives every mode (`rule_based`, `astar`, `bfs`,
`rl`, `llm_astar`, `llm_rl`, `hybrid`) through the identical
view → safety → hints → policy → arbitration pipeline, so mode comparisons in
benchmarks isolate the planner, not the plumbing.

## 8. Post-game analysis (`llm/analysis.py`)

After a game, if the LLM is enabled and reachable, the manager's stats and
the game record are summarised into a prompt; the model's narrative is stored
in the record under `ai_summary` **marked as model-generated**. Without an
LLM, a deterministic summary (decisions, interventions, food breakdown,
outcome) is produced instead. Both paths are labelled; neither is presented
as ground truth.
