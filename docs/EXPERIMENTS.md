# Experiments & Evaluation

Methodology, metrics, artifacts, and a fully reproducible example with the
**actual measured numbers** from this development machine (Arch Linux, CPU,
Ollama absent, 2026-10-19). Nothing in this document is invented; every
number can be regenerated with the given commands.

---

## Methodology

### Seed control (the core of a fair comparison)

For solo benchmarks the runner assigns each agent the **identical seed list**
(`base_seed + 0..N-1`, default base 1000). Episode k for agent A and episode k
for agent B therefore start from the same board, same food set, same initial
snakes. Any score difference is attributable to the agent's policy — not to
luck. (A* and rule-based diverge immediately because their first moves
differ; that is the point of the comparison.)

### Agents under test

`rule_based`, `astar`, `bfs`, `rl` (checkpoint or honest "untrained"),
`llm_astar`, `llm_rl` (LLM posture + base policy), `hybrid` (A\*+DQN with LLM
posture). LLM modes also record `llm_calls`, `llm_failures`, latency, and a
run-level `llm_available` flag (`true` only if calls happened and none
failed). Without Ollama, LLM agents run the deterministic fallback and the
flag is `false` — results are labelled accordingly, never quietly mixed.

### Episode end conditions

Death (wall/self/opponent), time limit (300 s default), or
`max_ticks_per_episode` (1 500 default). Survival rate counts episodes that
ended by *timeout* (i.e., the agent never died).

### Head-to-head & tournament

`run_match(agent_a, agent_b, seed)` plays a duel on a shared board and
reports scores, ticks, outcome (`survivor_win`/`timeout`/`draw`/`ongoing`),
winner, and death reasons. The tournament runner plays a full round-robin
over selected agents and stores wins/win-rates per pairing (draws included).

## Metrics (`evaluation/metrics.py`)

Per episode record: agent, seed, score, steps, length, outcome, death reason,
per-type food counts, and (LLM modes) decision stats.

Summary per agent: `episodes, mean_score, median_score, std, min, max,
survival_rate, mean_steps, traps_per_1k_ticks, decision_ms, death_reasons`.

- `traps_per_1k_ticks` — normalises trap hits by exposure time.
- `decision_ms` — mean wall-clock decision time per tick (policy cost).

## Artifacts (per run, `data/benchmarks/<source>-<timestamp>-<id>/`)

| File | Content |
|---|---|
| `results.json` | full records + per-agent summaries + run metadata |
| `summary.csv` | one row per agent (the table the UI shows) |
| `records.csv` | one row per episode |
| `chart_scores.png` | score distribution / mean chart |
| `chart_survival.png` | survival + death reasons |
| `chart_foods.png` | per-type food consumption |

`load_run(dir)` reads `results.json` back and the UI displays previous runs;
a round-trip is byte-identical in the summary table (tested).

## Reproducing

```bash
# Solo benchmark, all agents, config defaults (20 episodes/agent):
.venv/bin/python main.py --benchmark 20

# Fast pair:
.venv/bin/python main.py --benchmark 5 --agents rule_based,astar

# Tournament (writes tournament section into results.json):
.venv/bin/python main.py --benchmark 5 --agents rule_based,astar,bfs --tournament
```

(For tournaments use `scripts/benchmark.py` directly if you want the
headless tournament entry point: `.venv/bin/python -m scripts.benchmark
--tournament …` — `main.py --benchmark` covers solo runs.)

## Measured example (real run, 2026-10-19)

Command:

```bash
.venv/bin/python main.py --benchmark 2 --agents rule_based,astar
```

Output directory: `data/benchmarks/cli-20261019-022020-abb6/` — kept in the
repository working tree as a genuine artifact.

| agent | eps | mean | median | std | survival | traps/1k | ms/tick |
|---|---|---|---|---|---|---|---|
| Rule-Based | 2 | 2562.5 | 2562.5 | 342.5 | 0% | 1.45 | 0.10 |
| A* | 2 | 1825.0 | 1825.0 | 255.0 | 0% | 2.52 | 0.70 |

Per-episode detail (from `results.json`):

| agent | seed | score | steps | length | ending |
|---|---|---|---|---|---|
| Rule-Based | 1000 | 2220 | 1479 | 105 | wall |
| Rule-Based | 1001 | 2905 | 1286 | 114 | wall |
| A* | 1000 | 1570 | 918 | 84 | self |
| A* | 1001 | 2080 | 672 | 80 | self |

`llm_available: false` (Ollama not installed on this machine).

**Reading, with the caveat that N=2:** rule-based survived longer and scored
more because it plays conservatively (danger weights) while A\* plays fast
and dies to self-trapping around food clusters — a known A\*-with-dead-end-
check failure mode on crowded boards. Two episodes cannot support a strong
claim; they demonstrate that the pipeline produces *real, plausible,
differing* results. Run `--benchmark 20`+ before drawing conclusions, and
expect the benchmark dashboard's charts to show it.

### RL reference (same day)

40-episode functional training run (see
[RL_TRAINING.md](RL_TRAINING.md)): greedy eval mean 83.75 over 20 episodes,
17/20 wall endings. The shipped checkpoint is **not** a tuned policy and the
benchmark labels it as such. No RL improvement is claimed anywhere in this
project's artifacts — that claim would require the long training run the
tooling is ready to perform.

## Threats to validity (stated honestly)

- **N is small** in the committed examples; statistical strength needs
  dozens of episodes per agent.
- **LLM untested live:** all LLM-mode numbers here come from the fallback
  path; with a live Ollama the LLM modes' behaviour will differ.
- **RL checkpoint is undertrained**; benchmarks including `rl` reflect that.
- **Identical seeds couple episodes across agents** (by design) but do not
  remove within-agent variance across k.
- Decision-time numbers are wall-clock on one machine; treat `decision_ms`
  as order-of-magnitude only.
