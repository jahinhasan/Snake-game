# Project Report — LLM-Enhanced Intelligent Snake Arena

*Academic write-up. All results quoted are real measurements from the
development run (Arch Linux, CPU-only, 2026-10-19); no result, score, or
citation in this report is invented.*

---

## Abstract

This project builds and evaluates a competitive Snake arena in which
classical search agents, a deep reinforcement learning agent, and a locally
hosted large-language-model (LLM) planner coexist under a single,
mandatory safety layer. The system contributes: (1) a deterministic
simultaneous-turn engine with five food types whose values, lifetimes and
risks create explicit exploration/exploitation trade-offs; (2) a decision
pipeline in which every AI action — from any planner — is independently
validated against wall, reversal, self, opponent, and dead-end hazards
before execution; (3) an LLM integration that issues strategic *postures*
rather than moves, is cached, rate-limited, and thread-isolated so it can
never stall or endanger the game, and degrades to a documented deterministic
fallback when the model server is absent; (4) a Double DQN (30→128→128→4)
with reproducible evaluation and honest checkpoint reporting; and (5) an
experiment harness with identical per-index seeds across agents producing
JSON/CSV/chart artifacts from real runs. The delivered system passes 101
automated offline tests, runs fully without any LLM, and reports its own
limitations: the shipped RL checkpoint is undertrained and all LLM-path
results were produced through the fallback path.

## 1. Problem statement

Classic Snake is a benign, fully observable grid world, which makes it a
useful laboratory for comparing decision-making systems under identical
rules. Three questions motivate this project:

1. **Heterogeneous agent comparison.** How do heuristic, search-based,
   learned, and LLM-guided policies compare *fairly* — same seeds, same
   food statistics, same safety constraints — on a task with asymmetric
   rewards (normal +10 … special +100, trap −20) and expiring opportunities?
2. **Safe autonomy.** Can an LLM participate in real-time control without
   becoming a correctness or latency liability? LLMs hallucinate, think in
   seconds, and cannot be trusted with raw movement; what architecture
   bounds that risk?
3. **Honest evaluation.** Academic game-AI demos routinely show tuned
   cherry-picked runs. What does tooling look like that makes fabrication
   impossible by construction (identical seeds, auto-generated artifacts,
   truthful labels for untrained networks and fallback LLM paths)?

## 2. Objectives

- O1: A polished, configurable game (menu → arena → results), playable by a
  human against any agent, and by agents against each other.
- O2: Four decision systems — rule-based heuristics, BFS, A\*, Double DQN —
  plus LLM-augmented variants (`llm_astar`, `llm_rl`) and a hybrid
  combiner, all behind one decision pipeline.
- O3: A safety checker that is *independent of every policy* and has the
  final say on every action, with interventions measured and displayed.
- O4: An LLM planner (Ollama, local) with strict schema validation,
  snapshot caching, rate limiting, background execution, and a
  deterministic fallback; the game must be fully playable with the LLM off.
- O5: An experiment harness producing reproducible benchmarks and
  persistent artifacts (JSON/CSV/PNG) from real runs only.
- O6: Engineering quality: atomic persistence with corrupt-file recovery,
  100+ offline tests, cross-platform setup scripts, and documentation at
  academic-report depth.

All six objectives are met in the delivered code; O2's RL component is met
as *infrastructure* (training, evaluation, checkpoints, honest labels) — the
shipped weights are not a tuned policy (§5, §7).

## 3. System design (summary)

The engine is a deterministic tick simulation; `Snake.commit_move` applies
growth gradually (one segment per tick), collisions are tail-aware (moving
onto the vacating tail cell is legal), and simultaneous head-on collisions
eliminate both snakes (a draw). Five food types are defined declaratively in
`config/foods.json` (probability, lifetime range, growth, risk tier, score)
and spawned by a food-blind-safe spawner that never stacks food.

Every AI snake is driven by a `DecisionManager` executing the same pipeline:
**view → safety pre-computation → strategy hints → policy vote → safety
arbitration → logging**. The `SafetyValidator` checks each candidate action
for wall, reversal, self-collision, opponent-collision, and dead-end
(flood-fill area < snake length) and `select_safe_action` overrides unsafe
proposals with a safe alternative in a fixed order; a genuinely lost
position is reported as `no_safe_action` rather than hidden. Full detail:
[ARCHITECTURE.md](ARCHITECTURE.md), [AI_ALGORITHMS.md](AI_ALGORITHMS.md).

The LLM layer sends a compact JSON snapshot (top-6 foods by value per
distance, own state, nearest opponent, danger level, safe-route flags) to a
local Ollama model (default `qwen3:4b`) at most every 3 seconds, from a
background thread, and receives a strict-schema posture (strategy enum,
three [0,1] scalars, optional food target, one-sentence reason). Replies are
enum-checked, clamped, and target-validated; the snapshot is SHA-1-hashed
for caching; malformed or slow responses fall back deterministically.
Because the fallback is deterministic, LLM-mode behaviour is exactly
testable offline — the test suite exploits this.

The RL agent is a Double DQN (γ=0.9, Huber, Adam 5e-4, ε 1.0→0.05 over
25k steps) over a documented 30-float observation (danger triple,
post-move free-area ratios, heading, target/secondary food features,
opponent features, length ratio, hunger). Evaluation is greedy with fixed
seeds; checkpoints atomically store weights plus training metadata and are
described truthfully ("trained (N episodes, best score X)" vs "untrained").

## 4. Methodology

- **Fairness:** solo benchmarks assign every agent the identical seed list
  (`base_seed + 0..N-1`); divergent outcomes are therefore attributable to
  policies, not starting conditions (verified by test).
- **End conditions:** death, 300 s time limit, or 1 500-tick cap; survival
  counts only timeout endings (the agent never died).
- **Artifacts:** every run writes `results.json`, `summary.csv`,
  `records.csv`, and three matplotlib charts to a timestamped run
  directory; the UI can reload previous runs and the round-trip is
  byte-identical (tested).
- **Testing:** 101 pytest tests across engine mechanics, food system,
  pathfinding, safety, RL components, LLM client (programmable fake
  transport), decision manager (all seven modes + safety override),
  benchmark runner, persistence, end-to-end integration, and headless UI
  rendering; the suite needs no network, no display, and no Ollama.
- **Environment:** Arch Linux, Python 3.14.8, torch 2.14.1+cpu,
  pygame-ce 2.5.8, numpy 2.5.4, matplotlib 3.11.2. **Ollama was not
  installed**, so all LLM-mode results exercise the fallback path and are
  flagged `llm_available: false`.

## 5. Results (measured, this machine, 2026-10-19)

### 5.1 Benchmark (2 episodes per agent, identical seeds 1000–1001)

| agent | mean | median | std | survival | traps/1k ticks | ms/tick |
|---|---|---|---|---|---|---|
| Rule-Based | 2562.5 | 2562.5 | 342.5 | 0% | 1.45 | 0.10 |
| A* | 1825.0 | 1825.0 | 255.0 | 0% | 2.52 | 0.70 |

Per-episode: rule-based reached lengths 105/114 (endings: wall at 1479/1286
steps); A\* reached 84/80 (endings: self-collision at 918/672 steps).
Reading (with N=2, this is illustrative, not statistically supported): the
conservative heuristic survives longer, while optimal-per-step A\* dies to
self-trapping in food clusters — the classic greed/optimal-path tension this
arena is designed to expose. Artifacts: `data/benchmarks/cli-20261019-022020-abb6/`.

### 5.2 RL functional run (40 episodes, greedy eval on 20 seeds)

Mean score 83.75, median 90.0, mean 40 steps, endings {wall: 17, self: 3};
ε ended at ≈0.950 (the schedule spans 25 000 steps). The claim this supports
is narrow and deliberate: **the training/evaluation/checkpoint pipeline is
correct and honest**, not that the policy is good. It is not — 40 episodes
is orders of magnitude short, and no artifact in the project claims
otherwise.

### 5.3 Correctness results

- `python main.py --check` → **ALL CHECKS PASSED** (paths, dependencies,
  config, food catalog, engine smoke test, RL build, LLM probe).
- `pytest` → **101 passed** (~5.5 s), including: identical-seed verification
  across agents; safety-override of a deliberately stubborn policy;
  `no_safe_action` honesty; checkpoint round-trip determinism; LLM
  retry/fallback/fake-transport paths; corrupt-file quarantine and
  recovery; headless rendering of all seven screens.

## 6. Discussion

The architecture's central bet — that an LLM should *steer* rather than
*steer-wheel* — held up in testing: because postures only re-weight known
policies and the validator overrides everything, no LLM failure mode can
produce an invalid move, a stall, or a crash. The cost is a ceiling on LLM
upside: the planner cannot invent tactics the underlying policies cannot
express. Given reliability requirements, this trade is defensible and is the
paper's main design lesson.

The second lesson is methodological: identical seeds plus auto-generated
artifacts plus honest labels change what can be claimed. The benchmark table
above is simultaneously a real result and a demonstration of the system's
transparency — the reader can regenerate every cell with one command.

## 7. Limitations

1. **Undertrained RL.** The shipped checkpoint reflects 40 episodes;
   competitive RL play needs thousands and was not performed here.
2. **LLM path not exercised live.** No Ollama server was available on the
   development machine; LLM behaviour was verified against a programmable
   fake transport and the deterministic fallback, never against a real
   model's outputs.
3. **Windows untested.** Setup scripts were authored for Windows but the
   project was developed and tested on Arch Linux only (disclosed in
   [WINDOWS_SETUP.md](WINDOWS_SETUP.md) and the README).
4. **Small N.** Committed benchmark examples use 2 episodes to demonstrate
   the pipeline; conclusions need dozens.
5. **Simultaneous-turn model.** Opponent intentions are not predicted
   (head-aware avoidance only); adversarial modelling is out of scope.
6. **Single-node evaluation.** Decision times are one machine's wall clock.

## 8. Future work

- Long RL training with curriculum (solo → duel → crowded boards) and a
  published learning curve from the existing harness.
- Live-LLM study: posture quality vs. latency/failure rates across models
  (`qwen3:4b` vs larger), and prompt ablations — the harness already logs
  calls, failures, and latency per episode.
- Opponent-intention features in the observation (heading + predicted next
  cell), enabling genuine adversarial RL.
- MCTS as a fifth planner; its UCB machinery would slot into the existing
  pipeline unchanged.
- Multi-agent self-play league with Elo tracking in the tournament runner.

## 9. Artifacts & reproduction index

| Claim | Command | Artifact |
|---|---|---|
| Correctness | `python main.py --check` | console report |
| Test suite | `python -m pytest tests/ -q` | 101 passed |
| Benchmark table (§5.1) | `python main.py --benchmark 2 --agents rule_based,astar` | `data/benchmarks/cli-20261019-022020-abb6/` |
| RL numbers (§5.2) | `python main.py --train 40` (then delete `models/dqn_default.pt` to re-run from scratch) | `data/training/training_curves.png` |

Third-party assets: none — all graphics and audio are generated in code.
No personal data, API keys, or machine-specific absolute paths are embedded
in the repository (checked; runtime paths are resolved relative to the
project root at start-up).
