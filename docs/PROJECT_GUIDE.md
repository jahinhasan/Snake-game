# PROJECT GUIDE — Learn Every Part of the Snake Arena

*This guide was written for you (the project owner) to read before describing,
defending, or modifying the project. Everything here matches the real code in
`src/snake_arena/` — every snippet was checked against the actual files.*

---

## Table of contents

1. [What this project is (the 60-second version)](#1-what-this-project-is)
2. [How it was built (the true story, in order)](#2-how-it-was-built)
3. [Every technology used, and why](#3-every-technology-used)
4. [The folder map (what lives where)](#4-the-folder-map)
5. **The code, module by module** — the big section
   - 5.1 [Shared game types](#51-game_game-typespy--the-vocabulary)
   - 5.2 [The Snake entity](#52-gamesnakepy--the-snake-entity)
   - 5.3 [Collision logic](#53-gamecollisionpy--collisions--flood-fill)
   - 5.4 [Game state](#54-gamestatepy--the-world-state)
   - 5.5 [The engine (the heart)](#55-gameenginepy--the-simulator)
   - 5.6 [The food system](#56-gamefoodpy--the-food-system)
   - 5.7 [Observations + RL environment](#57-gameenvironmentpy--observations--rl-environment)
   - 5.8 [The agent contract](#58-agentsbasepy--the-agent-contract)
   - 5.9 [Rule-based agent](#59-agentsrule_basedpy)
   - 5.10 [BFS agent](#510-agentsbfspy--breadth-first-search)
   - 5.11 [A* agent](#511-agentsastarpy--a-search)
   - 5.12 [The DQN (deep RL)](#512-the-dqn--rl--deep-reinforcement-learning)
   - 5.13 [The safety checker](#513-safetyvalidatorpy--the-safety-checker)
   - 5.14 [The LLM layer](#514-the-llm-layer--llm)
   - 5.15 [The Decision Manager](#515-agentsdecision_managerpy--the-brain-stem)
   - 5.16 [The factory](#516-agentsfactorypy--the-factory)
   - 5.17 [Evaluation & experiments](#517-evaluation--benchmarks-metrics-artifacts)
   - 5.18 [Persistence](#518-persistence--safe-saving--history)
   - 5.19 [Config system](#519-configpy--the-settings-system)
   - 5.20 [The UI](#520-the-ui--pygame-screens)
   - 5.21 [Entry points](#521-entry-points--mainpy--scripts)
6. **Every command, explained** ([section 6](#6-every-command-explained))
7. [The tests](#7-the-tests)
8. [How to change things (recipes)](#8-how-to-change-things-recipes)
9. [Glossary + questions you may be asked](#9-glossary--likely-questions)

---

# 1. What this project is

**LLM-Enhanced Intelligent Snake Arena** — a Snake game where four kinds of AI
play on the same board under the same rules, and you can compare them fairly:

| Agent | How it thinks |
|---|---|
| **Rule-Based** | A scoring formula over the 4 possible moves (no search). |
| **BFS** | Breadth-first search: shortest path to the nearest food. |
| **A\*** | A\* search: optimal path + dead-end avoidance + value-aware food choice. |
| **RL (DQN)** | A neural network trained with Deep Q-Learning (PyTorch). |
| **LLM + A\* / LLM + RL** | A local LLM (Ollama, e.g. Qwen3) adds *strategy* on top. |
| **Hybrid** | Combines DQN Q-values + A\* path quality + free space into one score. |

The two big ideas that make this project special:

1. **The safety checker.** No matter which agent proposes a move, an
   independent validator checks it first (wall? reversal? own body? opponent?
   dead-end?) and replaces it with a safe move if needed. The LLM can never
   directly move the snake — it can only send *strategy hints*.
2. **Honest evaluation.** All agents get the *same seeds* in benchmarks, so
   score differences are skill differences. Untrained networks are labelled
   "untrained". Missing LLM is labelled "fallback". Nothing is faked.

---

# 2. How it was built

The true chronological order (useful when someone asks "how did you build
this?"):

1. **Environment first.** Created a Python virtual environment (`.venv/`),
   installed PyTorch (CPU build), NumPy, pygame, matplotlib, pytest. Checked
   that the machine has no Ollama — which shaped the design: *the game must
   work fully without the LLM*.
2. **The game engine** (`game/`): types → snake → collisions → state → engine
   → food. This is the referee: it knows the rules and nothing about AI.
3. **Pathfinding agents** (`agents/`): A\* and BFS written as real, generic
   search functions that return paths, plus thin agent wrappers that turn a
   path into a move. A rule-based baseline too.
4. **RL** (`rl/` + `game/environment.py`): the 30-number observation, the
   Q-network, replay buffer, reward function, trainer, checkpoints.
5. **Safety layer** (`safety/validator.py`): written *after* the agents, on
   purpose — it validates every agent through one pipeline.
6. **LLM layer** (`llm/`): HTTP client with an injectable "transport"
   function (so tests work with no network), strict JSON validation, the
   prompt, the snapshot builder, the background worker thread, and the
   deterministic fallback strategy.
7. **Decision Manager** (`agents/decision_manager.py`): the pipeline that
   glues view → safety → strategy → policy → safety again.
8. **Evaluation** (`evaluation/`): benchmark runner with shared seeds,
   metrics, JSON/CSV/PNG artifacts, tournaments.
9. **Persistence** (`persistence/`): atomic saves, corrupt-file quarantine,
   game history.
10. **UI** (`ui/`): App shell + widgets + 7 screens, all drawn with pygame
    (no image files — everything is drawn in code).
11. **Tests** (`tests/`): 101 tests, written by verifying each module's real
    API first. Writing them caught **two real bugs** (a food-spawning bug and
    a wrong `pygame.time.Clock` name) — a good story about why tests matter.
12. **Docs** (`docs/` + `README.md`), **support scripts** (Linux + Windows),
    and final verification: `--check` passes, benchmarks produce real
    artifacts, a genuine 40-episode training run produced `models/dqn_default.pt`.

Two honest notes that are part of the build story:
- The environment's pygame had to be replaced with **pygame-ce** (see §3).
- **Windows** setup scripts were written but never executed (development
  happened on Arch Linux) — the docs say this openly.

---

# 3. Every technology used

| Technology | What it is | Why this project uses it |
|---|---|---|
| **Python 3.14** | The programming language | Ecosystem for AI; required by the spec. |
| **venv** (`.venv/`) | Isolated Python environment | Keeps project dependencies separate from the system. |
| **pygame-ce 2.5.8** | 2D game library (community edition of pygame; still `import pygame`) | Window, drawing, input, fonts, sound. **Why not classic pygame?** Classic pygame 2.6.1 has no ready-made package for Python 3.14; building it from source here produced a broken `font` module (a circular import bug). pygame-ce ships official wheels for new Python versions and is a drop-in replacement. |
| **PyTorch 2.14 (CPU)** | Deep-learning framework | Builds and trains the DQN; runs on CPU by design (GPU optional). |
| **NumPy** | Fast numeric arrays | The observation vector and replay buffer storage. |
| **Matplotlib** | Plotting library | Training curves and benchmark charts (PNG). |
| **pytest** | Testing framework | The 101 automated tests. |
| **Ollama + Qwen3 4B** | Local LLM server + model (optional) | The strategic planner. Runs on your machine, no API key, no internet. |
| **urllib** (Python stdlib) | HTTP client | Talks to Ollama — no extra dependency needed. |
| **dataclasses** (stdlib) | Auto-generated data classes | Clean structs: `Snake`, `Food`, `EpisodeRecord`… |
| **threading** (stdlib) | Background threads | The LLM worker and UI training/benchmark runners never block the game loop. |
| **JSON / JSONL / CSV** | File formats | Config, benchmarks, history. All written *atomically*. |

Key standard-library tools used and worth knowing for questions:
`collections.deque` (snake body), `heapq` (A\* priority queue), `random.Random`
(seeded reproducibility), `enum.IntEnum` (actions), `pathlib` (paths),
`argparse` (CLI), `logging` (logs to console + `data/logs/app.log`).

---

# 4. The folder map

```
snake-game/
├── main.py                  ← THE entry point (UI by default, --check/--train/--benchmark)
├── requirements.txt         ← dependency list
├── setup_linux.sh           ← one-time installer (Linux/macOS)
├── run_linux.sh             ← launcher (Linux/macOS)
├── setup_windows.ps1        ← installer (Windows — written, not yet tested)
├── run_windows.bat          ← launcher (Windows)
├── config/
│   ├── default.json         ← ALL settings (board, game, LLM, RL, benchmark…)
│   └── foods.json           ← the 5 food types (value, %, lifetime, colour…)
├── src/snake_arena/         ← the actual program
│   ├── game/                engine, state, snake, collisions, food, RL environment
│   ├── agents/              rule_based, bfs, astar, rl_agent, decision_manager, factory
│   ├── safety/              validator.py — the safety checker
│   ├── llm/                 ollama_client, strategist, schemas, analysis
│   ├── rl/                  network, replay_buffer, rewards, trainer, checkpoints
│   ├── evaluation/          benchmark, metrics, experiments (artifacts)
│   ├── persistence/         storage (atomic IO), history (games.jsonl)
│   ├── ui/                  app shell + 7 screens + widgets + theme + audio
│   ├── config.py            AppConfig: load/validate/save/clamp settings
│   └── utils/               paths.py (where files live), logging_setup.py
├── scripts/                 headless CLIs: check_environment, train, benchmark
├── tests/                   101 pytest tests (12 files)
├── models/                  DQN checkpoints (models/dqn_default.pt)
├── data/                    runtime output (created on first run)
│   ├── benchmarks/<run-id>/ results.json, summary.csv, records.csv, charts
│   ├── training/            training_curves.png
│   ├── history/games.jsonl  every played game
│   └── logs/app.log         rotating log
└── docs/                    ARCHITECTURE, AI_ALGORITHMS, RL_TRAINING,
                             EXPERIMENTS, WINDOWS_SETUP, REPORT, this guide
```

---

# 5. The code, module by module

## 5.1 `game/types.py` — the vocabulary

The smallest module but everything depends on it. It defines the shared
language of the game, with **no imports from other game modules** (this
prevents circular imports).

```python
Cell = tuple[int, int]          # a grid square, e.g. (12, 7)

class Action(IntEnum):
    UP = 0; DOWN = 1; LEFT = 2; RIGHT = 3

DIR_VECTORS = {Action.UP: (0, -1), Action.DOWN: (0, 1),
               Action.LEFT: (-1, 0), Action.RIGHT: (1, 0)}

OPPOSITE = {Action.UP: Action.DOWN, ...}   # used for reversal checks
```

Helper functions you will see everywhere:
- `manhattan(a, b)` — grid distance: `|dx| + |dy|`. It's the distance an A\*
  heuristic uses, and the "how far is the food" measure.
- `neighbors(cell)` — yields the 4 surrounding cells with the action that
  reaches each.
- `direction_from_cells(tail, head)` — infers which direction points from one
  cell to an adjacent one. The snake uses it to know "which way am I facing"
  **from geometry**, not from a stored variable — so it can never lie.

> **Why tuples for cells?** They are immutable, hashable, and fast — required
> for putting cells into `set`s (collision and pathfinding use sets heavily).

## 5.2 `game/snake.py` — the snake entity

```python
@dataclass
class Snake:
    id: str
    name: str
    is_human: bool
    body: deque[Cell]          # head is body[0], tail is body[-1]
    direction: Action
    alive: bool = True
    score: int = 0
    pending_growth: int = 0    # segments the snake still "owes" from eaten food
    ...
```

**The single most important trick in the file** — gradual growth:

```python
def commit_move(self, new_head: Cell, growth_now: int = 0) -> None:
    self.pending_growth += max(0, growth_now)
    self.body.appendleft(new_head)          # new head goes to the front
    if self.pending_growth > 0:
        self.pending_growth -= 1            # keep the tail → grow by 1
    else:
        self.body.pop()                     # tail moves away → same length
```

When a snake eats a "special" food with `growth=2`, the 2 is *not* applied at
once. The counter goes up, and the tail is retained for the next two ticks.
This mirrors the classic game and matters for collision math: while
`pending_growth > 0`, the tail will **not** vacate, so pathfinding must treat
it as solid.

Also important: `is_reversal(action)` — moving onto your own neck cell. The
snake of length 1 has no neck, so it can go anywhere.

> **Learn this word: `deque`** (pronounced "deck"). A double-ended queue:
> adding/removing at either end is O(1). A snake body is the perfect use case:
> new head at the front (`appendleft`), tail off the back (`pop`).

## 5.3 `game/collision.py` — collisions & flood fill

Pure logic shared by the engine, the safety validator, and all planners, so
everyone agrees on the rules.

**Tail-awareness** — the core idea:

```python
def will_tail_vacate(snake, growth_now=0) -> bool:
    return snake.pending_growth == 0 and growth_now == 0
```

A tail cell is only a wall if it will still be there next tick. Planners use
`planning_blocked_cells(state, snake_id)` which builds the set of walls: all
bodies, minus tails that will vacate.

**`is_deadly_for(state, snake, cell)`** answers "would moving here kill me
right now?" and returns `"wall"`, `"self"`, `"opponent"`, or `None`. It is
tail-aware in the same way.

**`flood_fill_count(start, blocked, w, h, limit)`** — dead-end detection.
From a candidate cell, it spreads to all connected free cells (like water
flowing) and *counts* them:

```python
seen = {start}; frontier = [start]; count = 0
while frontier and count < limit:
    cell = frontier.pop(); count += 1
    for each 4-neighbour not seen/blocked/in bounds:
        seen.add(...); frontier.append(...)
return count
```

If `count < snake.length`, that move enters a *pocket* the snake cannot
survive. That one number powers the dead-end check everywhere.

## 5.4 `game/state.py` — the world state

A plain data container:

```python
@dataclass
class GameState:
    width, height
    snakes: dict[str, Snake]
    foods: list[Food]
    tick: int
    max_foods, moves_per_second, max_ticks
    rng: random.Random          # THE single source of randomness
    game_id: str
```

Note there is **no pygame here** — the state is pure data. That's what lets
the same engine run the GUI, headless training, and benchmarks. Useful
queries: `is_inside`, `food_at`, `occupied_cells(include_food=False)`,
`living_snakes`, `opponents_of(id)`.

## 5.5 `game/engine.py` — the simulator

`GameEngine.step(actions: dict[str, Action])` advances the world by exactly
one tick. Read it as an 8-step recipe:

1. **Resolve intent.** Missing/invalid action → keep current direction.
   A reversal attempt is *ignored* (event `reversal_ignored`) — the snake
   keeps going forward. This enforces the no-U-turn rule at engine level,
   for humans and AIs alike.
2. **Wall deaths.** Any candidate head outside the board dies (`"wall"`).
3. **Head-on deaths.** Two heads aiming at the *same cell*, or *swapping*
   cells (A→B's cell and B→A's cell) → both die (`"head_on"` → a draw).
4. **Body collisions, simultaneously.** For every survivor, compute its
   `future_body` (new head inserted, tail popped if it vacates). Then:
   your head in your own future body → `"self"`; your head in someone else's
   future body → `"opponent"`. Because everyone's future is computed *before*
   anyone dies, two snakes moving into each other's vacated tails both
   survive — correct simultaneous resolution.
5. **Commit movement + eating.** `commit_move(..., growth_now=food.growth)`,
   add `food.value` to score, reset `steps_since_food`, record stats, emit
   an `"eat"` (or `"trap"`) event.
6. **Apply deaths** (mark `alive=False`, set `death_reason`, `death_tick`).
7. **Food maintenance.** Expire old food, spawn replacements (`FoodSystem`).
8. **End conditions.** If someone died: last snake standing wins
   (`survivor_win`), nobody → `draw` (multi-snake) or `death` (solo). If
   `tick >= max_ticks` → `timeout`, highest score wins (tie → `draw`).

**Determinism:** all randomness flows through `self.rng = random.Random(seed)`.
Same seed → same food spawns, same bounces, same everything. This is what
makes fair benchmarks and stable tests possible.

## 5.6 `game/food.py` — the food system

Five types are **data, not code** — defined in `config/foods.json`:

| type | value | chance | lifetime (s) | growth |
|---|---|---|---|---|
| normal | +10 | 55% | 25–45 | 1 |
| bonus | +25 | 20% | 14–26 | 1 |
| golden | +50 | 13% | 9–16 | 1 |
| special | +100 | 7% | 6–11 | 2 |
| trap | −20 | 5% | 18–30 | 0 |

`FoodCatalog.load()` reads the JSON, **normalizes probabilities** (so they
always sum to 1 even if you edit them badly), and falls back to built-in
defaults if the file is missing/corrupt. `weighted_choice(rng)` spins the
loaded probabilities.

`FoodSystem.spawn_one` picks a uniformly random **genuinely free cell**
(occupied = snakes **+ existing food** — this used to be a real bug where
food could spawn on food; it was caught by a test) and skips silently if the
board is full. `maintain()` each tick: remove expired food, refill up to
`max_foods`.

Each `Food` has `spawned_tick` and `lifetime_ticks`, so
`remaining_ratio(tick)` goes 1.0 → 0.0 — the UI uses it to blink food that's
about to disappear.

## 5.7 `game/environment.py` — observations + RL environment

Three things live here.

**(a) `StateView`** — a per-tick "briefing pack" built once and shared:

```python
view = build_state_view(state, snake_id)
view.me            # my Snake
view.opponents     # other living snakes
view.blocked       # tail-aware wall set (for planning)
view.foods_sorted  # best food first: value / (1 + manhattan)
view.target        # best positive food (or nearest if only traps remain)
```

**(b) `ObservationBuilder.build()`** — turns the world into **30 numbers**
(the neural network's eyes). Layout (also documented in the file header):

| indices | meaning |
|---|---|
| 0–2 | danger straight/left/right (wall, body, or opponent) |
| 3–5 | free area after each of those moves, `min(1, area/length)` — dead-end sense |
| 6–9 | current heading, one-hot |
| 10–16 | target food: 4 direction signs, distance, |value|/100, is-trap |
| 17–20 | 2nd-best food: 4 direction signs |
| 21–27 | opponent: present?, head signs, distance, longer-than-me? |
| 28 | my length / board area |
| 29 | hunger: `steps_since_food / 150` |

All values are ~0..1 (good for neural nets). The same builder is used in
training *and* in games — they can never drift apart.

**(c) `SnakeEnv`** — wraps the engine in a gym-style RL interface:

```python
env = SnakeEnv(width=20, height=15, max_foods=4, max_steps=1200, seed=...)
obs = env.reset(seed)
obs, events, terminated, truncated, info = env.step(action_int)
```

`terminated` = died (real end). `truncated` = hit the step cap (artificial
end). The RL code treats these differently — that distinction matters in RL.

## 5.8 `agents/base.py` — the agent contract

Every agent implements one method:

```python
class BaseAgent(ABC):
    @abstractmethod
    def decide(self, state, snake_id, view=None, hints=None) -> AgentDecision: ...
```

`AgentDecision` is a **proposal**, never a command:

```python
@dataclass
class AgentDecision:
    agent: str
    action: Action              # what I *want* to do
    path: list[Cell]            # the planned route (UI draws this!)
    target_food_id / target_pos # what I'm going for
    q_values: dict | None       # RL: my four Q-values (UI shows bars)
    notes: list[str]            # human-readable reasons
    diagnostics: dict           # agent-specific extras
    computation_ms: float       # how long I thought
```

`StrategyHints` is the LLM's *only* output channel:

```python
@dataclass
class StrategyHints:
    strategy: str = "balanced"   # defensive|balanced|aggressive|greedy|survival
    food_priority: float = 0.5   # 0..1
    survival_priority: float = 0.5
    risk_tolerance: float = 0.5
    target_food_id: str | None = None
    source: str = "default"      # default | llm | fallback
```

Notice: **no `Action` anywhere in the hints.** The LLM literally cannot say
"move right". That is the design decision the whole safety story hangs on.

## 5.9 `agents/rule_based.py`

The "classical AI baseline". For each of the 4 actions it computes a utility:

```
U(a) = 2.0 * progress(a)              # do I get closer to the best food?
     + 0.5 * min(1, free_area(a)/8)   # do I keep breathing room?
     - 3.0 * trap(a)                  # is this (or near) a trap food?
     - 1.5 * contested(a)             # can the opponent reach this cell next tick?
```

Deadly moves get −1,000,000 up front. Highest utility wins. No search at all
— one pass over 4 actions → ~0.1 ms. It survives long because it values
space and avoids traps, but it doesn't plan, so it's inefficient.

## 5.10 `agents/bfs.py` — breadth-first search

`bfs_search(start, goal, width, height, blocked)` is a textbook BFS:

```python
queue = deque([start]); parent = {start: start}; visited = {start}
while queue:
    current = queue.popleft()              # FIFO = waves of distance
    for each neighbour:
        if blocked/visited/out: continue
        visited.add; parent[nxt] = current
        if nxt == goal: rebuild path via parent links; return
```

BFS explores in "waves" of increasing distance, so the first time it reaches
the goal, the path is a **shortest path** (fewest steps) — on a grid where
every step costs 1. Cost: visits many cells; no notion of "better" food.

The **BFS agent** targets the *nearest non-trap food*, deliberately ignoring
food values — it's the control group that shows what value-aware targeting
(A\* agent) is worth. Both agents also flood-fill-check that the destination
pocket is survivable, with a tail-chase fallback when no food is reachable.

## 5.11 `agents/astar.py` — A\* search

A\* = Dijkstra + a hint. It keeps a **priority queue** ordered by

```
f(n) = g(n) + h(n)      g = cost paid so far, h = estimated cost to goal
```

`h` here is Manhattan distance, which is *admissible* (never overestimates on
a 4-connected grid) — that guarantees the found path is optimal, and *A\*
expands far fewer cells than BFS* because it leans toward the goal. Extra
features:

- **turn penalty (0.02):** changing direction costs a tiny bit → smoother,
  nicer-looking paths without changing their length.
- **survivability check** (`route_is_survivable`): flood-fill from the path's
  end; if the pocket is smaller than the snake, skip that food (it's bait).
- **food scoring:** candidates are ranked by a documented utility:

```
U(f) = w_p * value/(1+dist)/10  -  w_r * risk(f)  -  0.3 if it expires before arrival
w_p = 0.35 + 1.3 * food_priority        (from the LLM hints!)
w_r = 0.6 * (1 - risk_tolerance)        (from the LLM hints!)
```

- **survival fallback:** chase own vacating tail — the classic safe loop.
- Re-plans **every tick** (cheap at these board sizes), so paths are never stale.

## 5.12 The DQN — `rl/` — deep reinforcement learning

**The idea in one paragraph.** Q-learning learns a "quality" number Q(s, a)
— "if the world looks like `s` and I do `a`, how good is my future?" — by
trial and error. A DQN approximates Q with a neural network. After every
step we store the experience and, occasionally, nudge the network so that

```
Q(s,a)  →  closer to  reward + γ · max_a' Q(s', a')     (γ = 0.9 = value of the future)
```

**`rl/network.py`** — the network is a small MLP:

```python
obs(30) → Linear(128) + ReLU → Linear(128) + ReLU → Linear(4)   # one Q per action
```

Small on purpose: 30 inputs is tiny; 128×128 trains fine on a CPU.
`pick_device()` uses CUDA/MPS only if genuinely present.

**`rl/replay_buffer.py`** — a fixed-capacity **ring buffer** of experiences
`(obs, action, reward, next_obs, done)` stored as preallocated NumPy arrays.
`pos` wraps around (`(pos+1) % capacity`) → old memories are overwritten.
Sampling a random batch of 64 breaks the correlation between consecutive
steps — a key trick that stabilizes DQN training.

**`rl/rewards.py`** — the reward signal:

| event | reward |
|---|---|
| eat food | `0.1 × value` → +1 … +10 |
| trap | −2 |
| death (wall/self/opponent) | −10 |
| move closer/farther to target food | ±`0.12 × Δdistance` (shaping) |
| 120+ steps without eating | small growing penalty |
| step cost | 0 |

The shaping term is *potential-based* (it depends only on distance change),
which speeds learning without changing which behaviour is optimal. Deaths
(−10) dominate any accumulated shaping (≤ ~1/episode), so the agent cannot
"win" by hiding forever — it must eat.

**`rl/trainer.py`** — the learning loop:

- `epsilon()` — exploration schedule: with probability ε take a random
  action; ε decays **1.0 → 0.05 over 25 000 steps** (first: pure random
  exploration; later: trust the net).
- `learn()` — one gradient step:
  ```python
  q_sa = online(obs).gather(1, actions)                     # predicted value
  with torch.no_grad():                                     # target is fixed
      best = online(next_obs).argmax(...)                   # Double DQN:
      q_next = target(next_obs).gather(1, best)             #   online PICKS, target VALUES
  q_target = rewards + gamma * (1 - dones) * q_next         # dones cut the future
  loss = SmoothL1Loss(q_sa, q_target)                       # Huber = robust to outliers
  optimizer.zero_grad(); loss.backward()
  clip_grad_norm_(online.parameters(), 10.0)                # stability
  optimizer.step()
  ```
  **Two networks** fight instability: `online` learns each step; `target` is
  a frozen copy refreshed every 1000 steps that provides the "goal" values.
  **Double DQN** (config `double_dqn: true`) decouples *choosing* the best
  next action (online net) from *valuing* it (target net) — a standard fix
  for DQN overestimating values.
- `run_episode(env, greedy, seed, ...)` — play one full episode, optionally
  learn each step; `evaluate(env_factory, episodes, seed_base)` plays greedy
  with fixed seeds → **deterministic, reproducible numbers**.
- `save()/load()` — full checkpoints (see below); `training_worker()` is the
  loop used by both the CLI and the UI dashboard, with pause/stop flags and
  autosave every 50 episodes.

**`rl/checkpoints.py`** — a checkpoint is a dict saved with
`torch.save` **atomically** (write `.tmp` → `rename`): weights of both
networks, optimizer state, hyperparameters, full training history, and the
honest labels `episodes` / `trained`. `describe_checkpoint()` turns that
into `"trained (440 episodes, best score 190)"` or `"untrained"` — this exact
string appears in logs and UI, so an untrained model can never pass as smart.

## 5.13 `safety/validator.py` — the safety checker

Independent of every agent. `check(state, snake_id, action)` returns a
`SafetyReport(action, safe, reason, free_area)` and rejects:

| reason | meaning |
|---|---|
| `wall` | next cell outside the board |
| `reversal` | 180° turn into the neck |
| `self_collision` | next cell in own body (tail-aware, strict mode) |
| `opponent_collision` | next cell in an opponent's body |
| `dead_end` | flood-fill area after the move < snake length |

The arbiter — the method every mode's action passes through:

```python
def select_safe_action(self, state, snake_id, preferred, fallback_order=..., target=None):
    reports = self.evaluate_all(state, snake_id, view)
    if reports[preferred].safe:
        return preferred, preferred_report            # 1. wish granted
    rejected.append(...)
    candidates = [a for a in fallback_order if a != preferred and reports[a].safe]
    if candidates:
        best = min(candidates, key=lambda a: (-free_area, distance_to_target))
        return best, SafetyReport(best, True, None, ..., rejected)   # 2. overridden
    return preferred, SafetyReport(preferred, False, "no_safe_action", 0, rejected)
```

Rule 3 matters for honesty: if the snake is truly trapped, we don't fake a
magic escape — we return the move flagged `no_safe_action` and the record
shows the truth. Every intervention is counted
(`manager.stats["safety_interventions"]`, `snake.rejected_moves`) and can be
drawn on the board.

## 5.14 The LLM layer — `llm/`

**`ollama_client.py`** — a minimal HTTP client using only `urllib`
(no `requests` dependency). Three endpoints matter: `GET /api/tags` (is the
server up? which models exist?), `POST /api/chat` (talk to the model). Key
design points:

- **Injectable transport.** The HTTP call is a callable
  `transport(path, payload, timeout) -> (status, body)`. Tests inject a fake
  — the whole LLM stack is tested offline because of this one seam.
  ```python
  Transport = Callable[[str, bytes | None, float], tuple[int, bytes]]
  ```
- **JSON mode**: the request includes `"format": "json"` and
  `"think": False` (Qwen3's thinking mode off for speed).
- `chat_raw` retries (≤ `max_retries`) on transport errors *and* malformed
  replies, then raises typed exceptions: `OllamaUnavailable`, `OllamaTimeout`,
  `OllamaBadResponse`. Callers always degrade gracefully — never crash.

**`schemas.py`** — strict validation of what the model said:

```python
def parse_strategy(obj) -> Strategy:      # raises StrategyValidationError on ANY violation
    strategy must be one of 5 names
    food_priority / survival_priority / risk_tolerance must be numbers in [0,1]
    target_food_id must be null or a string
    reason must be a string (trimmed to 300 chars)
```

`extract_json_object(text)` handles Qwen3's `<think>...</think>` blocks and
stray chatter by finding the first balanced `{...}` and parsing that.
Rejection (not silent fixing) is the policy — a bad reply falls back.

**`strategist.py`** — the planner itself:

- `SYSTEM_PROMPT` tells the model: *"You do NOT control movement directly…
  Respond with ONLY a JSON object"*, and defines the schema + when to prefer
  each strategy.
- `build_snapshot(state, snake_id, view, safe_routes)` — a compact JSON: top-6
  foods (id, type, value, risk, distance, remaining lifetime, contested?),
  own score/length/hunger, nearest opponent + a danger level. Compact = fast
  and cheap for a 4B model.
- `snapshot_hash` — SHA-1 of canonical JSON → identical situations reuse the
  cached answer.
- `fallback_strategy(snapshot)` — the deterministic stand-in:
  high danger → defensive; valuable food within 7 → greedy; hungry → greedy;
  else balanced. This makes LLM-mode behaviour **fully testable offline**.
- `StrategyWorker` — a **daemon thread** with a `threading.Condition`:
  `submit(snapshot)` replaces any pending snapshot (only the latest matters);
  the loop calls the LLM at most once per `min_interval` seconds (3 s), stores
  the newest `StrategyResult`, and never touches pygame. The game loop only
  *peeks* at the latest result — if empty, it uses fallback and moves on. A
  hung LLM can freeze this thread, never the game.

**`analysis.py`** — after a game, optionally asks the model for a narrative
about the match; stored **clearly marked as model-generated**. Without an
LLM, a deterministic summary is stored instead.

## 5.15 `agents/decision_manager.py` — the brain-stem

One class drives all 7 modes through the same pipeline:

```python
def decide(self, state, snake_id) -> Action:
    view         = build_state_view(state, snake_id)          # 1. observe
    reports      = validator.evaluate_all(...)                # 2. safety pre-check (all 4 moves)
    hints        = self._hints(...)                           # 3. strategy: worker | sync | fallback
    proposed     = self._policy(...)                          # 4. policy vote
    final, report = validator.select_safe_action(             # 5. safety arbitration
                        ..., proposed.action, fallback_rank, ...)
    ...                                                        # 6. log everything
    return final
```

Mode dispatch (in `_policy`):

```python
"rule_based"            -> rule_agent
"astar" / "llm_astar"   -> astar_agent
"bfs"                   -> bfs_agent
"rl"   / "llm_rl"       -> rl_agent, then _gate_rl (risk gating)
"hybrid"                -> _hybrid (weighted fusion)
```

**RL risk gating** (`_gate_rl`) — how hints tame the network: walk the
Q-value ranking and take the first action whose free area is at least

```
min_area = length × (0.8 + 0.9·survival_priority) × (1 − 0.5·risk_tolerance)
                                                              (×0.7 if aggressive)
```

so a "survival" strategy makes the RL agent needier about open space, and
"aggressive" relaxes it. Every activation is counted (`gating_activations`).

**Hybrid scoring** (`_hybrid`) — combines three signals on each *safe* action:

```
score(a) = w_rl · Q_normalized(a) + w_path · path_quality(a) + w_space · free_area(a)
```

Weights come from config (`decision.hybrid`) and are re-weighted by the
strategy (`food_priority` boosts path, `survival_priority` boosts space),
then normalized to sum to 1. Q-values are min-max normalized first — you
can't average raw volts with metres.

Everything is recorded per tick in a `DecisionLog` (proposed vs final,
reasons, strategy source, Q-values, notes, timings) kept in a 60-entry deque
that the arena screen renders live.

## 5.16 `agents/factory.py` — the factory

The one place that knows how to *build* things:

- `resolve_checkpoint_path(config, name)` — absolute stays; `models/x.pt`
  resolves from the project root; a bare `x.pt` resolves into `models/`.
  (This function had a real bug — bare names landed in the project root —
  found and fixed during end-to-end testing.)
- `build_rl_agent(config)` — loads the checkpoint if present and valid;
  otherwise returns an honest **untrained** agent (random init + safety net).
- `build_llm_client` / `build_strategy_worker` / `build_manager` — wiring;
  passing neither worker nor client gives a guaranteed-offline manager.

## 5.17 Evaluation — benchmarks, metrics, artifacts

**`evaluation/benchmark.py`** — `BenchmarkRunner`:

- solo episodes: `seed = base_seed + episode_index` for **every agent** →
  agent A's episode 3 and agent B's episode 3 are the *same board*. That is
  the entire fairness mechanism, and there's a test that proves it.
- LLM agents (`llm_astar`, `llm_rl`) play fewer episodes by default
  (LLM calls are expensive) and each record carries
  `llm_available = (llm_calls > 0 and llm_failures == 0)`; the run header
  records a live probe of Ollama.
- `run_match(a, b, seed)` — symmetric duel; `run_tournament(...)` —
  round-robin with wins/draws/win-rate.

**`evaluation/metrics.py`** — `EpisodeRecord` (agent, seed, score, steps,
length, outcome, death reason, foods, safety interventions, decision ms, LLM
stats) and `summarize(records)` → mean/median/std/min/max, survival rate,
traps per 1k ticks, death reasons.

**`evaluation/experiments.py`** — `make_run_dir(prefix)` (timestamped folder
under `data/benchmarks/`), `save_benchmark_results` (results.json +
summary.csv + records.csv), `plot_benchmark_charts` (3 PNGs),
`load_run`, `plot_training_curves`. Every number in every artifact comes
from a real run — the code path that writes them is the same one that plays
the games.

## 5.18 Persistence — safe saving & history

**`persistence/storage.py`** — the atomic-IO toolbox:

```python
def save_json_atomic(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ...))
    tmp.replace(path)            # rename is atomic on POSIX & Windows
```

A crash mid-write can never destroy the last good file. `load_json`
quarantines unreadable files as `*.corrupt` and returns the default.
`read_jsonl` skips damaged lines instead of failing the whole history.

**`persistence/history.py`** — every played game is appended to
`data/history/games.jsonl` via `build_game_record(...)` + `record_game(...)`.
`aggregate()` computes per-agent games/mean/max and the top game — that's
what the Statistics screen shows. `clear_history()` deletes it.

## 5.19 `config.py` — the settings system

`AppConfig` is a dataclass tree with 12 sections (`board`, `game`, `window`,
`audio`, `visual`, `agent`, `llm`, `rl`, `safety`, `decision`, `benchmark`,
`training`). Behaviours to remember:

- `validate()` **clamps** bad values into legal ranges (e.g. board size
  10–40) instead of crashing.
- `save()` is atomic; `load_app_config()` recovers from a corrupt file by
  quarantining it and returning defaults; partial JSON merges into defaults
  (missing keys keep defaults); env overrides:
  `SNAKE_ARENA_DISABLE_LLM`, `OLLAMA_HOST`, `OLLAMA_MODEL`.
- Default: **`llm.enabled = false`** — the game never requires the LLM.

## 5.20 The UI — pygame screens

**`ui/app.py`** — the shell. `App` owns the window, fonts, audio, and a
**screen stack**:

```python
app.push_screen(ArenaScreen(...))   # go deeper (menu → arena → results)
app.pop_screen()                    # back
app.replace_screen(MenuScreen(app)) # switch (with "Back to menu")
```

The main loop runs at up to 60 fps: pump events → `active_screen.update(dt)`
→ `draw(surface)` → `flip()`. Screens are only ever touched on the main
thread; LLM and training run on their own threads (see §5.14, §5.12).
`App` also caches shared services: `llm_client()`, `llm_worker()`,
`rl_agent()` — so the checkpoint loads once, the worker thread exists once.

**`ui/theme.py`** — a colour palette (`COLORS`), font management (`Fonts`),
and tiny helpers (`text()`, `panel()`, `hline()`, `wrap_text()`).

**`ui/components.py`** — a small widget kit written from scratch:
`Button`, `Slider`, `Toggle`, `Dropdown`, `TextInput`, `ProgressBar`,
`DataTable`, `BarChart`, plus `draw_widgets(surface, widgets, fonts)` and
`send_event(widgets, event)`. Every widget owns a `pygame.Rect` and
`handle_event` does hit-testing with `rect.collidepoint(pos)`.

**The 7 screens** (`ui/menu.py`, `arena.py`, `results.py`,
`training_dashboard.py`, `benchmark_dashboard.py`, `settings.py`,
`statistics.py`):

- **MenuScreen / ModeSelectScreen** — title, mode pick (human vs AI, AI vs
  AI), agent dropdowns, speed slider, seed, "Environment check" status.
- **ArenaScreen** — the game. Grid, snakes (rounded segments), food shapes
  per type with expiry blink, HUD (scores, timer), overlays you can toggle:
  A\* path, danger cells, target, rejected moves, Q-value bars; the
  diagnostics panel shows the latest `DecisionLog` (strategy source,
  interventions, notes). Human input: arrows/WASD, Space pause, R restart.
- **ResultsScreen** — end-of-game table (scores, lengths, foods by type,
  deaths), safety/LLM summary, Rematch / Menu.
- **TrainingDashboard** — start/pause/stop training in a background thread;
  live episode/score/loss/epsilon chart (`BarChart`), eval results, autosave
  status, checkpoint name input.
- **BenchmarkDashboard** — pick agents + episode counts, run in a thread
  with a progress bar and Cancel; results table; load previous runs;
  tournament mode; a gold banner when LLM results are fallback-only.
- **SettingsScreen** — every config section as widgets, bound live to
  `app.config`; Save (validate + atomic save) / Restore Saved / Back.
- **StatisticsScreen** — history aggregates + the last 30 games table,
  Clear History.

**`ui/audio.py`** — small generated tones (no sound files); safely no-ops
when audio is unavailable.

> Everything visual is drawn with `pygame.draw` — **zero image/font asset
> files**, which is why the project runs from a clean clone with no asset
> downloads.

## 5.21 Entry points — `main.py` & `scripts`

`main.py` = argument parsing + logging setup + dispatch:

| flag | does |
|---|---|
| *(none)* | open the graphical app |
| `--check` | `scripts/check_environment.py` → verify everything, exit 0/1 |
| `--train N` | `scripts/train.py` → N episodes of DQN training |
| `--benchmark N` | `scripts/benchmark.py` → headless benchmark |
| `--agents a,b` | restrict benchmark agents (comma-separated) |
| `--no-llm` | force `config.llm.enabled = False` for this run |
| `--config PATH` | use an alternative config JSON |
| `-v` | debug logging |
| `--version` | print version |

It also inserts `src/` into `sys.path` so a plain
`python main.py` works without installing the package.

---

# 6. Every command, explained

Run all of these from the project folder `~/snake-game`.

### One-time setup

```bash
./setup_linux.sh
```
*What it does:* creates `.venv` (an isolated Python), installs CPU-only
PyTorch from PyTorch's own index (small download), then the rest of
`requirements.txt`. You only run this once (or after changing requirements).

### Everyday launch

```bash
./run_linux.sh                    # or directly:
.venv/bin/python main.py
```
*Meaning:* run the program with the project's private Python. The window
opens on the Menu. Arena controls (from `ui/arena.py`): **arrows / WASD**
move, **Space** pause, **R** restart, **Enter** continue after game over,
**+ / −** speed up / slow down, **1 / 2** toggle AI overlays, **Esc** back.

### Health check

```bash
.venv/bin/python main.py --check
```
*Meaning:* no window; verifies paths, all libraries, config + food catalog,
runs a 30-tick engine smoke test, builds the DQN, probes Ollama. Aim for
**ALL CHECKS PASSED** (exit code 0). Run this first whenever anything feels
broken.

### Training the RL agent

```bash
.venv/bin/python main.py --train 400
```
*Meaning:* train the DQN headless for 400 episodes (each episode = the snake
plays until it dies or 1200 steps). Prints progress every ~5% (score, total
reward, ε, loss), autosaves every 50 episodes, runs a greedy evaluation at
the end, saves `models/dqn_default.pt`, writes
`data/training/training_curves.png`.

*Facts to know:*
- It **resumes**: re-running continues from the saved checkpoint. Delete
  `models/dqn_default.pt` to start over.
- 40 episodes ≈ seconds (functional test). Real skill needs **thousands**.
- ε (epsilon) is the "explore randomly" probability: starts at 1.0 (pure
  random), decays to 0.05 by 25 000 steps.

Equivalent form: `.venv/bin/python scripts/train.py --episodes 400`.

### Benchmarking agents

```bash
.venv/bin/python main.py --benchmark 20
```
*Meaning:* every agent (all 7) plays 20 headless episodes with **identical
seeds**, then each gets a summary (mean/median/std score, survival, traps/1k
ticks, ms per decision). Results are saved to
`data/benchmarks/cli-<timestamp>-<id>/` as `results.json`, two CSVs, and
three PNG charts.

Subsets:

```bash
.venv/bin/python main.py --benchmark 10 --agents rule_based,astar
```
*Meaning:* only those two agents; agent names must be from:
`rule_based, astar, bfs, rl, llm_astar, llm_rl, hybrid`.

### AI-vs-AI tournament

```bash
.venv/bin/python -m scripts.benchmark --episodes 1 --agents rule_based,astar --tournament
```
*Meaning:* `-m scripts.benchmark` runs that script as a program;
`--tournament` adds a round-robin of duels (2 rounds per pair) and saves a
tournament file with wins/win-rates/draws, printed as a table.

### Testing

```bash
.venv/bin/python -m pytest tests/ -q
```
*Meaning:* run all 101 automated tests quietly (`-q` = compact output).
Expected last line: **`101 passed`** (~5 s). No internet, no window, no
Ollama needed. Useful variants:

```bash
.venv/bin/python -m pytest tests/test_safety.py -v       # one file, verbose
.venv/bin/python -m pytest tests/ -k checkpoint          # tests with 'checkpoint' in the name
```

### Forcing the LLM off (and on)

```bash
.venv/bin/python main.py --no-llm     # this run: no LLM at all
```
*Meaning:* same game, LLM planner disabled for this launch only. To enable
the LLM: start Ollama (`ollama serve`), pull a model (`ollama pull
qwen3:4b`), then turn it on in **Settings → LLM Planner → enabled → Save**
(or set `"enabled": true` in `config/default.json`). If the server is down,
agents just use the fallback strategy — nothing breaks.

### Where everything lands

| You ran | You get |
|---|---|
| `--train` | `models/dqn_default.pt`, `data/training/training_curves.png` |
| `--benchmark` | `data/benchmarks/cli-…/` (results.json, 2×CSV, 3×PNG) |
| `--tournament` | `data/benchmarks/tournament-…/results.json` |
| playing games | `data/history/games.jsonl` (Statistics screen reads it) |
| anything | `data/logs/app.log` |

---

# 7. The tests

`tests/` — 101 tests, all offline. Why they matter for your defence: they
prove the claims, and during development they caught **real bugs** (the
food-stacking bug, a wrong pygame API, path-resolution, three UI import
errors caught by the headless screen test).

| File | Covers |
|---|---|
| `conftest.py` | shared helpers: `make_engine(...)`, `make_env(...)`, a default config |
| `test_game.py` | movement, gradual growth, reversal ignored, wall/self deaths, head-on draws, tail-vacating, score = Σ food values, determinism, reset |
| `test_food.py` | catalog loads 5 types, probabilities normalized, weighted_choice, spawn/expire/refill, full-board skip, FoodStats |
| `test_pathfinding.py` | A\* optimal + detour + unreachable, BFS parity, flood-fill survivability |
| `test_safety.py` | each rejection reason, dead-end geometry, `select_safe_action` override, honest `no_safe_action` |
| `test_rl.py` | network shapes, replay ring behaviour, each reward component, training runs, **deterministic greedy eval**, checkpoint round-trip, CheckpointError |
| `test_llm.py` | a **FakeTransport** (programmer-programmable fake HTTP): availability, JSON mode, `<think>` stripping, retries, strict validation rejections, fallback determinism, snapshot hash |
| `test_decision_manager.py` | all 7 modes decide + log; LLM modes use fallback with `llm_calls == 0`; safety overrides a stubborn policy; the exact risk-gate formula |
| `test_benchmark.py` | structure, **identical seeds across agents**, honest `llm_available: false`, match fields, tick cap, cancellation |
| `test_persistence.py` | atomic round-trip, corrupt quarantine, damaged-JSONL skip, config clamping/recovery |
| `test_integration.py` | full episodes through every mode; observation shape; benchmark → saved artifacts → reload identical; history receives played games |
| `test_ui_headless.py` | every screen constructs, updates, draws, and takes an event — with SDL's **dummy** video driver (`SDL_VIDEODRIVER=dummy`), i.e. no monitor needed |

**How to add a test:** create `tests/test_whatever.py`, import from
`.conftest` (`from .conftest import make_engine`), write functions named
`test_*` using plain `assert`. Run pytest. That's the whole pattern.

---

# 8. How to change things (recipes)

All recipes assume you re-run the tests afterwards.

**Change food values / chances** — edit `config/foods.json` (values,
`probability`, `lifetime_seconds`, `growth`, `color`). Probabilities are
normalized automatically. No code changes.

**Change board size / speed / food count** — `config/default.json` →
`board.width/height`, `game.moves_per_second`, `game.max_foods`
(the UI Settings screen edits the same file).

**Change RL hyperparameters** — `config/default.json` → `rl`:
`hidden_sizes`, `gamma`, `learning_rate`, `eps_decay_steps`,
`double_dqn`, and every reward weight under `rl.reward`. Then retrain.

**Add a new agent** (the full recipe):
1. Create `src/snake_arena/agents/my_agent.py`:
   ```python
   from .base import AgentDecision, BaseAgent, StrategyHints
   class MyAgent(BaseAgent):
       name = "my_agent"
       def decide(self, state, snake_id, view=None, hints=None):
           # look at state/view, choose an Action
           return AgentDecision(agent=self.name, action=chosen, notes=["why"])
   ```
2. In `agents/decision_manager.py`: add `self.my_agent = MyAgent()` in
   `__init__`, and a branch in `_policy()` for a new mode name
   `"my_agent"`.
3. Register the name in `agents/decision_manager.py`'s `describe()` map and
   in `evaluation/benchmark.py` (`AGENT_LABELS`, maybe `LLM_MODES`), plus
   `config/default.json` → `benchmark.agents` if you want it by default.
4. Test it: `pytest`, then
   `python main.py --benchmark 2 --agents my_agent`.

**Make the LLM behave differently** — edit `SYSTEM_PROMPT` and
`fallback_strategy()` in `llm/strategist.py`, and/or the model in Settings
(`llm.model`). The schema in `llm/schemas.py` defines what answers are legal.

**Change colours / fonts** — `ui/theme.py` (`COLORS`, `Fonts`).

**Change the neural network** — `rl/network.py` (layer sizes come from
`rl.hidden_sizes` in config). If you change the **observation** (the 30
floats), you **must** bump `OBS_SIZE` in `game/environment.py` and retrain —
old checkpoints will be rejected by `load_checkpoint` (by design: it checks
`obs_size`).

**Change win/lose rules** — `game/engine.py` step 8 (termination checks).

**Tune difficulty for humans** — `game.moves_per_second`, `game.max_foods`,
trap probability in `config/foods.json`.

---

# 9. Glossary & likely questions

**Glossary**

- **A\*** — best-first pathfinding: explores cells ordered by
  `cost-so-far + estimated-distance-left`; optimal when the estimate never
  overestimates (Manhattan distance on a grid).
- **BFS** — breadth-first search: explores in waves; first path found is
  shortest (in steps).
- **DQN** — Deep Q-Network: a neural network that learns action values
  Q(s,a) from stored experience (Mnih et al., 2015).
- **Replay buffer** — memory of past moves, sampled randomly to train.
- **ε-greedy** — explore randomly with probability ε, else act greedily.
- **Target network** — a frozen copy of the Q-network that stabilizes
  training targets; synced every 1000 steps.
- **Double DQN** — decouple "pick best action" (online net) from "value it"
  (target net) to reduce overestimation.
- **Reward shaping** — small intermediate rewards (distance progress) to
  guide learning without changing the optimal policy.
- **Flood fill** — count all cells reachable from a point; used for
  dead-end detection.
- **Tail-awareness** — treating a vacating tail cell as free.
- **Atomic save** — write to a temp file, then rename: a crash never
  corrupts the previous file.
- **Posture/hints** — the LLM's output: strategy parameters, never moves.
- **Fallback strategy** — the deterministic stand-in when the LLM is
  missing/slow/wrong.

**Questions you may be asked, with short true answers**

- *"How does the LLM control the snake?"* — It doesn't, on purpose. Every
  few seconds it receives a JSON snapshot and returns a posture (strategy
  name + three 0–1 weights + optional food target). The classical/RL
  policies consume those weights, and the safety validator still overrides
  anything unsafe. If the LLM is absent/slow/wrong, a deterministic
  fallback supplies the same shape of hints.
- *"How do you compare agents fairly?"* — Identical per-index seeds: agent
  A's episode k and agent B's episode k start from the same board and food.
  Differences are then purely policy. A test asserts the seed lists match.
- *"Is your RL agent good?"* — The shipped checkpoint is trained for 40
  episodes, which is a functional check only; the tooling labels it
  untrained/undertrained everywhere, and longer training is one command.
- *"How do you know the safety checker works?"* — Unit tests place snakes in
  geometrically impossible situations and assert each rejection reason; one
  test forces the policy to propose a fatal move and asserts the pipeline
  overrides it.
- *"What happens if Ollama crashes mid-game?"* — Calls time out, retries
  happen inside limits, then the fallback strategy takes over; counters
  (`llm_failures`) record it; the game loop never blocks on the LLM (it's
  on a background thread with a cached latest result).
- *"Why pygame-ce?"* — Classic pygame has no package for Python 3.14 and its
  source build here had a broken font module; pygame-ce is the maintained
  drop-in replacement with official wheels.
- *"What was the hardest bug?"* — Food could spawn on top of existing food
  because the spawn check ignored occupied food cells; a test caught it and
  the fix was one parameter: `occupied_cells(include_food=True)`.

---

*End of guide. Related docs: [ARCHITECTURE.md](ARCHITECTURE.md) (structure),
[AI_ALGORITHMS.md](AI_ALGORITHMS.md) (agents), [RL_TRAINING.md](RL_TRAINING.md)
(training), [EXPERIMENTS.md](EXPERIMENTS.md) (benchmarks),
[REPORT.md](REPORT.md) (academic write-up).*
