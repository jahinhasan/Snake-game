# LLM-Enhanced Intelligent Snake Arena

A polished, research-grade Snake game where classical search agents, a PyTorch
DQN, and an optional locally-hosted LLM strategic planner (Ollama) play on the
same grid — every AI action passes through an independent safety validator
before it moves a pixel.

Built as an academic project: real algorithms, honest reporting (untrained
networks are labelled as such; LLM fallback is labelled as fallback), and
benchmark artifacts (JSON/CSV/PNG) generated from actual runs, never typed in
by hand.

---

## Features

**Game**
- Simultaneous-turn grid arena (default 24×18, configurable 10×10 – 40×30),
  wall / self / opponent collisions, head-on collisions eliminate both snakes.
- 5 food types with spawn probabilities, lifetimes, risk tiers, and scoring:
  normal (+10, 55%), bonus (+25, 20%), golden (+50, 13%), special (+100, 7%,
  grows 2 segments), trap (−20, 5%).
- Modes: human vs AI, AI vs AI, and a round-robin tournament runner.
- Pause, restart, speed slider, seed control, keyboard controls
  (arrows / WASD, Space = pause).

**AI**
- **Rule-based** — weighted heuristics (food value, distance, danger, trap
  avoidance).
- **BFS** — shortest safe path to the best food.
- **A\*** — optimal pathing with flood-fill dead-end avoidance.
- **DQN (PyTorch)** — 30→128→128→4 MLP, Double DQN, Huber loss, epsilon-greedy
  1.0 → 0.05; honest about being untrained until it is actually trained.
- **LLM planner** — a local LLM (Ollama, default `qwen3:4b`) issues strategic
  postures (greedy / balanced / defensive / survival, food target, risk
  tolerance) every few seconds. It never blocks the game loop: cached,
  rate-limited, background-threaded, with a deterministic fallback strategy
  when the server is absent or slow.
- **Hybrid** — LLM posture modulates the weights between A\* and DQN votes;
  documented deterministic arbitration.
- **Safety checker (mandatory)** — every proposed action from every agent is
  independently validated (wall, reversal, self, opponent, dead-end
  flood-fill) and overridden with the best safe alternative. Interventions are
  counted and displayed.

**UI (7 screens)**
Menu · Mode select · Arena (with AI diagnostics overlays: chosen path,
danger zones, target, rejected moves, live Q-values) · RL training dashboard ·
Benchmark dashboard (solo runs + tournaments, progress bar, cancellable) ·
Settings · Statistics & history.

**Evaluation**
- Identical per-index seeds across agents, so score differences are policy
  differences, not luck.
- Artifacts per run: `results.json`, `summary.csv`, `records.csv`, and score /
  survival / food-type charts.
- Post-game LLM analysis (clearly marked as model-generated) when Ollama is
  available; a deterministic summary otherwise.

**Reliability**
- Atomic config/history saves with corrupt-file quarantine and recovery.
- The game runs fully without Ollama — LLM off is the default configuration.
- 101 automated tests (engine, food, pathfinding, safety, RL, LLM client with
  a fake transport, decision manager, benchmark, persistence, integration,
  headless UI smoke tests): `pytest` passes with no network and no Ollama.

---

## Requirements

- Python **3.10+** (developed and tested on **3.14**)
- CPU is sufficient for everything; CUDA torch works automatically if present
- Optional: [Ollama](https://ollama.com) with `ollama pull qwen3:4b`

This project uses **pygame-ce** (the community-maintained drop-in for pygame,
imported as `import pygame`) because it ships prebuilt wheels for current
Python versions where classic pygame may not.

## Quick start (Arch Linux / any Linux / macOS)

```bash
./setup_linux.sh     # creates .venv, installs CPU torch + requirements
./run_linux.sh       # or: .venv/bin/python main.py
```

CLI tools:

```bash
.venv/bin/python main.py --check          # environment self-test (no window)
.venv/bin/python main.py --train 400      # headless DQN training
.venv/bin/python main.py --benchmark 20   # headless benchmark, all agents
.venv/bin/python main.py --benchmark 10 --agents rule_based,astar
.venv/bin/python main.py --no-llm         # force the LLM planner off
```

Artifacts land in `data/benchmarks/<run-id>/` and `data/training/`;
checkpoints in `models/`.

## Windows

See [docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md) and use
`setup_windows.ps1` / `run_windows.bat`. **Honest disclosure:** these scripts
were authored for Windows but this project was developed and tested on Arch
Linux only — the Windows path has not been executed by the author.

## Enabling the LLM planner

```bash
ollama serve                     # if not already running as a service
ollama pull qwen3:4b
```

Then either set `llm.enabled = true` in Settings (or `config/default.json`),
or set the environment variable `SNAKE_ARENA_LLM=1` style overrides — the
simplest path is the Settings screen: *LLM Planner → enabled → Save*. If the
server or model is missing, agents silently use the documented deterministic
fallback and diagnostics show `strategy_source: fallback`; nothing crashes and
nothing is fabricated.

---

## Project layout

```
main.py                  entry point (UI by default; --check/--train/--benchmark)
config/default.json      validated, self-healing configuration
config/foods.json        food-type catalog
scripts/                 check_environment, train, benchmark CLIs
src/snake_arena/
  game/                  engine, state, snake, collision, food, RL environment
  agents/                rule_based, bfs, astar, rl_agent, decision_manager, factory
  llm/                   ollama_client, strategist (prompts, snapshot, fallback), schemas, analysis
  safety/                independent action validator
  rl/                    network, trainer, rewards, checkpoints
  evaluation/            benchmark runner, metrics, experiment artifacts
  persistence/           atomic storage, game history
  ui/                    app shell + 7 screens, widgets, theme
  utils/                 paths, logging helpers
tests/                   101 tests, fully offline
docs/                    architecture, algorithms, RL guide, experiments, Windows setup, report
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Measured on the development machine: **101 passed** in ~5.5 s, no network, no
GPU, no Ollama required.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — modules, data flow, threading
- [docs/AI_ALGORITHMS.md](docs/AI_ALGORITHMS.md) — each agent + safety layer + LLM protocol
- [docs/RL_TRAINING.md](docs/RL_TRAINING.md) — network, rewards, training recipe
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — methodology + how to reproduce
- [docs/REPORT.md](docs/REPORT.md) — academic write-up (abstract → limitations → future work)

## Known limitations (honest list)

- Trained here for only **40 episodes** (a functional check, not convergence):
  the shipped checkpoint plays visibly worse than A\*. Longer training is a
  command away and is reported honestly by the tooling.
- The LLM path was verified with a programmable fake transport + fallback
  logic; **Ollama was not installed on the development machine**, so no
  LLM-generated gameplay or narratives are included in the results.
- **Windows scripts are untested** (development happened on Arch Linux).
- Single machine, local-only: no networking, no accounts.
# Snake-game
