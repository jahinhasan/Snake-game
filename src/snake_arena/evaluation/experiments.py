"""Experiment artifacts: run directories, JSON/CSV persistence, PNG charts.

Matplotlib (Agg backend) renders benchmark and training charts to files; the
dashboards draw live charts with Pygame primitives from the same recorded
data, so there is exactly one source of truth for every number shown.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe: never opens a GUI window
import matplotlib.pyplot as plt

from ..persistence.storage import run_id, save_json_atomic, versions_dict, write_csv
from ..utils import paths
from .benchmark import AGENT_LABELS
from .metrics import summary_rows

log = logging.getLogger("snake_arena.experiments")

ACCENT = "#58a6ff"
GREEN = "#7ee787"
GOLD = "#e3b341"
RED = "#f85149"
TEXT = "#c9d1d9"
GRID = "#30363d"


# -- benchmark artifacts ----------------------------------------------------------


def make_run_dir(prefix: str = "bench") -> Path:
    directory = paths.benchmark_dir() / run_id(prefix)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_benchmark_results(run_dir: Path, results: dict[str, Any]) -> dict[str, Path]:
    results.setdefault("run_id", run_dir.name)
    results.setdefault("meta", {}).setdefault("versions", versions_dict())
    json_path = save_json_atomic(run_dir / "results.json", results)
    csv_path = write_csv(
        run_dir / "summary.csv",
        [
            {"agent": name, **{k: v for k, v in summary.items() if k not in ("death_reasons", "food_by_type")}}
            for name, summary in results.get("agents", {}).items()
        ],
    )
    records_path = write_csv(run_dir / "records.csv", results.get("records", []))
    return {"json": json_path, "csv": csv_path, "records": records_path}


def plot_benchmark_charts(run_dir: Path, results: dict[str, Any]) -> list[Path]:
    """Bar charts (mean score with std-dev error bars, survival, foods)."""
    agents = results.get("agents", {})
    if not agents:
        return []
    names = list(agents.keys())
    labels = [AGENT_LABELS.get(n, n) for n in names]
    outputs: list[Path] = []

    plt.style.use("dark_background")

    # Mean score with std-dev bars.
    means = [agents[n]["mean_score"] for n in names]
    stds = [agents[n]["std_score"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=ACCENT, edgecolor=TEXT)
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_ylabel("Mean score (std-dev bars)")
    ax.set_title(f"Agent comparison - {results.get('run_id', '')}")
    ax.grid(axis="y", color=GRID, alpha=0.6)
    fig.tight_layout()
    path = run_dir / "chart_scores.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    outputs.append(path)

    # Survival rate.
    survival = [agents[n]["survival_rate"] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, survival, color=GREEN, edgecolor=TEXT)
    ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.set_ylabel("Survival rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Survival rate at episode end")
    ax.grid(axis="y", color=GRID, alpha=0.6)
    fig.tight_layout()
    path = run_dir / "chart_survival.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    outputs.append(path)

    # Food type totals (stacked, capped to first 5 types).
    food_types: list[str] = []
    for n in names:
        for ftype in agents[n]["food_by_type"]:
            if ftype not in food_types:
                food_types.append(ftype)
    if food_types:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bottom = [0.0] * len(names)
        colors = [GREEN, ACCENT, GOLD, "#bc8cff", RED]
        for i, ftype in enumerate(food_types[:5]):
            values = [agents[n]["food_by_type"].get(ftype, 0) for n in names]
            ax.bar(labels, values, bottom=bottom, label=ftype, color=colors[i % len(colors)], edgecolor=TEXT)
            bottom = [b + v for b, v in zip(bottom, values)]
        ax.set_ylabel("Food items collected (all episodes)")
        ax.legend()
        ax.set_title("Food collection by type")
        ax.grid(axis="y", color=GRID, alpha=0.6)
        fig.tight_layout()
        path = run_dir / "chart_foods.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        outputs.append(path)
    return outputs


def benchmark_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat display rows for the benchmark dashboard table."""
    rows = []
    for name, summary in results.get("agents", {}).items():
        row = {"agent": AGENT_LABELS.get(name, name)}
        row.update(summary_rows_row(summary))
        rows.append(row)
    return rows


def summary_rows_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "episodes": summary.get("episodes", 0),
        "mean_score": summary.get("mean_score", 0.0),
        "median_score": summary.get("median_score", 0.0),
        "std": summary.get("std_score", 0.0),
        "survival": f"{summary.get('survival_rate', 0.0) * 100:.0f}%",
        "mean_steps": summary.get("mean_steps", 0.0),
        "traps/1k": summary.get("trap_rate_per_1k_ticks", 0.0),
        "decision_ms": summary.get("mean_decision_ms", 0.0),
    }


def load_run(run_dir: Path) -> dict[str, Any] | None:
    from ..persistence.storage import load_json

    return load_json(Path(run_dir) / "results.json")


def list_runs(limit: int = 20) -> list[tuple[Path, dict[str, Any] | None]]:
    base = paths.benchmark_dir()
    if not base.exists():
        return []
    runs = sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)[:limit]
    return [(p, load_run(p)) for p in runs]


# -- training artifacts --------------------------------------------------------------


def moving_average(values: list[float], window: int = 20) -> list[float]:
    if not values:
        return []
    out = []
    running = 0.0
    window = max(1, window)
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        out.append(running / min(i + 1, window))
    return out


def plot_training_curves(history: dict[str, Any], out_dir: Path, name: str = "training_curves") -> Path | None:
    """Render real recorded training history to a 2x2 chart panel."""
    episodes = history.get("episodes", [])
    if not episodes:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rewards = [float(e.get("reward", 0.0)) for e in episodes]
    scores = [float(e.get("score", 0.0)) for e in episodes]
    losses = [float(e["loss"]) for e in episodes if e.get("loss") is not None]
    epsilons = [float(e.get("epsilon", 0.0)) for e in episodes]
    x = list(range(1, len(episodes) + 1))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0][0]
    ax.plot(x, rewards, color=ACCENT, alpha=0.35, linewidth=0.8)
    ax.plot(x, moving_average(rewards, 25), color=ACCENT, linewidth=2.0, label="moving avg (25)")
    ax.set_title("Episode reward")
    ax.legend()

    ax = axes[0][1]
    ax.plot(x, scores, color=GREEN, alpha=0.35, linewidth=0.8)
    ax.plot(x, moving_average(scores, 25), color=GREEN, linewidth=2.0, label="moving avg (25)")
    evals = history.get("evaluations", [])
    if evals:
        ex = [e.get("episode", i + 1) for i, e in enumerate(evals)]
        ax.plot(ex, [e.get("mean_score", 0.0) for e in evals], "o--", color=GOLD, label="greedy eval")
        ax.legend()
    ax.set_title("Episode score")

    ax = axes[1][0]
    if losses:
        ax.plot(list(range(1, len(losses) + 1)), losses, color="#bc8cff", linewidth=0.9)
        ax.set_title("Training loss (Huber)")
    else:
        ax.text(0.5, 0.5, "no loss recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Training loss (Huber)")

    ax = axes[1][1]
    ax.plot(x, epsilons, color=RED, linewidth=1.4)
    ax.set_title("Exploration rate (epsilon)")
    for ax in axes.flat:
        ax.grid(color=GRID, alpha=0.6)
        ax.set_xlabel("episode")

    fig.suptitle("DQN training progress (recorded data)", color=TEXT)
    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("Training chart saved: %s", path)
    return path
