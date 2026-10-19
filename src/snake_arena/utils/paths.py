"""Platform-safe application paths.

All runtime paths are derived from the package location (never from the
current working directory) and can be overridden with environment variables,
which keeps the app portable between Arch Linux and Windows.

``SNAKE_ARENA_DATA_DIR``   override for the runtime data directory
``SNAKE_ARENA_MODELS_DIR`` override for the model checkpoint directory
"""

from __future__ import annotations

import os
from pathlib import Path

# src/snake_arena/utils/paths.py -> project root is four levels up.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
ASSETS_DIR = PROJECT_ROOT / "assets"

ENV_DATA_DIR = "SNAKE_ARENA_DATA_DIR"
ENV_MODELS_DIR = "SNAKE_ARENA_MODELS_DIR"


def project_root() -> Path:
    """Return the project root directory (contains ``config/``, ``src/`` ...)."""
    return PROJECT_ROOT


def data_dir() -> Path:
    """Runtime data directory (game history, benchmarks, logs, charts)."""
    override = os.environ.get(ENV_DATA_DIR, "").strip()
    path = Path(override).expanduser() if override else PROJECT_ROOT / "data"
    return path


def models_dir() -> Path:
    """Model checkpoint directory."""
    override = os.environ.get(ENV_MODELS_DIR, "").strip()
    path = Path(override).expanduser() if override else PROJECT_ROOT / "models"
    return path


def logs_dir() -> Path:
    return data_dir() / "logs"


def history_dir() -> Path:
    return data_dir() / "history"


def benchmark_dir() -> Path:
    return data_dir() / "benchmarks"


def training_dir() -> Path:
    """Directory holding training curves and metrics exported as PNG/JSON."""
    return data_dir() / "training"


def default_config_path() -> Path:
    return CONFIG_DIR / "default.json"


def default_foods_path() -> Path:
    return CONFIG_DIR / "foods.json"


def ensure_runtime_dirs() -> None:
    """Create runtime directories on demand so users never do it manually."""
    for directory in (
        data_dir(),
        models_dir(),
        logs_dir(),
        history_dir(),
        benchmark_dir(),
        training_dir(),
    ):
        directory.mkdir(parents=True, exist_ok=True)
