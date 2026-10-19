"""Central logging configuration.

Writes to console and to a rotating file under the runtime data directory so
diagnostics survive after the session ends.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from .paths import ensure_runtime_dirs, logs_dir

_CONFIGURED = False
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    env_level = os.environ.get("SNAKE_ARENA_LOG_LEVEL", "").upper()
    resolved = (level or env_level or "INFO").upper()
    numeric = getattr(logging, resolved, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(console)

    try:
        ensure_runtime_dirs()
        file_handler = RotatingFileHandler(
            logs_dir() / "snake_arena.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(file_handler)
    except OSError:  # pragma: no cover - unwritable disk is non-fatal
        root.warning("Could not open log file; logging to console only.")

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
