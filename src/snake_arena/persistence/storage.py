"""Safe file storage: atomic writes, corrupt-file recovery, JSONL streams.

Every save is write-to-temp-then-rename so a crash mid-write can never
destroy the previous valid file. Readers repair silently: a malformed file
is moved aside (``.corrupt``) and the default is returned.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger("snake_arena.storage")


def run_id(prefix: str = "run") -> str:
    """Unique, sortable experiment identifier: prefix-YYYYmmdd-HHMMSS-xxxx."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:4]}"


def save_json_atomic(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def load_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        log.error("Could not read %s (%s); quarantining as .corrupt", path, exc)
        try:
            path.replace(path.with_suffix(path.suffix + ".corrupt"))
        except OSError:
            pass
        return default


def append_jsonl(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return path


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip damaged lines rather than failing the read
    if limit is not None and limit >= 0:
        items = items[-limit:]
    return items


def write_csv(path: Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def versions_dict() -> dict[str, str]:
    """Reproducibility metadata embedded in every experiment artifact."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except Exception:  # noqa: BLE001
        info["numpy"] = "unavailable"
    try:
        import torch

        info["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        info["torch"] = "unavailable"
    try:
        import pygame

        info["pygame"] = pygame.version.ver
    except Exception:  # noqa: BLE001
        info["pygame"] = "unavailable"
    try:
        from .. import __version__

        info["snake_arena"] = __version__
    except Exception:  # noqa: BLE001
        info["snake_arena"] = "unknown"
    return info
