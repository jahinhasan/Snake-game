"""Console-script entry point: importable, parses args, never opens a window.

These tests only exercise argument paths that exit before the UI starts
(--help / --version), so they run headless anywhere, CI included.
"""

from __future__ import annotations

import pytest

app = pytest.importorskip("snake_arena.app", reason="snake_arena not on sys.path")


def test_run_is_importable_and_callable() -> None:
    assert callable(app.run)


def test_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        app.run(["--help"])
    assert exc.value.code == 0
    assert "snake-arena" in capsys.readouterr().out


def test_version_exits_zero_without_window(capsys) -> None:
    assert app.run(["--version"]) == 0
    assert "v" in capsys.readouterr().out
