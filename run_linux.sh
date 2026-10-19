#!/usr/bin/env bash
# Launch LLM-Enhanced Intelligent Snake Arena on Linux.
# Uses .venv when present; falls back to the system python.
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
    PY=".venv/bin/python"
else
    PY="python3"
    echo "note: no .venv found - using system $PY (run ./setup_linux.sh first for a isolated install)." >&2
fi

exec "$PY" main.py "$@"
