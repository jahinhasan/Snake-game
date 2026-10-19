#!/usr/bin/env bash
# Setup for LLM-Enhanced Intelligent Snake Arena on Linux (tested on Arch Linux).
# Creates a virtual environment and installs CPU-only PyTorch plus dependencies.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: $PYTHON not found. Install python (Arch: sudo pacman -S python) or set PYTHON=/path/to/python." >&2
    exit 1
fi

echo "== Python version check =="
"$PYTHON" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"python >= 3.10 required, found {sys.version.split()[0]}")
print(f"python {sys.version.split()[0]} - OK")
EOF

if [ ! -d .venv ]; then
    echo "== Creating virtual environment (.venv) =="
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "== Installing CPU-only PyTorch (small download) =="
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "== Installing project dependencies =="
pip install -r requirements.txt

if [ "${1:-}" = "--dev" ]; then
    echo "== Installing development dependencies (pytest) =="
    pip install -r requirements-dev.txt
fi

echo
echo "Setup complete. Start the game with:   ./run_linux.sh"
echo "Environment self-test:                 python main.py --check"
echo "Optional local LLM planner:            install Ollama (https://ollama.com) and run: ollama pull qwen3:4b"
echo "                                       (the game is fully playable without it)"
