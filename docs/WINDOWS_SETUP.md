# Windows Setup

> **Honest disclosure:** this project was developed and tested on **Arch
> Linux only**. The Windows scripts (`setup_windows.ps1`,
> `run_windows.bat`) were written for Windows and follow the same steps as
> the Linux scripts, but **have not been executed on a Windows machine by
> the author**. If something breaks, the failure will be in the scripts —
> the Python code itself is cross-platform (pure Python + well-known
> wheels), and the manual steps below are the reliable fallback.

## Automated (PowerShell)

```powershell
cd snake-game
Set-ExecutionPolicy -Scope Process Bypass   # if needed, this session only
.\setup_windows.ps1
.\run_windows.bat
```

`setup_windows.ps1` does, in order:

1. find `py` (Python launcher) and verify Python ≥ 3.10;
2. create `.venv` if missing;
3. install CPU-only PyTorch from the official CPU index
   (`https://download.pytorch.org/whl/cpu`);
4. `pip install -r requirements.txt`.

`run_windows.bat` activates `.venv` and runs `python main.py`.

## Manual steps (equivalent, and the reference if scripts fail)

```powershell
py -3 --version                 # need >= 3.10
py -3 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python main.py --check          # environment self-test: aim for ALL CHECKS PASSED
python main.py                  # launch the game
```

## Windows-specific notes

- **pygame-ce** ships Windows wheels for all supported Pythons; no compiler
  needed. (Classic `pygame` may lack wheels for brand-new Python versions —
  that is why this project pins `pygame-ce`.)
- **PyTorch CPU** wheel is ~200 MB; the CPU index avoids a multi-GB CUDA
  download. GPU is optional and auto-detected if you install a CUDA build.
- **Ollama on Windows:** install from <https://ollama.com/download>, then
  `ollama pull qwen3:4b`. The game detects the server at
  `http://127.0.0.1:11434`; enable the planner in Settings → *LLM Planner*.
  Without it, everything still runs (fallback strategy).
- **Paths:** all runtime data goes to `data\` next to the project; the code
  uses `pathlib` and never hard-codes absolute paths.
- **Console encoding:** if the CLI prints garbled box characters, run
  `chcp 65001` first, or use Windows Terminal.
- **Firewall prompt:** Ollama binds localhost only; no inbound access is
  needed. The game itself makes no network connections unless the LLM
  planner is enabled (then: localhost to Ollama only).

## Verification checklist

```powershell
python main.py --check          # expect: ALL CHECKS PASSED
python -m pytest tests/ -q      # expect: all tests pass (no network needed)
python main.py --benchmark 2 --agents rule_based,astar
```

The last command should create `data\benchmarks\cli-<timestamp>-<id>\` with
`results.json`, two CSVs, and three PNG charts — that is the proof the whole
stack works on your machine.
