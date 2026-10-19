# Setup for LLM-Enhanced Intelligent Snake Arena on Windows 10/11.
#
# NOTE (honest disclosure): this script was authored for Windows but the
# project was developed and tested on Arch Linux only - see WINDOWS_SETUP.md
# and README.md. If a step fails, run the commands manually and report the
# exact error. Requires Python 3.10+ from https://www.python.org/downloads/
# (check "Add python.exe to PATH" during install).

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { $py = "py -3" } else {
        Write-Error "Python not found. Install Python 3.10+ from python.org and re-run."
    }
} else { $py = "python" }

Write-Host "== Python version check =="
Invoke-Expression "$py -c ""import sys; print('python ' + sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 10) else 1)"""
if ($LASTEXITCODE -ne 0) { Write-Error "Python >= 3.10 is required." }

if (-not (Test-Path .venv)) {
    Write-Host "== Creating virtual environment (.venv) =="
    Invoke-Expression "$py -m venv .venv"
}

$pip = ".\.venv\Scripts\python.exe -m pip"
Write-Host "== Installing CPU-only PyTorch (small download) =="
Invoke-Expression "$pip install --upgrade pip"
Invoke-Expression "$pip install torch --index-url https://download.pytorch.org/whl/cpu"

Write-Host "== Installing project dependencies =="
Invoke-Expression "$pip install -r requirements.txt"

Write-Host ""
Write-Host "Setup complete. Start the game with:   .\run_windows.bat"
Write-Host "Environment self-test:                 .\.venv\Scripts\python.exe main.py --check"
Write-Host "Optional local LLM planner:            install Ollama for Windows (https://ollama.com)"
Write-Host "                                       then: ollama pull qwen3:4b"
Write-Host "                                       (the game is fully playable without it)"
