@echo off
rem Launch LLM-Enhanced Intelligent Snake Arena on Windows 10/11.
rem Uses the project virtual environment when present.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py %*
) else (
    echo note: no .venv found - using system python. Run setup_windows.ps1 first for an isolated install.
    python main.py %*
)
endlocal
