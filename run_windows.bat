@echo off
rem ===========================================================================
rem  ONE-COMMAND bootstrap for Windows 10/11.
rem
rem  On a FRESH clone this single file does everything:
rem    1. Finds Python 3.11+ (py launcher, PATH, or installs it via winget)
rem    2. Creates .venv and downloads ALL dependencies (CPU PyTorch,
rem       pygame-ce, numpy, matplotlib) - happens one time
rem    3. Installs Ollama via winget if missing and pulls qwen3:4b (~2.4 GB)
rem       so the LLM planner works - OPTIONAL, never blocks the game
rem    4. Starts the game
rem
rem  Every later run: jumps straight to step 4 and starts instantly.
rem
rem  Honest note: this project was developed on Linux; this script is
rem  straightforward batch but has not been executed on a real Windows
rem  machine. If anything fails, read the error, then see
rem  docs/WINDOWS_SETUP.md for the manual equivalent of each step.
rem ===========================================================================

setlocal EnableExtensions
cd /d "%~dp0"

rem ---- 1. find a Python 3.11+ (py launcher, then PATH, then winget) ---------
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    rem Python too old or missing: try a user-scope install through winget.
    winget --version >nul 2>&1
    if not errorlevel 1 (
        echo.
        echo  Python 3.11+ not found. Installing Python 3.12 via winget...
        winget install --id Python.Python.3.12 -e --scope user --silent ^
            --accept-source-agreements --accept-package-agreements
        if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" ^
            set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    )
)
if not defined PY (
    echo.
    echo  [ERROR] Python 3.11+ was not found and winget is unavailable.
    echo  1. Install Python from https://www.python.org/downloads/
    echo  2. In the installer, TICK "Add python.exe to PATH".
    echo  3. Run run_windows.bat again.
    echo.
    pause
    exit /b 1
)

rem ---- 2. first run? create .venv and install ALL requirements at once -------
if not exist ".venv\Scripts\python.exe" (
    echo ==========================================================
    echo   First run detected.
    echo   Creating .venv and downloading all dependencies
    echo   ^(CPU PyTorch, pygame-ce, numpy, matplotlib^).
    echo   This happens ONE time and needs a few minutes.
    echo ==========================================================
    %PY% -m venv .venv
    if errorlevel 1 goto :failed
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto :failed
    ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 goto :failed
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
    echo.
    echo   All dependencies installed successfully.
    echo.
)

rem ---- 3. LLM planner: Ollama + qwen3:4b (optional, best effort) -------------
set "OLLAMA_EXE="
ollama --version >nul 2>&1
if not errorlevel 1 set "OLLAMA_EXE=ollama"
if not defined OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" ^
    set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLLAMA_EXE (
    winget --version >nul 2>&1
    if not errorlevel 1 (
        echo.
        echo  Installing Ollama via winget for the LLM planner...
        winget install --id Ollama.Ollama -e --silent ^
            --accept-source-agreements --accept-package-agreements
        if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" ^
            set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    )
)
if not defined OLLAMA_EXE (
    echo.
    echo  [note] Ollama not installed - the LLM planner will use its
    echo         built-in fallback strategy. The game is fully playable.
    goto :launch
)

rem Make sure the Ollama server is running, then pull the model if needed.
"%OLLAMA_EXE%" list >nul 2>&1
if errorlevel 1 (
    echo  Starting the Ollama server...
    start "ollama-serve" /min "%OLLAMA_EXE%" serve
    timeout /t 4 /nobreak >nul
)
"%OLLAMA_EXE%" list 2>nul | findstr /c:"qwen3:4b" >nul
if errorlevel 1 (
    echo  Pulling qwen3:4b for the LLM planner - about 2.4 GB, ONE time...
    "%OLLAMA_EXE%" pull qwen3:4b
    if errorlevel 1 (
        echo  [note] Model pull failed - the game still runs; the LLM planner
        echo         falls back to its built-in strategy.
    )
) else (
    echo  qwen3:4b already present.
)

rem ---- 4. launch the game -----------------------------------------------------
:launch
".venv\Scripts\python.exe" main.py %*
exit /b 0

:failed
echo.
echo  [ERROR] Installation failed. Common fixes:
echo  - Check your internet connection, then run run_windows.bat again.
echo  - Make sure C: has a few GB free.
echo  - Manual retry: .venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
pause
exit /b 1
