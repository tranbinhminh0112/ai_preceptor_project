@echo off
REM ============================================================
REM  One-time setup for the Nursing AI Assistant
REM    1. creates a local virtual environment (.venv)
REM    2. installs the Python requirements
REM    3. downloads all the models the app needs
REM  Run this once after cloning. Re-running is safe.
REM ============================================================
setlocal
cd /d "%~dp0"

REM --- locate a Python interpreter ---
set "PYLAUNCHER="
where py >nul 2>nul && set "PYLAUNCHER=py -3"
if not defined PYLAUNCHER (
    where python >nul 2>nul && set "PYLAUNCHER=python"
)
if not defined PYLAUNCHER (
    echo [ERROR] Python was not found on your PATH.
    echo Install Python 3.11+ from https://www.python.org/downloads/ and re-run setup.bat
    echo Make sure to tick "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

REM --- create the virtual environment ---
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PYLAUNCHER% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists — reusing it.
)

set "VPY=.venv\Scripts\python.exe"

REM --- install dependencies ---
echo.
echo Installing Python packages (this may take several minutes)...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Package installation failed. See the messages above.
    pause
    exit /b 1
)

REM --- download / pull the models ---
echo.
echo Downloading models (Hugging Face + Ollama)...
echo Make sure Ollama is installed and running first (run "ollama serve" in another window).
echo.
set "KMP_DUPLICATE_LIB_OK=TRUE"
"%VPY%" setup_models.py

echo.
echo ============================================================
echo  Setup finished. Start the app any time with:  run.bat
echo ============================================================
pause
endlocal
