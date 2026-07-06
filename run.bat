@echo off
setlocal
cd /d "%~dp0"

set "VENV=%~dp0.venv\Scripts"
if not exist "%VENV%\python.exe" set "VENV=%~dp0..\_program_development\_program_development_venv\Scripts"

if not exist "%VENV%\python.exe" (
    echo [ERROR] No Python environment found.
    echo Run setup.bat first to create one.
    pause
    exit /b 1
)

set "LAUNCHER=%VENV%\pythonw.exe"
if not exist "%LAUNCHER%" set "LAUNCHER=%VENV%\python.exe"

start "" /b "%LAUNCHER%" "%~dp0launch_demo.py"
exit /b 0
