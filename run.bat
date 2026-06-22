@echo off
REM ============================================================
REM  Launch the Nursing AI Assistant (UI/UX PoC)
REM  Uses the existing project virtual environment.
REM ============================================================
setlocal
cd /d "%~dp0"

REM  Use the local .venv created by setup.bat if present; otherwise fall back
REM  to the original project dev environment (one folder up).
set "VENV=%~dp0.venv\Scripts"
if not exist "%VENV%\python.exe" set "VENV=%~dp0..\_program_development\_program_development_venv\Scripts"

if not exist "%VENV%\python.exe" (
    echo [ERROR] No Python environment found.
    echo Run setup.bat first to create one, or activate your own and run:  streamlit run app.py
    echo.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM  Prevent the silent crash on Windows.
REM  PyTorch and scikit-learn each ship their own OpenMP runtime;
REM  loading the second one aborts the process right after
REM  "Loading weights: 100%". This makes them coexist.
REM ------------------------------------------------------------
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=1"

REM  Skip the first-run e-mail prompt and the usage-stats phone-home.
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "STREAMLIT_SERVER_HEADLESS=true"
set "PYTHONUNBUFFERED=1"

echo Make sure Ollama is running ^(ollama serve^) with the qwen2.5 model pulled.
echo Opening the app at http://localhost:8502 ...
echo.

REM ------------------------------------------------------------
REM  Open the browser in a SEPARATE window after a short delay.
REM  IMPORTANT: do NOT use "start /b" here. /b shares THIS console,
REM  and the delay command would take over the console's stdin; when
REM  it finishes, Streamlit (which watches stdin) sees end-of-input
REM  and shuts itself down a few seconds after start-up. A separate
REM  window has its own stdin, so Streamlit keeps running.
REM  ping is just a ~4s timer that never reads the keyboard.
REM ------------------------------------------------------------
start "Open browser" /min cmd /c "ping -n 5 127.0.0.1 >nul & start http://localhost:8502"

REM  Run Streamlit directly in this console (output prints live).
"%VENV%\python.exe" -m streamlit run "app.py" --server.port 8502 --server.fileWatcherType none

echo.
echo ============================================================
echo  Streamlit has stopped.
echo  If it closed unexpectedly, copy the messages above and send them.
echo ============================================================
pause
endlocal
