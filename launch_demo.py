from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8502
APP_URL = f"http://localhost:{PORT}"
PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "data"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "streamlit.log"
ERROR_LOG_PATH = LOG_DIR / "launcher_error.log"


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def launch_streamlit():
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(PORT),
        "--server.fileWatcherType",
        "none",
    ]

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)

    log_handle = LOG_PATH.open("a", encoding="utf-8")
    subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=False,
    )


def wait_for_server(timeout_seconds: int = 45) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if port_is_open(HOST, PORT):
            return True
        time.sleep(1)
    return False


def main():
    try:
        if not port_is_open(HOST, PORT):
            launch_streamlit()
            wait_for_server()
        webbrowser.open(APP_URL)
    except Exception as exc:
        ERROR_LOG_PATH.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
