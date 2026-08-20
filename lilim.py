"""lilim — desktop launcher.

    python lilim.py

Opens the workbench in a real application window: its own title bar, no
browser, no URL to copy. Streamlit still does the serving, but on a private
port bound to loopback that nothing else is told about, and the window is a
native OS window driven by pywebview (WebView2 on Windows, WebKit elsewhere).

WHY A WRAPPER RATHER THAN A REWRITE
-----------------------------------
The workbench and its sixteen Plotly figures are verified working. Rebuilding
them in a native toolkit would throw that away to change nothing the user can
see except the window frame. This wrapper is ~200 lines and keeps every one of
them.

THE PART THAT IS EASY TO GET WRONG
----------------------------------
Streamlit spawns a child process. `Popen.terminate()` on the parent leaves that
child running, holding the port, so the next launch silently fails or attaches
to a stale server. Teardown here kills the whole process tree and is registered
on every exit path, including the exception one.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKBENCH = os.path.join(HERE, "workbench.py")
GEOMETRY_FILE = os.path.join(HERE, ".lilim-window.json")

WINDOW_TITLE = "lilim — SECPI workbench"

# The minimum is not cosmetic: below roughly this width the layout enters the
# regime where side-by-side charts squeeze instead of stacking. The stylesheet
# has breakpoints that handle narrower windows gracefully, but starting wide
# and refusing to go absurdly narrow avoids the problem rather than managing it.
DEFAULT_SIZE = (1440, 960)
MIN_SIZE = (900, 640)

STARTUP_TIMEOUT_S = 90

_server: subprocess.Popen | None = None


# ----------------------------------------------------------------- utilities
def free_port() -> int:
    """Claim a port from the OS, then release it for Streamlit to bind.

    There is a small race between releasing and rebinding. It is acceptable
    here: the alternative is a fixed port, which collides with a previous
    instance far more often than this races.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def port_is_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def load_geometry() -> dict:
    """Restore the window's last size and position, if it is still sensible."""
    try:
        with open(GEOMETRY_FILE, encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return {}

    width = int(saved.get("width", DEFAULT_SIZE[0]))
    height = int(saved.get("height", DEFAULT_SIZE[1]))
    geometry = {
        "width": max(MIN_SIZE[0], min(width, 10000)),
        "height": max(MIN_SIZE[1], min(height, 10000)),
    }

    # Only restore a position if it is plausibly on-screen. A window saved on a
    # monitor that is no longer attached would otherwise reopen invisibly, and
    # the app would look like it failed to start.
    x, y = saved.get("x"), saved.get("y")
    if isinstance(x, int) and isinstance(y, int) and -50 <= x < 10000 and -50 <= y < 10000:
        geometry["x"], geometry["y"] = x, y
    return geometry


def save_geometry(window) -> None:
    """Persist size and position. Never allowed to break shutdown."""
    try:
        payload = {"width": int(window.width), "height": int(window.height)}
        try:
            payload["x"], payload["y"] = int(window.x), int(window.y)
        except Exception:
            pass  # Position is not exposed on every backend; size still is.
        with open(GEOMETRY_FILE, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception:
        pass


# -------------------------------------------------------------- the server
def start_server(port: int) -> subprocess.Popen:
    """Launch Streamlit headless on `port`, bound to loopback only."""
    command = [
        sys.executable, "-m", "streamlit", "run", WORKBENCH,
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",          # do not open a browser of its own
        "--server.fileWatcherType", "none",   # nothing to hot-reload in a shipped app
        "--browser.gatherUsageStats", "false",
    ]
    kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
              "text": True, "cwd": HERE}
    if os.name == "nt":
        # Its own process group, so the whole tree can be signalled at once.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(command, **kwargs)


def wait_for_server(process: subprocess.Popen, port: int) -> None:
    """Block until the port answers, or fail with Streamlit's own output.

    Reporting the child's stdout on failure is the difference between "it
    hangs" and "it says what is wrong".
    """
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        if process.poll() is not None:
            output = (process.stdout.read() if process.stdout else "") or "(no output)"
            raise RuntimeError(
                f"Streamlit exited before it started serving.\n\n{output.strip()}")
        if port_is_open(port):
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"Streamlit did not start within {STARTUP_TIMEOUT_S}s on port {port}.")


def stop_server() -> None:
    """Kill the server and everything it spawned. Safe to call repeatedly."""
    global _server
    process, _server = _server, None
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            # taskkill /T is what actually reaches Streamlit's child process;
            # terminate() alone leaves it holding the port.
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, check=False)
        else:
            process.terminate()
        process.wait(timeout=10)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


# ------------------------------------------------------------------- main
def main() -> int:
    global _server

    try:
        import webview
    except ImportError:
        print("pywebview is not installed.\n"
              "  pip install -r requirements.txt\n\n"
              "Or run in a browser instead:\n"
              "  streamlit run workbench.py", file=sys.stderr, flush=True)
        return 1

    if not os.path.exists(WORKBENCH):
        print(f"Cannot find {WORKBENCH}", file=sys.stderr, flush=True)
        return 1

    port = free_port()
    print(f"lilim - starting on 127.0.0.1:{port} ...", flush=True)

    _server = start_server(port)
    atexit.register(stop_server)

    try:
        wait_for_server(_server, port)
    except RuntimeError as exc:
        stop_server()
        print(f"\n{exc}", file=sys.stderr, flush=True)
        return 1

    # pywebview ships this OFF, and its Windows backend does not merely ignore a
    # download -- `on_download_starting` answers with `args.Cancel = True`, so
    # the browser download button produces no file, no dialog and no error. The
    # app's own save control writes to `lilim/exports/` and does not depend on
    # this, but the download button beside it is dead without it.
    webview.settings["ALLOW_DOWNLOADS"] = True

    geometry = load_geometry()
    window = webview.create_window(
        WINDOW_TITLE,
        f"http://127.0.0.1:{port}",
        width=geometry.get("width", DEFAULT_SIZE[0]),
        height=geometry.get("height", DEFAULT_SIZE[1]),
        x=geometry.get("x"),
        y=geometry.get("y"),
        min_size=MIN_SIZE,
        background_color="#F7F5F0",   # matches the app's paper ground
        text_select=True,
    )

    window.events.closing += lambda: save_geometry(window)

    print("lilim - window open. Close it to quit.", flush=True)
    try:
        webview.start()
    finally:
        stop_server()
    return 0


if __name__ == "__main__":
    sys.exit(main())
