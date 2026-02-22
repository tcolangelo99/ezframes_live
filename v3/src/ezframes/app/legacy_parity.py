from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from ezframes.app.auth_gate import enforce_authorized_launch
from ezframes.common.logging_utils import configure_logging
from ezframes.common.config import AppPaths


log = logging.getLogger(__name__)


MUTEX_NAME = "Global\\MyEzFramesMutex"


def _find_legacy_script() -> Path | None:
    env_path = os.environ.get("EZFRAMES_LEGACY_EZCUTIE", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        Path.cwd() / "ezcutie.py",
        Path.cwd().parent / "ezcutie.py",
        Path(__file__).resolve().parents[5] / "ezcutie.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _create_legacy_mutex():
    try:
        import win32event  # type: ignore
        import win32api  # type: ignore
    except Exception:
        return None

    handle = win32event.CreateMutex(None, False, MUTEX_NAME)
    if handle:
        log.info("Legacy compatibility mutex created")
    else:
        log.warning("Failed to create legacy mutex")

    class _Handle:
        def __init__(self, h):
            self._h = h

        def close(self):
            if self._h:
                try:
                    win32api.CloseHandle(self._h)
                except Exception:
                    pass

    return _Handle(handle)


def main() -> int:
    if os.environ.get("EZFRAMES_APP_AUTH_OK", "") != "1":
        auth_code = enforce_authorized_launch(interactive_error=True, consume_ticket=True)
        if auth_code != 0:
            return auth_code
        os.environ["EZFRAMES_APP_AUTH_OK"] = "1"

    paths = AppPaths.default()
    paths.ensure_layout()
    configure_logging(paths.logs_dir)

    legacy_script = _find_legacy_script()
    if legacy_script is None:
        log.warning("Legacy ezcutie.py not found; falling back to new UI.")
        from ezframes.app.ui_controller import main as new_main

        return int(new_main())

    mutex = _create_legacy_mutex()
    try:
        cmd = [sys.executable, str(legacy_script)]
        log.info("Launching legacy parity UI: %s", cmd)
        proc = subprocess.Popen(cmd, cwd=str(legacy_script.parent), shell=False)
        return int(proc.wait())
    finally:
        if mutex:
            mutex.close()
