from __future__ import annotations

import logging
import sys

from ezframes.common.auth_tokens import validate_launch_ticket_from_env
from ezframes.common.config import AppPaths


log = logging.getLogger(__name__)


def _show_error(message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror("EzFrames Login Required", message, parent=root)
        root.destroy()
    except Exception:
        # pythonw may have no console, but keep a fallback for python.exe flows.
        print(message, file=sys.stderr)


def enforce_authorized_launch(interactive_error: bool = True, consume_ticket: bool = True) -> int:
    paths = AppPaths.default()
    validation = validate_launch_ticket_from_env(paths.state_dir, consume=consume_ticket)
    if validation.valid:
        return 0

    reason = validation.reason or "Unauthorized launch."
    log.warning("Blocking unauthorized app launch: %s", reason)
    if interactive_error:
        _show_error(
            "Please launch EzFrames from the launcher and sign in.\n\n"
            f"Reason: {reason}"
        )
    return 4
