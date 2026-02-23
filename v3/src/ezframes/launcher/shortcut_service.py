from __future__ import annotations

import logging
import os
import subprocess
import shutil
from pathlib import Path

from ezframes.common.config import AppPaths


log = logging.getLogger(__name__)


class ShortcutService:
    def __init__(self, paths: AppPaths):
        self.paths = paths

    @staticmethod
    def _is_windows() -> bool:
        return os.name == "nt"

    def _native_launcher_path(self) -> Path:
        return self.paths.install_root / "EzFramesLauncher.exe"

    def _staged_launcher_path(self) -> Path:
        return self.paths.app_dir / "EzFramesLauncher.exe"

    def ensure_native_launcher(self) -> bool:
        if not self._is_windows():
            return False

        native = self._native_launcher_path()
        if native.exists() and native.is_file():
            return True

        staged = self._staged_launcher_path()
        if not staged.exists() or not staged.is_file():
            log.info("Native launcher is missing and no staged copy was found at %s", staged)
            return False

        try:
            native.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, native)
            log.info("Staged native launcher copied to %s", native)
            return True
        except Exception as exc:
            log.warning("Failed to stage native launcher to install root: %s", exc)
            return False

    @staticmethod
    def _shortcut_paths() -> list[Path]:
        appdata = Path(os.environ.get("APPDATA", "")).expanduser()
        userprofile = Path(os.environ.get("USERPROFILE", "")).expanduser()

        start_menu_group = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "EzFrames" / "EzFrames.lnk"
        start_menu_flat = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "EzFrames.lnk"
        desktop = userprofile / "Desktop" / "EzFrames.lnk"

        return [start_menu_group, start_menu_flat, desktop]

    def _write_shortcut(self, shortcut_path: Path, target_exe: Path) -> None:
        import win32com.client  # type: ignore

        shortcut_path.parent.mkdir(parents=True, exist_ok=True)
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortcut(str(shortcut_path))
        shortcut.TargetPath = str(target_exe)
        shortcut.Arguments = ""
        shortcut.WorkingDirectory = str(self.paths.install_root)
        shortcut.IconLocation = str(target_exe)
        shortcut.Description = "Launch EzFrames"
        shortcut.Save()

    @staticmethod
    def _ps_quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def _write_shortcut_powershell(self, shortcut_path: Path, target_exe: Path) -> None:
        shortcut_path.parent.mkdir(parents=True, exist_ok=True)
        script = (
            "$ws=New-Object -ComObject WScript.Shell; "
            f"$s=$ws.CreateShortcut({self._ps_quote(str(shortcut_path))}); "
            f"$s.TargetPath={self._ps_quote(str(target_exe))}; "
            "$s.Arguments=''; "
            f"$s.WorkingDirectory={self._ps_quote(str(self.paths.install_root))}; "
            f"$s.IconLocation={self._ps_quote(str(target_exe))}; "
            "$s.Description='Launch EzFrames'; "
            "$s.Save();"
        )
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            creationflags=creationflags,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"powershell exit code {completed.returncode}"
            raise RuntimeError(detail)

    def repair_shortcuts(self) -> None:
        if not self._is_windows():
            return
        if not self.ensure_native_launcher():
            return

        launcher = self._native_launcher_path()
        use_pywin32 = False
        try:
            import win32com.client  # type: ignore  # noqa: F401
            use_pywin32 = True
        except Exception as exc:
            log.info("pywin32 COM unavailable, using PowerShell fallback for shortcuts: %s", exc)

        for shortcut in self._shortcut_paths():
            try:
                if use_pywin32:
                    self._write_shortcut(shortcut, launcher)
                else:
                    self._write_shortcut_powershell(shortcut, launcher)
                log.info("Shortcut repaired: %s -> %s", shortcut, launcher)
            except Exception as exc:
                log.warning("Failed to repair shortcut %s: %s", shortcut, exc)
