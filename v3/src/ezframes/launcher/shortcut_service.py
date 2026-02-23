from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
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

    def _shortcut_prefs_path(self) -> Path:
        return self.paths.state_dir / "shortcut_prefs.v1.json"

    def _load_shortcut_prefs(self) -> dict[str, object]:
        path = self._shortcut_prefs_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_shortcut_prefs(self, prefs: dict[str, object]) -> None:
        try:
            path = self._shortcut_prefs_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("Failed to persist shortcut prefs: %s", exc)

    @staticmethod
    def _file_needs_refresh(source: Path, target: Path) -> bool:
        try:
            s = source.stat()
            t = target.stat()
            if s.st_size != t.st_size:
                return True
            return int(s.st_mtime) != int(t.st_mtime)
        except Exception:
            return True

    def ensure_native_launcher(self) -> bool:
        if not self._is_windows():
            return False

        native = self._native_launcher_path()
        staged = self._staged_launcher_path()
        staged_exists = staged.exists() and staged.is_file()
        native_exists = native.exists() and native.is_file()

        if not staged_exists:
            if native_exists:
                return True
            log.info("Native launcher is missing and no staged copy was found at %s", staged)
            return False

        if native_exists and not self._file_needs_refresh(staged, native):
            return True

        try:
            native.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, native)
            log.info("Staged native launcher copied/refreshed at %s", native)
            return True
        except Exception as exc:
            log.warning("Failed to stage native launcher to install root: %s", exc)
            return False

    @staticmethod
    def _start_menu_shortcut_paths() -> list[Path]:
        appdata = Path(os.environ.get("APPDATA", "")).expanduser()
        start_menu_group = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "EzFrames" / "EzFrames.lnk"
        start_menu_flat = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "EzFrames.lnk"
        return [start_menu_group, start_menu_flat]

    @staticmethod
    def _desktop_shortcut_path() -> Path:
        userprofile = Path(os.environ.get("USERPROFILE", "")).expanduser()
        return userprofile / "Desktop" / "EzFrames.lnk"

    def _should_manage_desktop_shortcut(self) -> bool:
        # Respect user choice: only maintain desktop shortcut if it already exists.
        # This prevents recreating it after a user deletes it.
        desktop = self._desktop_shortcut_path()
        if desktop.exists():
            return True
        return os.environ.get("EZFRAMES_FORCE_DESKTOP_SHORTCUT", "").strip() == "1"

    def _shortcut_icon_location(self, target_exe: Path) -> str:
        candidates: list[Path] = [
            self.paths.assets_dir / "icons" / "launcher_icon.ico",
            self.paths.assets_dir / "icons" / "ezframes_icon.ico",
            self.paths.install_root / "icons" / "launcher_icon.ico",
            self.paths.install_root / "icons" / "ezframes_icon.ico",
        ]
        for root in self.paths.source_roots():
            candidates.append(root / "icons" / "launcher_icon.ico")
            candidates.append(root / "icons" / "ezframes_icon.ico")
            candidates.append(root / "launcher_icon.ico")
            candidates.append(root / "ezframes_icon.ico")
        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    return str(candidate)
            except Exception:
                continue
        return str(target_exe)

    def _icon_candidates(self) -> list[Path]:
        candidates: list[Path] = [
            self.paths.assets_dir / "icons" / "ezframes_icon.ico",
            self.paths.assets_dir / "icons" / "launcher_icon.ico",
            self.paths.install_root / "icons" / "ezframes_icon.ico",
            self.paths.install_root / "icons" / "launcher_icon.ico",
        ]
        for root in self.paths.source_roots():
            candidates.append(root / "icons" / "ezframes_icon.ico")
            candidates.append(root / "icons" / "launcher_icon.ico")
            candidates.append(root / "ezframes_icon.ico")
            candidates.append(root / "launcher_icon.ico")
        return candidates

    def _apply_window_icon(self, window) -> None:
        for candidate in self._icon_candidates():
            try:
                if candidate.exists():
                    window.iconbitmap(str(candidate))
                    return
            except Exception:
                continue

    @staticmethod
    def _pywin32_available() -> bool:
        try:
            import win32com.client  # type: ignore  # noqa: F401

            return True
        except Exception:
            return False

    def _write_shortcut(self, shortcut_path: Path, target_exe: Path) -> None:
        import win32com.client  # type: ignore

        shortcut_path.parent.mkdir(parents=True, exist_ok=True)
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortcut(str(shortcut_path))
        shortcut.TargetPath = str(target_exe)
        shortcut.Arguments = ""
        shortcut.WorkingDirectory = str(self.paths.install_root)
        shortcut.IconLocation = self._shortcut_icon_location(target_exe)
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
            f"$s.IconLocation={self._ps_quote(self._shortcut_icon_location(target_exe))}; "
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

    def _create_desktop_shortcut(self, use_pywin32: bool) -> bool:
        launcher = self._native_launcher_path()
        desktop = self._desktop_shortcut_path()
        try:
            if use_pywin32:
                self._write_shortcut(desktop, launcher)
            else:
                self._write_shortcut_powershell(desktop, launcher)
            log.info("Desktop shortcut created: %s -> %s", desktop, launcher)
            return True
        except Exception as exc:
            log.warning("Failed to create desktop shortcut %s: %s", desktop, exc)
            return False

    def prompt_desktop_shortcut_once(self) -> None:
        if not self._is_windows():
            return
        if not self.ensure_native_launcher():
            return

        desktop = self._desktop_shortcut_path()
        prefs = self._load_shortcut_prefs()
        prompted = bool(prefs.get("desktop_prompted", False))
        if desktop.exists():
            if not prompted:
                prefs["desktop_prompted"] = True
                prefs["desktop_opt_in"] = True
                self._save_shortcut_prefs(prefs)
            return
        if prompted:
            return

        use_pywin32 = self._pywin32_available()
        if not use_pywin32:
            log.info("pywin32 COM unavailable, using PowerShell fallback for shortcuts.")

        try:
            import tkinter as tk
            from tkinter import messagebox
        except Exception as exc:
            log.info("Desktop shortcut prompt unavailable (tkinter import failed): %s", exc)
            return

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        self._apply_window_icon(root)
        create_shortcut = messagebox.askyesno(
            "EzFrames Shortcut",
            "Create a desktop shortcut for EzFrames?",
            parent=root,
        )
        root.destroy()

        prefs["desktop_prompted"] = True
        prefs["desktop_opt_in"] = bool(create_shortcut)
        self._save_shortcut_prefs(prefs)

        if create_shortcut:
            self._create_desktop_shortcut(use_pywin32=use_pywin32)

    def repair_shortcuts(self) -> None:
        if not self._is_windows():
            return
        if not self.ensure_native_launcher():
            return

        launcher = self._native_launcher_path()
        use_pywin32 = self._pywin32_available()
        if not use_pywin32:
            log.info("pywin32 COM unavailable, using PowerShell fallback for shortcuts.")

        shortcut_targets = list(self._start_menu_shortcut_paths())
        if self._should_manage_desktop_shortcut():
            shortcut_targets.append(self._desktop_shortcut_path())

        for shortcut in shortcut_targets:
            try:
                if use_pywin32:
                    self._write_shortcut(shortcut, launcher)
                else:
                    self._write_shortcut_powershell(shortcut, launcher)
                log.info("Shortcut repaired: %s -> %s", shortcut, launcher)
            except Exception as exc:
                log.warning("Failed to repair shortcut %s: %s", shortcut, exc)
