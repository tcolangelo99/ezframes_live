from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from ezframes.common.config import AppPaths


log = logging.getLogger(__name__)


class ProcessService:
    def __init__(self, paths: AppPaths):
        self.paths = paths

    @staticmethod
    def _merge_pythonpath(parts: list[str], existing: str = "") -> str:
        merged: list[str] = []
        seen: set[str] = set()

        for part in [*parts, *existing.split(os.pathsep)]:
            p = str(part).strip()
            if not p:
                continue
            key = os.path.normcase(os.path.normpath(p))
            if key in seen:
                continue
            seen.add(key)
            merged.append(p)
        return os.pathsep.join(merged)

    def _runtime_pythonpath_parts(self) -> list[str]:
        parts: list[str] = []

        app_dir = self.paths.app_dir
        if (app_dir / "ezframes").exists():
            parts.append(str(app_dir))
        if (app_dir / "src" / "ezframes").exists():
            parts.append(str(app_dir / "src"))

        env_dir = self.paths.runtime_env_dir
        if env_dir.exists():
            parts.append(str(env_dir))
        env_site_packages = env_dir / "Lib" / "site-packages"
        if env_site_packages.exists():
            parts.append(str(env_site_packages))

        return parts

    def _build_child_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = self._merge_pythonpath(
            self._runtime_pythonpath_parts(),
            env.get("PYTHONPATH", ""),
        )
        env.setdefault("EZFRAMES_SOURCE_ROOT", str(self.paths.install_root))
        return env

    def _python_can_import_module(self, python_exe: str, module_name: str, env: dict[str, str] | None = None) -> bool:
        try:
            completed = subprocess.run(
                [python_exe, "-c", f"import {module_name}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                shell=False,
                env=env,
            )
            return completed.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _existing_python(path: Path) -> Path | None:
        if path.exists():
            return path
        return None

    def _candidate_pythons(self) -> list[Path]:
        candidates: list[Path] = []
        runtime_env_py = self.paths.runtime_env_dir / "Scripts" / "python.exe"
        runtime_py = self.paths.runtime_python_dir / "python.exe"
        runtime_pythonw = self.paths.runtime_python_dir / "pythonw.exe"

        for candidate in (
            runtime_env_py,
            runtime_py,
            runtime_pythonw,
            self.paths.python_exe,
            Path(sys.executable),
        ):
            c = self._existing_python(candidate)
            if c is not None:
                candidates.append(c)

        for root in self.paths.source_roots():
            c = self._existing_python(root / "venv" / "Scripts" / "python.exe")
            if c is not None:
                candidates.append(c)
            c = self._existing_python(root / "cutievenv" / "Scripts" / "python.exe")
            if c is not None:
                candidates.append(c)

        dedup: list[Path] = []
        seen: set[str] = set()
        for c in candidates:
            key = os.path.normcase(os.path.normpath(str(c)))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(c)
        return dedup

    def _select_app_python(self) -> str:
        env = self._build_child_env()
        candidates = self._candidate_pythons()

        for candidate in candidates:
            c_str = str(candidate)
            if self._python_can_import_module(c_str, "cv2", env=env) and self._python_can_import_module(
                c_str, "ezframes", env=env
            ):
                return c_str

        # Fallback: use the first existing candidate and allow the child process
        # to provide a precise import error if dependencies are missing.
        for candidate in candidates:
            return str(candidate)
        return str(self.paths.python_exe)

    def launch_main_app(self, launch_ticket_path: Path | None = None) -> subprocess.Popen:
        python_exe = self._select_app_python()
        # Default app mode is new/refactored UI. Set EZFRAMES_APP_MODE=legacy
        # only as a temporary fallback path.
        cmd = [python_exe, "-m", "ezframes.app"]
        env = self._build_child_env()
        env.setdefault("EZFRAMES_APP_MODE", "new")
        if launch_ticket_path is not None:
            env["EZFRAMES_LAUNCH_TICKET"] = str(launch_ticket_path)
        log.info("Launching main app: %s", cmd)
        return subprocess.Popen(cmd, shell=False, env=env, cwd=str(self.paths.install_root))
