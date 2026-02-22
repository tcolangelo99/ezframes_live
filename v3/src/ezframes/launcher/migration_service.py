from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ezframes.common.config import AppPaths
from ezframes.common.state import InstallState


log = logging.getLogger(__name__)


class MigrationService:
    def __init__(self, paths: AppPaths):
        self.paths = paths

    @staticmethod
    def _legacy_program_files_root() -> Path | None:
        program_files = os.environ.get("ProgramFiles", "").strip()
        if not program_files:
            return None
        return Path(program_files) / "ezframes_live"

    @staticmethod
    def _legacy_documents_root() -> Path:
        return Path.home() / "Documents" / "ezframes"

    def _should_cleanup_legacy_root(self, legacy_root: Path) -> bool:
        # Only clean known install roots. Do not wipe ad-hoc developer folders.
        if legacy_root.resolve() == self.paths.install_root.resolve():
            return False

        candidates: list[Path] = [self._legacy_documents_root()]
        pf = self._legacy_program_files_root()
        if pf is not None:
            candidates.append(pf)

        for candidate in candidates:
            try:
                if legacy_root.resolve() == candidate.resolve():
                    return True
            except Exception:
                if str(legacy_root).lower() == str(candidate).lower():
                    return True
        return False

    def discover_legacy_paths(self) -> list[Path]:
        candidates: list[Path] = []
        pf = self._legacy_program_files_root()
        if pf is not None:
            candidates.append(pf)
        candidates.append(self._legacy_documents_root())
        cwd_candidate = Path.cwd()
        if (cwd_candidate / "ezcutie.py").exists() and (cwd_candidate / "weights").exists():
            candidates.append(cwd_candidate)
        return [p for p in candidates if p.exists()]

    def run_if_needed(self, state: InstallState) -> InstallState:
        if state.migration_completed:
            return state
        legacy_paths = self.discover_legacy_paths()
        state.legacy_source_paths = [str(p) for p in legacy_paths]
        warnings: list[str] = []

        for legacy in legacy_paths:
            try:
                self._migrate_from(legacy)
            except Exception as exc:
                msg = f"Migration warning from {legacy}: {exc}"
                log.warning(msg)
                warnings.append(msg)

        state.migration_warnings = warnings
        state.migration_completed = True
        return state

    def _copytree_if_exists(self, src: Path, dst: Path) -> None:
        if not src.exists():
            return
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    def _copyfile_if_exists(self, src: Path, dst: Path) -> None:
        if not src.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    @staticmethod
    def _remove_if_exists(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
        else:
            path.unlink()

    def _write_migration_marker(self, legacy_root: Path) -> None:
        marker = legacy_root / "MIGRATED_TO_EZFRAMES_V3.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        marker.write_text(
            (
                "EzFrames has been migrated to v3 runtime.\n"
                f"Migrated at: {now}\n"
                f"New install root: {self.paths.install_root}\n"
                "Legacy runtime files were cleaned after migration.\n"
            ),
            encoding="utf-8",
        )

    def _cleanup_legacy_root(self, legacy_root: Path) -> None:
        if not self._should_cleanup_legacy_root(legacy_root):
            log.info("Skipping legacy cleanup for non-install path: %s", legacy_root)
            return

        remove_dirs = (
            "_internal",
            "Cutie_docs",
            "RIFE_docs",
            "ffmpeg",
            "icons",
            "model",
            "weights",
            "workspace",
            ".git",
            ".github",
            "__pycache__",
        )
        remove_files = (
            "ezcutie.exe",
            "launcher.exe",
            "launcher_icon.ico",
            "ezframes_icon.ico",
            "python311.dll",
            "libcrypto-3.dll",
            "libssl-3.dll",
            "IFNet_HDv3.py",
            "RIFE_HDv3.py",
            "motion_interpolation.py",
            "flownet.pkl",
            "WWE Raw.ttf",
            "setup_v2.bat",
            "current_version.txt",
            "version.txt",
            "_internal.tar",
            "launcher_log.txt",
            "launcher_error_log.txt",
            "launcher_login_error_log.txt",
            "downloader_log.txt",
            "updater_log.txt",
            "universal_key.txt",
            "machine_key.txt",
        )

        errors: list[str] = []
        for rel in remove_dirs:
            target = legacy_root / rel
            if not target.exists():
                continue
            try:
                self._remove_if_exists(target)
            except Exception as exc:
                errors.append(f"{target}: {exc}")

        for rel in remove_files:
            target = legacy_root / rel
            if not target.exists():
                continue
            try:
                self._remove_if_exists(target)
            except Exception as exc:
                errors.append(f"{target}: {exc}")

        # Remove top-level leftovers that are clearly runtime artifacts.
        for child in legacy_root.glob("*.spec"):
            try:
                self._remove_if_exists(child)
            except Exception as exc:
                errors.append(f"{child}: {exc}")
        for child in legacy_root.glob("*.py"):
            try:
                self._remove_if_exists(child)
            except Exception as exc:
                errors.append(f"{child}: {exc}")

        self._write_migration_marker(legacy_root)

        if errors:
            raise RuntimeError("Legacy cleanup had failures: " + "; ".join(errors[:8]))
        log.info("Legacy install cleaned: %s", legacy_root)

    def _migrate_from(self, legacy_root: Path) -> None:
        # Workspace and model payload migration.
        self._copytree_if_exists(legacy_root / "workspace", self.paths.workspace_dir)
        self._copytree_if_exists(legacy_root / "weights", self.paths.models_dir / "weights")
        self._copytree_if_exists(legacy_root / "icons", self.paths.assets_dir / "icons")
        self._copytree_if_exists(legacy_root / "ffmpeg", self.paths.assets_dir / "ffmpeg")

        self._copyfile_if_exists(legacy_root / "flownet.pkl", self.paths.models_dir / "flownet.pkl")
        self._copyfile_if_exists(legacy_root / "current_version.txt", self.paths.state_dir / "legacy_current_version.txt")
        self._copyfile_if_exists(legacy_root / "version.txt", self.paths.state_dir / "legacy_version.txt")
        self._cleanup_legacy_root(legacy_root)
        log.info("Migrated legacy payload from %s", legacy_root)
