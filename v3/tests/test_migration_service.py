from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ezframes.common.config import AppPaths
from ezframes.launcher.migration_service import MigrationService


def _build_paths(root: Path) -> AppPaths:
    install_root = root / "install"
    return AppPaths(
        install_root=install_root,
        runtime_root=install_root / "runtime",
        runtime_python_dir=install_root / "runtime" / "python",
        runtime_env_dir=install_root / "runtime" / "env",
        app_dir=install_root / "app",
        assets_dir=install_root / "assets",
        models_dir=install_root / "models",
        workspace_dir=install_root / "workspace",
        logs_dir=install_root / "logs",
        state_dir=install_root / "state",
        temp_dir=install_root / "state" / "tmp",
    )


class MigrationServiceTests(unittest.TestCase):
    def test_migrate_and_cleanup_known_legacy_install(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = _build_paths(root)
            paths.ensure_layout()

            program_files = root / "ProgramFiles"
            legacy = program_files / "ezframes_live"
            (legacy / "workspace").mkdir(parents=True, exist_ok=True)
            (legacy / "workspace" / "clip1").mkdir(parents=True, exist_ok=True)
            (legacy / "workspace" / "clip1" / "frame.txt").write_text("x", encoding="utf-8")
            (legacy / "weights").mkdir(parents=True, exist_ok=True)
            (legacy / "weights" / "cutie-base-mega.pth").write_text("w", encoding="utf-8")
            (legacy / "ezcutie.exe").write_text("exe", encoding="utf-8")
            (legacy / "launcher.exe").write_text("launcher", encoding="utf-8")
            (legacy / "some_helper.py").write_text("print('x')", encoding="utf-8")

            svc = MigrationService(paths)
            with patch.dict(os.environ, {"ProgramFiles": str(program_files)}):
                svc._migrate_from(legacy)

            self.assertTrue((paths.workspace_dir / "clip1" / "frame.txt").exists())
            self.assertTrue((paths.models_dir / "weights" / "cutie-base-mega.pth").exists())
            self.assertFalse((legacy / "ezcutie.exe").exists())
            self.assertFalse((legacy / "launcher.exe").exists())
            self.assertFalse((legacy / "some_helper.py").exists())
            self.assertTrue((legacy / "MIGRATED_TO_EZFRAMES_V3.txt").exists())

    def test_migrate_does_not_cleanup_dev_staging_folder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = _build_paths(root)
            paths.ensure_layout()

            dev = root / "dev_staging"
            (dev / "workspace").mkdir(parents=True, exist_ok=True)
            (dev / "workspace" / "clip2").mkdir(parents=True, exist_ok=True)
            (dev / "workspace" / "clip2" / "frame.txt").write_text("x", encoding="utf-8")
            (dev / "weights").mkdir(parents=True, exist_ok=True)
            (dev / "weights" / "cutie-base-mega.pth").write_text("w", encoding="utf-8")
            (dev / "ezcutie.py").write_text("print('legacy')", encoding="utf-8")

            svc = MigrationService(paths)
            with patch.dict(os.environ, {"ProgramFiles": str(root / "SomewhereElse")}):
                svc._migrate_from(dev)

            self.assertTrue((paths.workspace_dir / "clip2" / "frame.txt").exists())
            self.assertTrue((dev / "ezcutie.py").exists())
            self.assertFalse((dev / "MIGRATED_TO_EZFRAMES_V3.txt").exists())


if __name__ == "__main__":
    unittest.main()
