from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    install_root: Path
    runtime_root: Path
    runtime_python_dir: Path
    runtime_env_dir: Path
    app_dir: Path
    assets_dir: Path
    models_dir: Path
    workspace_dir: Path
    logs_dir: Path
    state_dir: Path
    temp_dir: Path

    @classmethod
    def default(cls) -> "AppPaths":
        override_root = os.environ.get("EZFRAMES_INSTALL_ROOT", "").strip()
        if override_root:
            install_root = Path(override_root)
        else:
            local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            install_root = local_app_data / "EzFrames"
        return cls(
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

    @property
    def python_exe(self) -> Path:
        pythonw = self.runtime_python_dir / "pythonw.exe"
        if pythonw.exists():
            return pythonw
        python = self.runtime_python_dir / "python.exe"
        if python.exists():
            return python
        env_python = os.environ.get("PYTHON_EXECUTABLE", "").strip() or os.environ.get("PYTHON", "").strip()
        if env_python:
            return Path(env_python)
        return Path(sys.executable)

    def ensure_layout(self) -> None:
        for path in (
            self.install_root,
            self.runtime_root,
            self.app_dir,
            self.assets_dir,
            self.models_dir,
            self.workspace_dir,
            self.logs_dir,
            self.state_dir,
            self.temp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def source_roots(self) -> list[Path]:
        roots: list[Path] = []

        env_root = os.environ.get("EZFRAMES_SOURCE_ROOT", "").strip()
        if env_root:
            roots.append(Path(env_root))

        roots.append(Path.cwd())
        roots.append(self.install_root)

        try:
            # src/ezframes/common/config.py -> project root is parents[4]
            roots.append(Path(__file__).resolve().parents[4])
        except Exception:
            pass

        dedup: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root.resolve()) if root.exists() else str(root)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(root)
        return dedup


@dataclass(frozen=True)
class RuntimeConfig:
    manifest_url: str
    update_timeout_seconds: int = 30
    download_chunk_size: int = 1024 * 1024
    connect_timeout_seconds: int = 10
    read_timeout_seconds: int = 60
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            manifest_url=os.environ.get(
                "EZFRAMES_MANIFEST_URL",
                "https://github.com/tcolangelo99/ezframes_live/releases/latest/download/manifest.v1.json",
            ),
            update_timeout_seconds=int(os.environ.get("EZFRAMES_UPDATE_TIMEOUT", "30")),
            download_chunk_size=int(os.environ.get("EZFRAMES_DOWNLOAD_CHUNK", str(1024 * 1024))),
            connect_timeout_seconds=int(os.environ.get("EZFRAMES_CONNECT_TIMEOUT", "10")),
            read_timeout_seconds=int(os.environ.get("EZFRAMES_READ_TIMEOUT", "60")),
            max_retries=int(os.environ.get("EZFRAMES_MAX_RETRIES", "3")),
        )
