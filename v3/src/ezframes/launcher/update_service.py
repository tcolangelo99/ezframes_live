from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ezframes import __version__ as EZFRAMES_VERSION
from ezframes.common.config import AppPaths, RuntimeConfig
from ezframes.common.state import InstallState
from ezframes.common.types import ManifestAsset, ReleaseManifest
from ezframes.launcher.runtime_installer import RuntimeInstaller


log = logging.getLogger(__name__)


class UpdateService:
    def __init__(self, paths: AppPaths, runtime: RuntimeConfig):
        self.paths = paths
        self.runtime = runtime
        self.installer = RuntimeInstaller(paths, runtime)
        self.session = requests.Session()
        retry = Retry(
            total=runtime.max_retries,
            connect=runtime.max_retries,
            read=runtime.max_retries,
            status=runtime.max_retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def fetch_manifest(self) -> ReleaseManifest:
        log.info("Fetching manifest from %s", self.runtime.manifest_url)
        resp = self.session.get(
            self.runtime.manifest_url,
            timeout=(self.runtime.connect_timeout_seconds, self.runtime.read_timeout_seconds),
        )
        resp.raise_for_status()
        data = resp.json()
        return self._parse_manifest(data)

    def _parse_manifest(self, data: dict) -> ReleaseManifest:
        required = ["schema_version", "app_version", "min_launcher_version", "published_at", "notes_url", "assets"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Manifest missing fields: {missing}")
        assets = []
        for item in data["assets"]:
            assets.append(
                ManifestAsset(
                    name=str(item["name"]),
                    url=str(item["url"]),
                    sha256=str(item["sha256"]),
                    size=int(item.get("size", 0)),
                )
            )
        return ReleaseManifest(
            schema_version=str(data["schema_version"]),
            app_version=str(data["app_version"]),
            min_launcher_version=str(data["min_launcher_version"]),
            published_at=str(data["published_at"]),
            notes_url=str(data["notes_url"]),
            assets=tuple(assets),
        )

    def has_update(self, state: InstallState, manifest: ReleaseManifest) -> bool:
        return state.installed_version != manifest.app_version

    @staticmethod
    def _parse_version(version: str) -> Tuple[int, int, int]:
        parts = [p for p in str(version).strip().split(".") if p]
        nums: list[int] = []
        for part in parts[:3]:
            digits = "".join(ch for ch in part if ch.isdigit())
            nums.append(int(digits) if digits else 0)
        while len(nums) < 3:
            nums.append(0)
        return (nums[0], nums[1], nums[2])

    def launcher_version(self) -> str:
        return EZFRAMES_VERSION

    def launcher_meets_min_version(self, manifest: ReleaseManifest) -> bool:
        current = self._parse_version(self.launcher_version())
        required = self._parse_version(manifest.min_launcher_version)
        return current >= required

    def prompt_launcher_upgrade(self, manifest: ReleaseManifest) -> None:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror(
            "Launcher Upgrade Required",
            "This release requires a newer EzFrames launcher.\n\n"
            f"Current launcher: {self.launcher_version()}\n"
            f"Required launcher: {manifest.min_launcher_version}\n\n"
            f"Install the latest bootstrap installer:\n{manifest.notes_url}",
            parent=root,
        )
        root.destroy()

    def prompt_update(self, current: str | None, target: str) -> bool:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        msg = f"Update EzFrames from {current or 'none'} to {target}?"
        ok = messagebox.askyesno("EzFrames Update", msg, parent=root)
        root.destroy()
        return ok

    def apply_update(self, state: InstallState, manifest: ReleaseManifest) -> InstallState:
        checksums = self.installer.install_from_manifest(manifest)
        state.installed_version = manifest.app_version
        state.asset_checksums = checksums
        state.touch_update_time()
        log.info("Update applied: %s", asdict(state))
        return state
