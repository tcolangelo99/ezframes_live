from __future__ import annotations

import logging
import queue
import threading
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ezframes import __version__ as EZFRAMES_VERSION
from ezframes.common.config import AppPaths, RuntimeConfig
from ezframes.common.manifest_security import (
    MANIFEST_SCHEMA_VERSION,
    validate_trusted_url,
    verify_manifest_signature,
)
from ezframes.common.state import InstallState
from ezframes.common.types import ManifestAsset, ReleaseManifest
from ezframes.launcher.runtime_installer import InstallerProgress, RuntimeInstaller


log = logging.getLogger(__name__)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(value)
    idx = 0
    while amount >= 1024.0 and idx < len(units) - 1:
        amount /= 1024.0
        idx += 1
    return f"{amount:.1f}{units[idx]}"


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

    def recover_interrupted_update(self) -> None:
        try:
            self.installer.recover_interrupted_apply()
        except Exception as exc:
            log.warning("Interrupted update recovery failed: %s", exc)

    def _icon_candidates(self) -> list[Path]:
        candidates: list[Path] = [
            self.paths.assets_dir / "icons" / "ezframes_icon.ico",
            self.paths.assets_dir / "icons" / "launcher_icon.ico",
            self.paths.install_root / "icons" / "ezframes_icon.ico",
        ]
        for root in self.paths.source_roots():
            candidates.append(root / "icons" / "ezframes_icon.ico")
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

    def fetch_manifest(self) -> ReleaseManifest:
        log.info("Fetching manifest from %s", self.runtime.manifest_url)
        validate_trusted_url(
            self.runtime.manifest_url,
            self.runtime.trusted_manifest_hosts,
            allow_http=self.runtime.allow_insecure_http,
        )
        resp = self.session.get(
            self.runtime.manifest_url,
            timeout=(self.runtime.connect_timeout_seconds, self.runtime.read_timeout_seconds),
        )
        resp.raise_for_status()
        validate_trusted_url(
            str(resp.url),
            self.runtime.trusted_manifest_hosts,
            allow_http=self.runtime.allow_insecure_http,
        )
        data = resp.json()
        verify_manifest_signature(data)
        return self._parse_manifest(data)

    def _parse_manifest(self, data: dict) -> ReleaseManifest:
        required = ["schema_version", "app_version", "min_launcher_version", "published_at", "notes_url", "assets"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Manifest missing fields: {missing}")
        schema_version = str(data["schema_version"]).strip()
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema version {schema_version!r}; expected {MANIFEST_SCHEMA_VERSION!r}."
            )
        assets = []
        seen_names: set[str] = set()
        for item in data["assets"]:
            asset_name = str(item["name"]).strip()
            if not asset_name:
                raise ValueError("Manifest asset name cannot be empty.")
            if asset_name in seen_names:
                raise ValueError(f"Manifest contains duplicate asset name: {asset_name}")
            seen_names.add(asset_name)
            asset_url = str(item["url"]).strip()
            validate_trusted_url(
                asset_url,
                self.runtime.trusted_asset_hosts,
                allow_http=self.runtime.allow_insecure_http,
            )
            assets.append(
                ManifestAsset(
                    name=asset_name,
                    url=asset_url,
                    sha256=str(item["sha256"]),
                    size=int(item.get("size", 0)),
                )
            )
        if not assets:
            raise ValueError("Manifest has no assets.")
        return ReleaseManifest(
            schema_version=schema_version,
            app_version=str(data["app_version"]),
            min_launcher_version=str(data["min_launcher_version"]),
            published_at=str(data["published_at"]),
            notes_url=str(data["notes_url"]),
            assets=tuple(assets),
        )

    def has_update(self, state: InstallState, manifest: ReleaseManifest) -> bool:
        current = state.installed_version or ""
        if not current:
            return True
        current_v = self._parse_version(current)
        target_v = self._parse_version(manifest.app_version)
        if target_v < current_v:
            log.warning(
                "Refusing downgrade manifest. installed=%s target=%s",
                current,
                manifest.app_version,
            )
            return False
        return target_v > current_v

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
        self._apply_window_icon(root)
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
        self._apply_window_icon(root)
        msg = f"Update EzFrames from {current or 'none'} to {target}?"
        ok = messagebox.askyesno("EzFrames Update", msg, parent=root)
        root.destroy()
        return ok

    def _apply_update_with_progress_ui(
        self,
        manifest: ReleaseManifest,
        handoff_callback: Callable[[dict[str, str], Callable[[str, str], None]], None] | None = None,
    ) -> dict[str, str]:
        try:
            import tkinter as tk
            from tkinter import messagebox, ttk
        except Exception as exc:
            raise RuntimeError(
                "Updater UI is unavailable; mandatory update cannot continue without a visible window."
            ) from exc

        total_assets = max(len(manifest.assets), 1)
        events: queue.Queue[tuple[str, object]] = queue.Queue()
        outcome: dict[str, object] = {"checksums": None, "error": None}
        seen_log_lines: set[str] = set()

        root = tk.Tk()
        root.title("EzFrames Updater")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", lambda: None)
        root.geometry("720x400")
        root.minsize(720, 400)
        self._apply_window_icon(root)

        container = ttk.Frame(root, padding=14)
        container.pack(fill="both", expand=True)

        title_var = tk.StringVar(value=f"Updating EzFrames to {manifest.app_version}")
        status_var = tk.StringVar(value="Preparing update...")
        detail_var = tk.StringVar(value="The app will launch automatically when this finishes.")
        overall_var = tk.StringVar(value=f"Assets: 0/{len(manifest.assets)}")
        download_var = tk.StringVar(value="Current file: waiting")

        ttk.Label(container, textvariable=title_var, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(container, textvariable=status_var, font=("Segoe UI", 10)).pack(anchor="w", pady=(8, 0))
        ttk.Label(container, textvariable=detail_var, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 10))

        ttk.Label(container, textvariable=overall_var).pack(anchor="w")
        overall_bar = ttk.Progressbar(container, mode="determinate", maximum=float(total_assets), value=0.0)
        overall_bar.pack(fill="x", pady=(2, 10))

        ttk.Label(container, textvariable=download_var).pack(anchor="w")
        download_bar = ttk.Progressbar(container, mode="indeterminate")
        download_bar.pack(fill="x", pady=(2, 10))
        download_bar.start(14)

        ttk.Label(container, text="Steps").pack(anchor="w")
        log_text = tk.Text(container, height=10, wrap="word", state="disabled")
        log_text.pack(fill="both", expand=True)

        def append_log(line: str) -> None:
            cleaned = line.strip()
            if not cleaned or cleaned in seen_log_lines:
                return
            seen_log_lines.add(cleaned)
            log_text.configure(state="normal")
            log_text.insert("end", cleaned + "\n")
            log_text.see("end")
            log_text.configure(state="disabled")

        def set_download_indeterminate() -> None:
            mode = str(download_bar.cget("mode"))
            if mode != "indeterminate":
                download_bar.stop()
                download_bar.configure(mode="indeterminate", value=0)
                download_bar.start(14)

        def set_download_progress(done: int, total: int) -> None:
            if total <= 0:
                set_download_indeterminate()
                return
            mode = str(download_bar.cget("mode"))
            if mode != "determinate":
                download_bar.stop()
                download_bar.configure(mode="determinate", maximum=100.0)
            pct = max(0.0, min(100.0, (float(done) / float(total)) * 100.0))
            download_bar.configure(value=pct)

        def on_progress(p: InstallerProgress) -> None:
            events.put(("progress", p))

        def worker() -> None:
            try:
                checksums = self.installer.install_from_manifest(manifest, progress_callback=on_progress)
                events.put(("done", checksums))
            except Exception as exc:  # pragma: no cover - exercised interactively
                tb = traceback.format_exc()
                events.put(("error", (exc, tb)))

        thread = threading.Thread(target=worker, daemon=True, name="ezframes-update-worker")
        thread.start()

        finished = {"value": False}
        handoff_started = {"value": False}

        def handle_progress(p: InstallerProgress) -> None:
            status_var.set(p.message)
            if p.phase in {"prepare", "complete", "rollback"}:
                detail_var.set("Please keep this window open.")
            elif p.phase.startswith("apply"):
                detail_var.set("Applying files...")
            elif p.phase.startswith("extract"):
                detail_var.set("Unpacking archive...")
            elif p.phase.startswith("verify"):
                detail_var.set("Verifying checksum...")
            else:
                detail_var.set("Downloading files...")

            if p.asset_index is not None and p.asset_total:
                completed = p.asset_index - 1
                if p.phase == "extract-done":
                    completed = p.asset_index
                elif p.phase == "download-progress" and p.bytes_done is not None and p.bytes_total:
                    completed = (p.asset_index - 1) + (float(p.bytes_done) / float(p.bytes_total))
                overall_bar.configure(maximum=float(max(p.asset_total, 1)), value=float(max(completed, 0.0)))
                shown = min(int(completed), p.asset_total)
                overall_var.set(f"Assets: {shown}/{p.asset_total}")

            if p.phase == "download-progress":
                if p.bytes_done is not None and p.bytes_total:
                    set_download_progress(p.bytes_done, p.bytes_total)
                    download_var.set(
                        f"Current file: {_format_bytes(p.bytes_done)} / {_format_bytes(p.bytes_total)}"
                    )
                elif p.bytes_done is not None:
                    set_download_indeterminate()
                    download_var.set(f"Current file: {_format_bytes(p.bytes_done)}")
            else:
                set_download_indeterminate()
                if p.asset_name:
                    download_var.set(f"Current file: {p.asset_name}")

            if p.message:
                append_log(p.message)

        def poll_events() -> None:
            while True:
                try:
                    kind, payload = events.get_nowait()
                except queue.Empty:
                    break

                if kind == "progress":
                    handle_progress(payload)  # type: ignore[arg-type]
                elif kind == "done":
                    outcome["checksums"] = payload
                    append_log("Update completed.")
                    if handoff_callback is not None and not handoff_started["value"]:
                        handoff_started["value"] = True
                        status_var.set("Finalizing update...")
                        detail_var.set("Starting updated launcher...")
                        overall_bar.configure(maximum=float(total_assets), value=float(total_assets))
                        overall_var.set(f"Assets: {len(manifest.assets)}/{len(manifest.assets)}")
                        set_download_indeterminate()
                        download_var.set("Current file: handoff")
                        append_log("Beginning relaunch handoff.")

                        def report_handoff(status: str, detail: str = "") -> None:
                            events.put(("handoff-progress", (str(status), str(detail))))

                        def handoff_worker() -> None:
                            try:
                                handoff_callback(payload, report_handoff)
                                events.put(("handoff-done", None))
                            except Exception as exc:
                                tb = traceback.format_exc()
                                events.put(("handoff-error", (exc, tb)))

                        threading.Thread(
                            target=handoff_worker,
                            daemon=True,
                            name="ezframes-update-handoff",
                        ).start()
                    else:
                        finished["value"] = True
                elif kind == "error":
                    finished["value"] = True
                    outcome["error"] = payload
                elif kind == "handoff-progress":
                    status, detail = payload  # type: ignore[misc]
                    status_var.set(str(status))
                    detail_var.set(str(detail))
                    if detail:
                        append_log(f"{status}: {detail}")
                    else:
                        append_log(str(status))
                elif kind == "handoff-done":
                    append_log("Relaunch handoff complete.")
                    finished["value"] = True
                elif kind == "handoff-error":
                    finished["value"] = True
                    outcome["error"] = payload

            if finished["value"]:
                download_bar.stop()
                root.destroy()
                return
            root.after(100, poll_events)

        root.after(50, poll_events)
        root.mainloop()

        err = outcome.get("error")
        if err is not None:
            exc, tb = err  # type: ignore[misc]
            log.error("Update apply failed:\n%s", tb)
            popup = tk.Tk()
            popup.withdraw()
            popup.attributes("-topmost", True)
            self._apply_window_icon(popup)
            messagebox.showerror(
                "Update Failed",
                f"{exc}\n\nSee logs: {self.paths.logs_dir / 'ezframes_v3.log'}",
                parent=popup,
            )
            popup.destroy()
            raise RuntimeError(str(exc)) from exc

        checksums = outcome.get("checksums")
        if not isinstance(checksums, dict):
            raise RuntimeError("Update finished without checksum data.")
        return checksums

    def apply_update(
        self,
        state: InstallState,
        manifest: ReleaseManifest,
        handoff_callback: Callable[[dict[str, str], Callable[[str, str], None]], None] | None = None,
    ) -> InstallState:
        current = state.installed_version or ""
        if current:
            current_v = self._parse_version(current)
            target_v = self._parse_version(manifest.app_version)
            if target_v <= current_v:
                raise RuntimeError(
                    f"Refusing non-forward update. installed={current} target={manifest.app_version}"
                )
        checksums = self._apply_update_with_progress_ui(manifest, handoff_callback=handoff_callback)
        state.installed_version = manifest.app_version
        state.asset_checksums = checksums
        state.touch_update_time()
        log.info("Update applied: %s", asdict(state))
        return state
