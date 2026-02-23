from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable


DEFAULT_REPO = "tcolangelo99/ezframes_live"
GITHUB_API = "https://api.github.com/repos/{repo}/releases/latest"
INSTALLER_SUFFIX = "_bootstrap_installer.exe"
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT_SECONDS = 60
INSTALL_TIMEOUT_SECONDS = 60 * 30
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
UI_UPDATE_MIN_INTERVAL_SECONDS = 0.15
LAUNCH_WAIT_TIMEOUT_SECONDS = 180
LAUNCH_UI_UPDATE_INTERVAL_SECONDS = 0.5


def _local_app_data() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))


def _install_root() -> Path:
    return _local_app_data() / "EzFrames"


def _log_path() -> Path:
    logs_dir = _install_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "v2_to_v3_bridge.log"


def configure_logging() -> None:
    logging.basicConfig(
        filename=str(_log_path()),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def show_message(title: str, text: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(0, text, title, 0x00000010 | 0x00040000)
    except Exception:
        pass


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    value = float(num_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PB"


class MigrationProgressWindow:
    def __init__(self) -> None:
        self._events: queue.Queue[tuple[str, tuple[object, ...]]] = queue.Queue()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="MigrationProgressUI", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("Migration progress UI startup timed out.")
        if not self._enabled:
            raise RuntimeError("Migration progress UI could not be initialized.")

    def _run(self) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk

            root = tk.Tk()
            root.title("EzFrames Update")
            root.geometry("520x160")
            root.resizable(False, False)
            root.attributes("-topmost", True)

            status_var = tk.StringVar(value="Preparing EzFrames migration...")
            detail_var = tk.StringVar(value="Please keep this window open.")
            progress_var = tk.DoubleVar(value=0.0)

            frame = ttk.Frame(root, padding=14)
            frame.pack(fill="both", expand=True)

            ttk.Label(frame, text="EzFrames is updating", font=("Segoe UI", 11, "bold")).pack(anchor="w")
            ttk.Label(frame, textvariable=status_var, wraplength=488).pack(anchor="w", pady=(8, 2))
            ttk.Label(frame, textvariable=detail_var, foreground="#444444", wraplength=488).pack(anchor="w", pady=(0, 8))

            progress = ttk.Progressbar(frame, variable=progress_var, maximum=100.0, mode="determinate", length=488)
            progress.pack(anchor="w")

            def on_close_request() -> None:
                detail_var.set("Update is still running. Please wait...")

            root.protocol("WM_DELETE_WINDOW", on_close_request)

            def handle_event(event_name: str, payload: tuple[object, ...]) -> bool:
                if event_name == "close":
                    root.destroy()
                    return False
                if event_name == "status":
                    status = str(payload[0]) if payload else ""
                    detail = str(payload[1]) if len(payload) > 1 else ""
                    status_var.set(status)
                    detail_var.set(detail)
                    return True
                if event_name == "progress":
                    status = str(payload[0]) if payload else ""
                    percent = float(payload[1]) if len(payload) > 1 else 0.0
                    detail = str(payload[2]) if len(payload) > 2 else ""
                    status_var.set(status)
                    detail_var.set(detail)
                    if str(progress.cget("mode")) != "determinate":
                        progress.stop()
                        progress.configure(mode="determinate")
                    progress_var.set(max(0.0, min(100.0, percent)))
                    return True
                if event_name == "mode":
                    mode = str(payload[0]) if payload else "determinate"
                    if mode not in ("determinate", "indeterminate"):
                        mode = "determinate"
                    if str(progress.cget("mode")) != mode:
                        progress.stop()
                        progress.configure(mode=mode)
                    if mode == "indeterminate":
                        progress.start(40)
                    return True
                return True

            def pump() -> None:
                keep_running = True
                while True:
                    try:
                        event_name, payload = self._events.get_nowait()
                    except queue.Empty:
                        break
                    keep_running = handle_event(event_name, payload)
                    if not keep_running:
                        return
                root.after(120, pump)

            self._enabled = True
            self._ready.set()
            root.after(50, pump)
            root.mainloop()
        except Exception:
            logging.exception("Progress UI unavailable.")
            self._enabled = False
            self._ready.set()

    def _emit(self, event_name: str, *payload: object) -> None:
        if self._enabled:
            self._events.put((event_name, payload))

    def set_status(self, status: str, detail: str = "") -> None:
        self._emit("status", status, detail)

    def set_progress(self, status: str, percent: float, detail: str = "") -> None:
        self._emit("progress", status, percent, detail)

    def set_mode(self, mode: str) -> None:
        self._emit("mode", mode)

    def close(self) -> None:
        self._emit("close")


def find_v3_launcher() -> list[str] | None:
    root = _install_root()
    native_launcher = root / "EzFramesLauncher.exe"
    if native_launcher.exists():
        return [str(native_launcher), "--log-level", "INFO"]

    pythonw = root / "runtime" / "python" / "pythonw.exe"
    bootstrap = root / "bootstrap_launcher.py"
    if pythonw.exists() and bootstrap.exists():
        return [str(pythonw), str(bootstrap), "--log-level", "INFO"]
    return None


def launch_v3(progress_ui: MigrationProgressWindow | None = None) -> bool:
    cmd = find_v3_launcher()
    if cmd is None:
        return False
    logging.info("Launching v3 runtime: %s", cmd)
    proc = subprocess.Popen(cmd, cwd=str(_install_root()), shell=False)

    if progress_ui is not None:
        progress_ui.set_mode("indeterminate")
        progress_ui.set_status("Starting EzFrames v3...", "Opening launcher and loading runtime.")

    start = time.time()
    next_ui_update = 0.0
    while True:
        rc = proc.poll()
        elapsed = time.time() - start
        if rc is not None:
            if rc == 0:
                return True
            logging.error("v3 launcher exited with code %s", rc)
            return False

        if elapsed > LAUNCH_WAIT_TIMEOUT_SECONDS:
            logging.warning("v3 launcher still running after %ss; assuming handoff is in progress.", LAUNCH_WAIT_TIMEOUT_SECONDS)
            return True

        if progress_ui is not None and elapsed >= next_ui_update:
            detail = f"Waiting for launcher handoff... {int(elapsed)}s"
            progress_ui.set_status("Starting EzFrames v3...", detail)
            next_ui_update = elapsed + LAUNCH_UI_UPDATE_INTERVAL_SECONDS
        time.sleep(0.2)


def github_latest_release(repo: str) -> dict:
    url = GITHUB_API.format(repo=repo)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "EzFrames-V2-Bridge/1.0",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def select_installer_and_bins(release: dict) -> tuple[dict, list[dict]]:
    assets = release.get("assets", [])
    if not isinstance(assets, list):
        raise RuntimeError("Release assets payload is invalid.")

    installer = None
    for asset in assets:
        name = str(asset.get("name", ""))
        if name.endswith(INSTALLER_SUFFIX):
            installer = asset
            break
    if installer is None:
        raise RuntimeError("No bootstrap installer asset was found in latest release.")

    installer_name = str(installer.get("name", ""))
    installer_stem = installer_name[:-4]
    bin_pattern = re.compile(re.escape(installer_stem) + r"-(\d+)\.bin$")

    bins_with_index: list[tuple[int, dict]] = []
    for asset in assets:
        name = str(asset.get("name", ""))
        m = bin_pattern.match(name)
        if not m:
            continue
        bins_with_index.append((int(m.group(1)), asset))

    bins_with_index.sort(key=lambda x: x[0])
    bins = [asset for _, asset in bins_with_index]
    return installer, bins


def download_asset(
    asset: dict,
    target_path: Path,
    progress_cb: Callable[[int, int], None] | None = None,
) -> None:
    url = str(asset.get("browser_download_url", "")).strip()
    if not url:
        raise RuntimeError(f"Asset {asset.get('name')} missing browser_download_url")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = _safe_int(asset.get("size"), 0)

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            logging.info("Downloading %s -> %s (attempt %d)", url, target_path, attempt)
            req = urllib.request.Request(url, headers={"User-Agent": "EzFrames-V2-Bridge/1.0"})
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:
                resp_len = _safe_int(resp.headers.get("content-length", "0"), 0)
                if resp_len > 0:
                    expected_size = resp_len
                downloaded = 0
                last_ui_update = 0.0
                with target_path.open("wb") as fh:
                    while True:
                        chunk = resp.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            now = time.time()
                            if now - last_ui_update >= UI_UPDATE_MIN_INTERVAL_SECONDS:
                                progress_cb(downloaded, expected_size)
                                last_ui_update = now
            if target_path.stat().st_size <= 0:
                raise RuntimeError(f"Downloaded empty file: {target_path}")
            if progress_cb:
                progress_cb(target_path.stat().st_size, expected_size)
            return
        except Exception as exc:
            logging.warning("Download failed for %s (attempt %d): %s", url, attempt, exc)
            if attempt == DOWNLOAD_RETRIES:
                raise
            time.sleep(1.5 * attempt)


def run_installer(installer_path: Path, progress_ui: MigrationProgressWindow | None = None) -> int:
    cmd = [
        str(installer_path),
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/SP-",
    ]
    logging.info("Running installer: %s", cmd)
    proc = subprocess.Popen(cmd, cwd=str(installer_path.parent), shell=False)
    start = time.time()
    phase_messages = [
        "Unpacking installer payload...",
        "Installing managed Python runtime...",
        "Installing dependencies and model files...",
        "Finalizing installation files...",
    ]
    if progress_ui is not None:
        progress_ui.set_mode("indeterminate")
        progress_ui.set_status("Installing EzFrames v3...", phase_messages[0])
    while proc.poll() is None:
        if time.time() - start > INSTALL_TIMEOUT_SECONDS:
            proc.kill()
            raise TimeoutError("Installer timed out.")
        if progress_ui is not None:
            elapsed = time.time() - start
            if elapsed < 25:
                phase = phase_messages[0]
            elif elapsed < 70:
                phase = phase_messages[1]
            elif elapsed < 140:
                phase = phase_messages[2]
            else:
                phase = phase_messages[3]
            progress_ui.set_status("Installing EzFrames v3...", phase)
        time.sleep(0.5)
    return int(proc.returncode or 0)


def wait_for_v3_install(
    timeout_seconds: int = 45,
    progress_ui: MigrationProgressWindow | None = None,
) -> bool:
    end = time.time() + timeout_seconds
    started = time.time()
    last_ui_update = 0.0
    while time.time() < end:
        if find_v3_launcher() is not None:
            return True
        if progress_ui is not None:
            now = time.time()
            if now - last_ui_update >= 0.5:
                elapsed = int(now - started)
                progress_ui.set_status("Finalizing installation...", f"Verifying installed files... {elapsed}s")
                last_ui_update = now
        time.sleep(1.0)
    return False


def migration_failed_exit() -> int:
    msg = (
        "EzFrames migration to v3 failed.\n\n"
        "Legacy mode is no longer supported.\n"
        "Please relaunch EzFrames to retry the update, or reinstall from the latest installer.\n\n"
        "Please reinstall EzFrames from the latest installer."
    )
    logging.error(msg)
    show_message("EzFrames Migration Error", msg)
    return 2


def migrate_to_v3(exe_dir: Path) -> int:
    progress_ui = MigrationProgressWindow()
    try:
        progress_ui.start()
    except Exception as exc:
        logging.exception("Migration UI startup failed: %s", exc)
        show_message(
            "EzFrames Migration Error",
            "EzFrames could not open the required update window.\n\n"
            "Please reinstall EzFrames from the latest installer.",
        )
        return 2
    progress_ui.set_status("Checking EzFrames installation...", "Looking for installed v3 runtime.")

    if find_v3_launcher() is not None:
        logging.info("Existing v3 runtime detected; launching without reinstall.")
        if launch_v3(progress_ui=progress_ui):
            progress_ui.set_progress("EzFrames is ready", 100.0, "Opening main app...")
            time.sleep(0.8)
            progress_ui.close()
            return 0
        logging.warning("Existing v3 runtime failed to launch; attempting repair install.")
        progress_ui.set_status("Installed runtime failed to launch", "Repairing installation from latest release.")
        time.sleep(0.8)

    repo = os.environ.get("EZFRAMES_MIGRATION_REPO", DEFAULT_REPO).strip() or DEFAULT_REPO
    logging.info("Starting v2->v3 migration using repo: %s", repo)
    progress_ui.set_status("Checking for EzFrames v3 update...", "Contacting GitHub Releases.")

    try:
        release = github_latest_release(repo)
        installer_asset, bin_assets = select_installer_and_bins(release)
        release_tag = str(release.get("tag_name", "latest"))
        tmp_dir = Path(tempfile.gettempdir()) / "EzFramesV3Migration" / release_tag
        tmp_dir.mkdir(parents=True, exist_ok=True)

        assets_to_download = [installer_asset, *bin_assets]
        total_expected = sum(max(_safe_int(asset.get("size"), 0), 0) for asset in assets_to_download)
        total_downloaded = 0

        for asset_index, asset in enumerate(assets_to_download, start=1):
            asset_name = str(asset.get("name", "asset"))
            asset_expected = max(_safe_int(asset.get("size"), 0), 0)
            asset_path = tmp_dir / asset_name
            if asset_path.exists() and asset_path.stat().st_size == asset_expected and asset_expected > 0:
                logging.info("Reusing cached migration asset: %s", asset_path)
                total_downloaded += asset_expected
                continue
            progress_ui.set_status(
                f"Downloading update file {asset_index}/{len(assets_to_download)}",
                f"{asset_name}",
            )

            def on_progress(downloaded: int, expected: int) -> None:
                nonlocal total_downloaded
                effective_expected = expected if expected > 0 else asset_expected
                overall_downloaded = total_downloaded + downloaded
                if total_expected > 0:
                    percent = (overall_downloaded / total_expected) * 100.0
                elif effective_expected > 0:
                    percent = (downloaded / effective_expected) * 100.0
                else:
                    percent = 0.0
                detail = f"{asset_name}: {_format_bytes(downloaded)}"
                if effective_expected > 0:
                    detail += f" / {_format_bytes(effective_expected)}"
                progress_ui.set_progress("Downloading EzFrames update...", percent, detail)

            download_asset(asset, asset_path, progress_cb=on_progress)
            total_downloaded += asset_path.stat().st_size

        installer_path = tmp_dir / str(installer_asset.get("name"))

        progress_ui.set_status("Installing EzFrames v3...", "Applying downloaded update package.")
        code = run_installer(installer_path, progress_ui=progress_ui)
        if code not in (0, 1641, 3010):
            raise RuntimeError(f"Installer returned unexpected exit code: {code}")
        progress_ui.set_status("Finalizing installation...", "Verifying runtime files.")
        progress_ui.set_mode("indeterminate")
        if not wait_for_v3_install(progress_ui=progress_ui):
            raise RuntimeError("v3 install did not materialize expected runtime files.")
        progress_ui.set_status("Launching EzFrames v3...", "Starting launcher with installed runtime.")
        if not launch_v3(progress_ui=progress_ui):
            raise RuntimeError("v3 runtime files found but launch failed.")
        logging.info("Migration complete, v3 launched.")
        progress_ui.set_progress("Migration complete", 100.0, "Launching EzFrames...")
        time.sleep(1.0)
        progress_ui.close()
        return 0
    except Exception as exc:
        logging.exception("Migration failed: %s", exc)
        progress_ui.set_status("Migration failed", "Legacy mode is disabled. Please relaunch to retry update.")
        time.sleep(1.0)
        progress_ui.close()
        return migration_failed_exit()


def main() -> int:
    configure_logging()
    exe_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    logging.info("Bridge started from %s", exe_dir)
    return migrate_to_v3(exe_dir)


if __name__ == "__main__":
    raise SystemExit(main())
