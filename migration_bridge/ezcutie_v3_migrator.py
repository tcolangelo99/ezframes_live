from __future__ import annotations

import ctypes
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_REPO = "tcolangelo99/ezframes_live"
GITHUB_API = "https://api.github.com/repos/{repo}/releases/latest"
INSTALLER_SUFFIX = "_bootstrap_installer.exe"
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT_SECONDS = 60
INSTALL_TIMEOUT_SECONDS = 60 * 30


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


def find_v3_launcher() -> tuple[Path, Path] | None:
    root = _install_root()
    pythonw = root / "runtime" / "python" / "pythonw.exe"
    bootstrap = root / "bootstrap_launcher.py"
    if pythonw.exists() and bootstrap.exists():
        return pythonw, bootstrap
    return None


def launch_v3() -> bool:
    found = find_v3_launcher()
    if not found:
        return False
    pythonw, bootstrap = found
    cmd = [str(pythonw), str(bootstrap)]
    logging.info("Launching v3 runtime: %s", cmd)
    subprocess.Popen(cmd, cwd=str(_install_root()), shell=False)
    return True


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


def download_asset(asset: dict, target_path: Path) -> None:
    url = str(asset.get("browser_download_url", "")).strip()
    if not url:
        raise RuntimeError(f"Asset {asset.get('name')} missing browser_download_url")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            logging.info("Downloading %s -> %s (attempt %d)", url, target_path, attempt)
            req = urllib.request.Request(url, headers={"User-Agent": "EzFrames-V2-Bridge/1.0"})
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:
                with target_path.open("wb") as fh:
                    shutil.copyfileobj(resp, fh)
            if target_path.stat().st_size <= 0:
                raise RuntimeError(f"Downloaded empty file: {target_path}")
            return
        except Exception as exc:
            logging.warning("Download failed for %s (attempt %d): %s", url, attempt, exc)
            if attempt == DOWNLOAD_RETRIES:
                raise
            time.sleep(1.5 * attempt)


def run_installer(installer_path: Path) -> int:
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
    while proc.poll() is None:
        if time.time() - start > INSTALL_TIMEOUT_SECONDS:
            proc.kill()
            raise TimeoutError("Installer timed out.")
        time.sleep(0.5)
    return int(proc.returncode or 0)


def wait_for_v3_install(timeout_seconds: int = 45) -> bool:
    end = time.time() + timeout_seconds
    while time.time() < end:
        if find_v3_launcher() is not None:
            return True
        time.sleep(1.0)
    return False


def launch_legacy_fallback(exe_dir: Path) -> int:
    legacy = exe_dir / "ezcutie_legacy.exe"
    if legacy.exists():
        logging.info("Launching legacy fallback: %s", legacy)
        subprocess.Popen([str(legacy)], cwd=str(exe_dir), shell=False)
        return 0

    msg = (
        "EzFrames migration to v3 failed and no legacy fallback was found.\n\n"
        "Please reinstall EzFrames from the latest installer."
    )
    logging.error(msg)
    show_message("EzFrames Migration Error", msg)
    return 2


def migrate_to_v3(exe_dir: Path) -> int:
    if launch_v3():
        logging.info("v3 already installed; launched successfully.")
        return 0

    repo = os.environ.get("EZFRAMES_MIGRATION_REPO", DEFAULT_REPO).strip() or DEFAULT_REPO
    logging.info("Starting v2->v3 migration using repo: %s", repo)

    try:
        release = github_latest_release(repo)
        installer_asset, bin_assets = select_installer_and_bins(release)
        release_tag = str(release.get("tag_name", "latest"))
        tmp_dir = Path(tempfile.gettempdir()) / "EzFramesV3Migration" / release_tag
        tmp_dir.mkdir(parents=True, exist_ok=True)

        installer_path = tmp_dir / str(installer_asset.get("name"))
        download_asset(installer_asset, installer_path)
        for bin_asset in bin_assets:
            download_asset(bin_asset, tmp_dir / str(bin_asset.get("name")))

        code = run_installer(installer_path)
        if code not in (0, 1641, 3010):
            raise RuntimeError(f"Installer returned unexpected exit code: {code}")
        if not wait_for_v3_install():
            raise RuntimeError("v3 install did not materialize expected runtime files.")
        if not launch_v3():
            raise RuntimeError("v3 runtime files found but launch failed.")
        logging.info("Migration complete, v3 launched.")
        return 0
    except Exception as exc:
        logging.exception("Migration failed: %s", exc)
        return launch_legacy_fallback(exe_dir)


def main() -> int:
    configure_logging()
    exe_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    logging.info("Bridge started from %s", exe_dir)
    return migrate_to_v3(exe_dir)


if __name__ == "__main__":
    raise SystemExit(main())
