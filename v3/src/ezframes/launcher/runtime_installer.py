from __future__ import annotations

import logging
import os
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ezframes.common.config import AppPaths, RuntimeConfig
from ezframes.common.hashing import sha256_file
from ezframes.common.manifest_security import (
    validate_archive_member_path,
    validate_trusted_url,
)
from ezframes.common.types import ManifestAsset, ReleaseManifest


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadedAsset:
    asset: ManifestAsset
    archive_path: Path
    extracted_path: Path


@dataclass(frozen=True)
class InstallerProgress:
    phase: str
    message: str
    asset_name: str | None = None
    asset_index: int | None = None
    asset_total: int | None = None
    bytes_done: int | None = None
    bytes_total: int | None = None


class RuntimeInstaller:
    def __init__(self, paths: AppPaths, runtime: RuntimeConfig):
        self.paths = paths
        self.runtime = runtime
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

    def _emit(
        self,
        callback: Callable[[InstallerProgress], None] | None,
        progress: InstallerProgress,
    ) -> None:
        if callback is None:
            return
        try:
            callback(progress)
        except Exception:
            log.exception("Installer progress callback failed.")

    def install_from_manifest(
        self,
        manifest: ReleaseManifest,
        progress_callback: Callable[[InstallerProgress], None] | None = None,
    ) -> dict[str, str]:
        self.paths.ensure_layout()
        self.recover_interrupted_apply(progress_callback=progress_callback)
        staging = self.paths.temp_dir / "staging"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        downloaded: list[DownloadedAsset] = []
        checksums: dict[str, str] = {}
        asset_total = len(manifest.assets)

        self._emit(
            progress_callback,
            InstallerProgress(
                phase="prepare",
                message=f"Preparing update ({asset_total} assets)",
                asset_total=asset_total,
            ),
        )

        for idx, asset in enumerate(manifest.assets, start=1):
            self._emit(
                progress_callback,
                InstallerProgress(
                    phase="download-start",
                    message=f"Downloading {asset.name}",
                    asset_name=asset.name,
                    asset_index=idx,
                    asset_total=asset_total,
                    bytes_done=0,
                    bytes_total=asset.size if asset.size > 0 else None,
                ),
            )
            archive = self._download_asset(
                asset,
                staging,
                progress_callback=progress_callback,
                asset_index=idx,
                asset_total=asset_total,
            )
            self._emit(
                progress_callback,
                InstallerProgress(
                    phase="verify",
                    message=f"Verifying {asset.name}",
                    asset_name=asset.name,
                    asset_index=idx,
                    asset_total=asset_total,
                ),
            )
            digest = sha256_file(archive, chunk_size=self.runtime.download_chunk_size)
            if digest.lower() != asset.sha256.lower():
                raise RuntimeError(f"Checksum mismatch for {asset.name}: {digest} != {asset.sha256}")
            checksums[asset.name] = digest
            self._emit(
                progress_callback,
                InstallerProgress(
                    phase="extract-start",
                    message=f"Extracting {asset.name}",
                    asset_name=asset.name,
                    asset_index=idx,
                    asset_total=asset_total,
                ),
            )
            extracted = self._extract_archive(archive, staging)
            self._emit(
                progress_callback,
                InstallerProgress(
                    phase="extract-done",
                    message=f"Extracted {asset.name}",
                    asset_name=asset.name,
                    asset_index=idx,
                    asset_total=asset_total,
                ),
            )
            downloaded.append(DownloadedAsset(asset=asset, archive_path=archive, extracted_path=extracted))

        self._emit(
            progress_callback,
            InstallerProgress(
                phase="apply-start",
                message="Applying update files",
                asset_total=asset_total,
            ),
        )
        self._atomic_apply(downloaded, progress_callback=progress_callback)
        self._emit(
            progress_callback,
            InstallerProgress(
                phase="complete",
                message="Update applied successfully",
                asset_total=asset_total,
            ),
        )
        return checksums

    @staticmethod
    def _remove_path(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            return
        try:
            path.unlink()
        except Exception:
            pass

    def recover_interrupted_apply(
        self,
        progress_callback: Callable[[InstallerProgress], None] | None = None,
    ) -> None:
        targets = (
            self.paths.runtime_python_dir,
            self.paths.runtime_env_dir,
            self.paths.app_dir,
            self.paths.models_dir,
            self.paths.assets_dir,
        )
        recovered_any = False
        for target in targets:
            backup = target.with_name(target.name + ".bak")
            if not backup.exists():
                continue

            recovered_any = True
            if target.exists():
                self._remove_path(backup)
                self._emit(
                    progress_callback,
                    InstallerProgress(
                        phase="recover",
                        message=f"Recovered stale backup for {target.name}",
                    ),
                )
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            backup.replace(target)
            self._emit(
                progress_callback,
                InstallerProgress(
                    phase="recover",
                    message=f"Restored interrupted update target: {target.name}",
                ),
            )

        if recovered_any:
            log.info("Recovered interrupted update state from .bak targets.")

    def _download_asset(
        self,
        asset: ManifestAsset,
        staging: Path,
        progress_callback: Callable[[InstallerProgress], None] | None = None,
        asset_index: int | None = None,
        asset_total: int | None = None,
    ) -> Path:
        safe_name = self._safe_asset_filename(asset.name)
        destination = staging / safe_name
        validate_trusted_url(
            asset.url,
            self.runtime.trusted_asset_hosts,
            allow_http=self.runtime.allow_insecure_http,
        )
        log.info("Downloading asset %s", asset.name)
        bytes_done = 0
        last_emitted = 0
        emit_threshold = max(4 * 1024 * 1024, self.runtime.download_chunk_size * 4)
        with self.session.get(
            asset.url,
            stream=True,
            timeout=(self.runtime.connect_timeout_seconds, self.runtime.read_timeout_seconds),
        ) as resp:
            resp.raise_for_status()
            validate_trusted_url(
                str(resp.url),
                self.runtime.trusted_asset_hosts,
                allow_http=self.runtime.allow_insecure_http,
            )
            content_len_raw = resp.headers.get("Content-Length", "").strip()
            bytes_total = int(content_len_raw) if content_len_raw.isdigit() else None
            if bytes_total is None and asset.size > 0:
                bytes_total = asset.size
            with destination.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=self.runtime.download_chunk_size):
                    if chunk:
                        fh.write(chunk)
                        bytes_done += len(chunk)
                        if (bytes_done - last_emitted) >= emit_threshold:
                            self._emit(
                                progress_callback,
                                InstallerProgress(
                                    phase="download-progress",
                                    message=f"Downloading {asset.name}",
                                    asset_name=asset.name,
                                    asset_index=asset_index,
                                    asset_total=asset_total,
                                    bytes_done=bytes_done,
                                    bytes_total=bytes_total,
                                ),
                            )
                            last_emitted = bytes_done
        self._emit(
            progress_callback,
            InstallerProgress(
                phase="download-progress",
                message=f"Downloaded {asset.name}",
                asset_name=asset.name,
                asset_index=asset_index,
                asset_total=asset_total,
                bytes_done=bytes_done,
                bytes_total=bytes_total if bytes_total is not None else bytes_done,
            ),
        )
        return destination

    @staticmethod
    def _safe_asset_filename(name: str) -> str:
        raw = str(name or "").strip()
        if not raw:
            raise ValueError("Manifest asset name cannot be empty.")
        path = Path(raw)
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError(f"Invalid manifest asset name: {name!r}")
        if raw in {".", ".."}:
            raise ValueError(f"Invalid manifest asset name: {name!r}")
        if any(ch in raw for ch in ("/", "\\")):
            raise ValueError(f"Invalid manifest asset name: {name!r}")
        return raw

    def _extract_archive(self, archive: Path, staging: Path) -> Path:
        target = staging / f"{archive.stem}_extracted"
        target.mkdir(parents=True, exist_ok=True)
        root = target.resolve()
        with zipfile.ZipFile(archive, "r") as zf:
            for info in zf.infolist():
                member = validate_archive_member_path(info.filename)

                # Block symlinks from archives.
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError(f"Archive contains a symbolic link entry: {info.filename}")

                dest_path = (target / Path(*member.parts)).resolve()
                if not str(dest_path).startswith(str(root) + os.sep) and dest_path != root:
                    raise ValueError(f"Archive entry escapes extraction root: {info.filename}")

                if info.is_dir():
                    dest_path.mkdir(parents=True, exist_ok=True)
                    continue

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, dest_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        return target

    def _target_for_asset(self, asset_name: str) -> Path:
        n = asset_name.lower()
        if n.startswith("runtime-python"):
            return self.paths.runtime_python_dir
        if n.startswith("runtime-env"):
            return self.paths.runtime_env_dir
        if n.startswith("app-bundle"):
            return self.paths.app_dir
        if n.startswith("models"):
            return self.paths.models_dir
        if n.startswith("assets"):
            return self.paths.assets_dir
        return self.paths.runtime_root / Path(asset_name).stem

    @staticmethod
    def _asset_prefers_raw_root(asset_name: str) -> bool:
        n = asset_name.lower()
        return n.startswith("app-bundle") or n.startswith("runtime-env")

    def _single_root(self, extracted: Path, asset_name: str) -> Path:
        if self._asset_prefers_raw_root(asset_name):
            return extracted
        children = [p for p in extracted.iterdir()]
        if len(children) == 1 and children[0].is_dir():
            return children[0]
        return extracted

    def _atomic_apply(
        self,
        downloaded: list[DownloadedAsset],
        progress_callback: Callable[[InstallerProgress], None] | None = None,
    ) -> None:
        grouped: dict[str, list[DownloadedAsset]] = defaultdict(list)
        for item in downloaded:
            target = self._target_for_asset(item.asset.name)
            grouped[str(target)].append(item)

        backups: list[tuple[Path, Path]] = []
        group_total = len(grouped)
        try:
            for group_index, (target_str, items) in enumerate(grouped.items(), start=1):
                target = Path(target_str)
                self._emit(
                    progress_callback,
                    InstallerProgress(
                        phase="apply-target",
                        message=f"Applying files to {target.name}",
                        asset_name=", ".join(i.asset.name for i in items),
                        asset_index=group_index,
                        asset_total=group_total,
                    ),
                )

                if len(items) == 1:
                    item = items[0]
                    source_dir = self._single_root(item.extracted_path, item.asset.name)
                    applied_names = [item.asset.name]
                else:
                    merged_dir = self.paths.temp_dir / "staging" / f"merged_{target.name}"
                    if merged_dir.exists():
                        shutil.rmtree(merged_dir, ignore_errors=True)
                    merged_dir.mkdir(parents=True, exist_ok=True)
                    applied_names = []
                    for item in items:
                        part_dir = self._single_root(item.extracted_path, item.asset.name)
                        shutil.copytree(part_dir, merged_dir, dirs_exist_ok=True)
                        applied_names.append(item.asset.name)
                    source_dir = merged_dir

                backup = target.with_name(target.name + ".bak")
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
                if target.exists():
                    target.replace(backup)
                    backups.append((target, backup))
                if target.parent:
                    target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_dir), str(target))
                log.info("Applied assets %s to %s", ", ".join(applied_names), target)
        except Exception:
            log.exception("Asset apply failed. Rolling back.")
            self._emit(
                progress_callback,
                InstallerProgress(
                    phase="rollback",
                    message="Update failed, restoring previous files",
                ),
            )
            for target, backup in reversed(backups):
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                if backup.exists():
                    backup.replace(target)
            raise
        else:
            for _, backup in backups:
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
