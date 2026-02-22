from __future__ import annotations

import logging
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ezframes.common.config import AppPaths, RuntimeConfig
from ezframes.common.hashing import sha256_file
from ezframes.common.types import ManifestAsset, ReleaseManifest


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadedAsset:
    asset: ManifestAsset
    archive_path: Path
    extracted_path: Path


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

    def install_from_manifest(self, manifest: ReleaseManifest) -> dict[str, str]:
        self.paths.ensure_layout()
        staging = self.paths.temp_dir / "staging"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        downloaded: list[DownloadedAsset] = []
        checksums: dict[str, str] = {}

        for asset in manifest.assets:
            archive = self._download_asset(asset, staging)
            digest = sha256_file(archive, chunk_size=self.runtime.download_chunk_size)
            if digest.lower() != asset.sha256.lower():
                raise RuntimeError(f"Checksum mismatch for {asset.name}: {digest} != {asset.sha256}")
            checksums[asset.name] = digest
            extracted = self._extract_archive(archive, staging)
            downloaded.append(DownloadedAsset(asset=asset, archive_path=archive, extracted_path=extracted))

        self._atomic_apply(downloaded)
        return checksums

    def _download_asset(self, asset: ManifestAsset, staging: Path) -> Path:
        destination = staging / asset.name
        log.info("Downloading asset %s", asset.name)
        with self.session.get(
            asset.url,
            stream=True,
            timeout=(self.runtime.connect_timeout_seconds, self.runtime.read_timeout_seconds),
        ) as resp:
            resp.raise_for_status()
            with destination.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=self.runtime.download_chunk_size):
                    if chunk:
                        fh.write(chunk)
        return destination

    def _extract_archive(self, archive: Path, staging: Path) -> Path:
        target = staging / f"{archive.stem}_extracted"
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target)
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

    def _atomic_apply(self, downloaded: list[DownloadedAsset]) -> None:
        grouped: dict[str, list[DownloadedAsset]] = defaultdict(list)
        for item in downloaded:
            target = self._target_for_asset(item.asset.name)
            grouped[str(target)].append(item)

        backups: list[tuple[Path, Path]] = []
        try:
            for target_str, items in grouped.items():
                target = Path(target_str)

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
