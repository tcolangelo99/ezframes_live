from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ManifestAsset:
    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ReleaseManifest:
    schema_version: str
    app_version: str
    min_launcher_version: str
    published_at: str
    notes_url: str
    assets: tuple[ManifestAsset, ...]

