from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_FILE_NAME = "install_state.v1.json"


@dataclass
class InstallState:
    installed_version: str | None = None
    asset_checksums: dict[str, str] = field(default_factory=dict)
    last_update_check_utc: str | None = None
    migration_completed: bool = False
    legacy_source_paths: list[str] = field(default_factory=list)
    migration_warnings: list[str] = field(default_factory=list)

    def touch_update_time(self) -> None:
        self.last_update_check_utc = datetime.now(timezone.utc).isoformat()


def _state_file(state_dir: Path) -> Path:
    return state_dir / STATE_FILE_NAME


def load_install_state(state_dir: Path) -> InstallState:
    path = _state_file(state_dir)
    if not path.exists():
        return InstallState()

    # Accept optional UTF-8 BOM for resilience against manual edits done by
    # Windows tooling that may emit BOM-prefixed UTF-8.
    with path.open("r", encoding="utf-8-sig") as fh:
        raw: dict[str, Any] = json.load(fh)

    return InstallState(
        installed_version=raw.get("installed_version"),
        asset_checksums=dict(raw.get("asset_checksums", {})),
        last_update_check_utc=raw.get("last_update_check_utc"),
        migration_completed=bool(raw.get("migration_completed", False)),
        legacy_source_paths=list(raw.get("legacy_source_paths", [])),
        migration_warnings=list(raw.get("migration_warnings", [])),
    )


def save_install_state(state_dir: Path, state: InstallState) -> None:
    path = _state_file(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(asdict(state), fh, indent=2, sort_keys=True)
    tmp.replace(path)
