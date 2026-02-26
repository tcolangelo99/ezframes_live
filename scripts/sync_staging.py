from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path, PurePosixPath


ROOT_FILES = (
    ".gitignore",
    "bootstrap_launcher.py",
    "ezcutie_v3.py",
    "launcher_v3.py",
    "pyproject.toml",
    "requirements_v3.txt",
)

MANAGED_DIRS = (
    "packaging",
    "scripts",
    "src",
    "tests",
)

SYNC_SUFFIXES = {
    ".cfg",
    ".cs",
    ".csproj",
    ".ini",
    ".iss",
    ".json",
    ".ps1",
    ".py",
    ".sln",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

EXCLUDED_DIR_PARTS = {
    "__pycache__",
    ".venv",
    ".tmp_entitlement_test",
    ".tmp_entitlement_test2",
    ".tmp_force_update_test",
    ".tmp_force_update_test2",
    ".dev_install",
    "build",
    "dist",
    "release",
    "runtime_bootstrap",
    "runtime_env",
    "runtime_python",
    "state",
    "tmp_cutie_lite_color",
}

EXCLUDED_PREFIXES = (
    "packaging/launcher_stub/bin/",
    "packaging/launcher_stub/obj/",
    "scripts/release/__pycache__/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync or verify canonical v3 source/config files into the local staging mirror."
    )
    parser.add_argument(
        "--mode",
        choices=("verify", "sync"),
        default="verify",
        help="Use sync to copy canonical files into staging, or verify to report drift only.",
    )
    parser.add_argument(
        "--source-v3",
        default=None,
        help="Canonical v3 source path (default: <repo>/v3).",
    )
    parser.add_argument(
        "--staging-master",
        default=None,
        help="Staging mirror master path (default: <workspace>/ezframes2.0/master).",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete extra files in managed staging paths that are not in canonical source.",
    )
    return parser.parse_args()


def default_source_v3() -> Path:
    return Path(__file__).resolve().parents[1] / "v3"


def default_staging_master() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parents[1]
    return workspace_root / "ezframes2.0" / "master"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_managed_path(rel: PurePosixPath) -> bool:
    rel_text = rel.as_posix()

    for prefix in EXCLUDED_PREFIXES:
        if rel_text.startswith(prefix):
            return False

    if any(part in EXCLUDED_DIR_PARTS for part in rel.parts):
        return False

    if rel.name.endswith(".md"):
        return False

    if rel.name == ".gitignore":
        return True

    return rel.suffix.lower() in SYNC_SUFFIXES


def collect_source_manifest(source_v3: Path) -> dict[str, Path]:
    manifest: dict[str, Path] = {}

    for rel_text in ROOT_FILES:
        rel = PurePosixPath(rel_text)
        src = source_v3 / rel_text
        if src.is_file() and is_managed_path(rel):
            manifest[rel.as_posix()] = src

    for dir_name in MANAGED_DIRS:
        root = source_v3 / dir_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = PurePosixPath(path.relative_to(source_v3).as_posix())
            if is_managed_path(rel):
                manifest[rel.as_posix()] = path

    return manifest


def collect_target_manifest(staging_master: Path) -> dict[str, Path]:
    manifest: dict[str, Path] = {}

    for rel_text in ROOT_FILES:
        rel = PurePosixPath(rel_text)
        path = staging_master / rel_text
        if path.is_file() and is_managed_path(rel):
            manifest[rel.as_posix()] = path

    for dir_name in MANAGED_DIRS:
        root = staging_master / dir_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = PurePosixPath(path.relative_to(staging_master).as_posix())
            if is_managed_path(rel):
                manifest[rel.as_posix()] = path

    return manifest


def compare_manifests(
    source_manifest: dict[str, Path],
    target_manifest: dict[str, Path],
    staging_master: Path,
) -> tuple[list[str], list[str], list[str]]:
    missing: list[str] = []
    mismatched: list[str] = []

    for rel, src in sorted(source_manifest.items()):
        dst = staging_master / rel
        if not dst.exists():
            missing.append(rel)
            continue
        if sha256(src) != sha256(dst):
            mismatched.append(rel)

    extras = sorted(set(target_manifest) - set(source_manifest))
    return missing, mismatched, extras


def sync_files(source_manifest: dict[str, Path], staging_master: Path) -> tuple[int, int]:
    copied = 0
    unchanged = 0

    for rel, src in sorted(source_manifest.items()):
        dst = staging_master / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha256(src) == sha256(dst):
            unchanged += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    return copied, unchanged


def prune_extras(
    source_manifest: dict[str, Path],
    target_manifest: dict[str, Path],
    staging_master: Path,
) -> int:
    pruned = 0
    extras = sorted(set(target_manifest) - set(source_manifest))
    for rel in extras:
        path = staging_master / rel
        if path.exists():
            path.unlink()
            pruned += 1
    return pruned


def print_preview(header: str, items: list[str], limit: int = 25) -> None:
    if not items:
        return
    print(header)
    for rel in items[:limit]:
        print(f"  - {rel}")
    remaining = len(items) - limit
    if remaining > 0:
        print(f"  - ... and {remaining} more")


def main() -> int:
    args = parse_args()
    source_v3 = Path(args.source_v3).resolve() if args.source_v3 else default_source_v3().resolve()
    staging_master = (
        Path(args.staging_master).resolve() if args.staging_master else default_staging_master().resolve()
    )

    if not source_v3.exists():
        print(f"ERROR: source v3 path not found: {source_v3}")
        return 2
    if not staging_master.exists():
        print(f"ERROR: staging master path not found: {staging_master}")
        return 2

    source_manifest = collect_source_manifest(source_v3)
    if not source_manifest:
        print("ERROR: source manifest is empty. Check include/exclude filters.")
        return 2

    if args.mode == "sync":
        copied, unchanged = sync_files(source_manifest, staging_master)
        target_manifest = collect_target_manifest(staging_master)
        pruned = 0
        if args.prune:
            pruned = prune_extras(source_manifest, target_manifest, staging_master)
            target_manifest = collect_target_manifest(staging_master)
        print(
            f"SYNC SUMMARY: copied={copied} unchanged={unchanged} pruned={pruned} "
            f"source_files={len(source_manifest)}"
        )

    target_manifest = collect_target_manifest(staging_master)
    missing, mismatched, extras = compare_manifests(source_manifest, target_manifest, staging_master)

    print(
        f"VERIFY SUMMARY: source_files={len(source_manifest)} target_files={len(target_manifest)} "
        f"missing={len(missing)} mismatched={len(mismatched)} extras={len(extras)}"
    )
    print_preview("Missing files:", missing)
    print_preview("Mismatched files:", mismatched)
    print_preview("Extra managed staging files:", extras)

    if missing or mismatched or (args.prune and extras):
        print("STAGING DRIFT DETECTED")
        return 1

    print("STAGING MIRROR IS IN SYNC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
