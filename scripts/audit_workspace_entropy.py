from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit EzFrames workspace layout to reduce root-level drift/entropy."
    )
    parser.add_argument(
        "--workspace-root",
        default=None,
        help="Workspace root path (default inferred from canonical repo location).",
    )
    parser.add_argument(
        "--verify-staging",
        action="store_true",
        help="Run sync_staging verify as part of the audit.",
    )
    return parser.parse_args()


def infer_workspace_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parents[1]


def check_path(path: Path, expect_dir: bool = True) -> tuple[bool, str]:
    if expect_dir:
        if path.exists() and path.is_dir():
            return True, f"OK: {path}"
        return False, f"MISSING DIRECTORY: {path}"
    if path.exists() and path.is_file():
        return True, f"OK: {path}"
    return False, f"MISSING FILE: {path}"


def run_sync_verify(repo_root: Path) -> tuple[bool, str]:
    cmd = [sys.executable, str(repo_root / "scripts" / "sync_staging.py"), "--mode", "verify"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    output = "\n".join([line for line in [out, err] if line]).strip()
    if proc.returncode == 0:
        return True, output or "sync_staging verify passed"
    return False, output or "sync_staging verify failed"


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else infer_workspace_root().resolve()

    canonical_repo = workspace_root / "EZFRAMES_PROD" / "ezframes"
    canonical_v3 = canonical_repo / "v3"
    staging_root = workspace_root / "ezframes2.0"
    staging_master = staging_root / "master"
    legacy_root = workspace_root / "ezframes"
    plugin_root = workspace_root / "ezframes_plugin"

    checks: list[tuple[bool, str]] = []
    checks.append(check_path(canonical_repo))
    checks.append(check_path(canonical_v3))
    checks.append(check_path(staging_root))
    checks.append(check_path(staging_master))
    checks.append(check_path(plugin_root))

    checks.append(check_path(workspace_root / "AGENTS.md", expect_dir=False))
    checks.append(check_path(canonical_repo / "AGENTS.md", expect_dir=False))
    checks.append(check_path(canonical_repo / ".ai" / "PATH_MAP.md", expect_dir=False))
    checks.append(check_path(staging_root / "AGENTS.md", expect_dir=False))
    checks.append(check_path(staging_master / "AGENTS.md", expect_dir=False))
    checks.append(check_path(staging_master / "README_STAGING_MIRROR.md", expect_dir=False))

    if legacy_root.exists():
        checks.append(check_path(legacy_root / "AGENTS.md", expect_dir=False))
        checks.append(check_path(legacy_root / "README_FROZEN_DUPLICATE.md", expect_dir=False))

    # Staging master should stay non-git to avoid accidental dual-source editing.
    if (staging_master / ".git").exists():
        checks.append((False, f"UNEXPECTED GIT REPO: {staging_master}"))
    else:
        checks.append((True, f"OK (non-git staging mirror): {staging_master}"))

    if args.verify_staging:
        ok, message = run_sync_verify(canonical_repo)
        checks.append((ok, f"sync_staging verify:\n{message}"))

    failures = [msg for ok, msg in checks if not ok]

    print("Workspace entropy audit results:")
    for ok, msg in checks:
        prefix = "[PASS]" if ok else "[FAIL]"
        print(f"{prefix} {msg}")

    if failures:
        print("")
        print("AUDIT FAILED")
        return 1

    print("")
    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
