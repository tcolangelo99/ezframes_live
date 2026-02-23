from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import PurePosixPath


STATE_FILE = ".ai/state.md"
TASKS_FILE = ".ai/tasks.md"
REQUIRED_MEMORY_FILES = {STATE_FILE, TASKS_FILE}


def normalize(path: str) -> str:
    value = path.strip().replace("\\", "/")
    if value.startswith("./"):
        value = value[2:]
    return value.lower()


def is_code_change(path: str) -> bool:
    p = normalize(path)
    base = PurePosixPath(p).name

    if p.endswith((".py", ".ps1", ".iss", ".cs", ".csproj", ".sln")):
        return True

    if p in {"pyproject.toml", "setup.cfg", "setup.py", "pytest.ini", "tox.ini", "noxfile.py"}:
        return True

    if fnmatch.fnmatch(base, "requirements*.txt"):
        return True

    if p.startswith(("src/", "tests/", "v3/src/", "v3/tests/", "v3/scripts/")):
        return True

    if p.startswith(".github/") and p.endswith((".yml", ".yaml")):
        return True

    return False


def read_changed_files(path: str | None) -> list[str]:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    else:
        lines = sys.stdin.readlines()
    return [line.strip() for line in lines if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI check for required .ai memory file updates.")
    parser.add_argument(
        "--changed-files",
        default=None,
        help="Path to newline-separated changed files. If omitted, stdin is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = read_changed_files(args.changed_files)
    if not files:
        print("No changed files found; skipping AI memory enforcement.")
        return 0

    normalized = {normalize(p) for p in files}
    changed_code_files = [p for p in files if is_code_change(p)]
    if not changed_code_files:
        print("No code/config/CI/build changes detected; AI memory enforcement not required.")
        return 0

    missing = [f for f in sorted(REQUIRED_MEMORY_FILES) if f not in normalized]
    if not missing:
        print("AI memory enforcement passed.")
        return 0

    print("AI MEMORY CHECK FAILED")
    print("Code/config/CI/build files changed, but required memory files were not both updated:")
    for item in missing:
        print(f"  - {item}")
    print("")
    print("Changed files that triggered this rule:")
    for item in changed_code_files[:25]:
        print(f"  - {item}")
    if len(changed_code_files) > 25:
        print(f"  - ... and {len(changed_code_files) - 25} more")
    print("")
    print("Please update and commit both .ai/STATE.md and .ai/TASKS.md.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
