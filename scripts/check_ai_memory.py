from __future__ import annotations

import fnmatch
import subprocess
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


def staged_files() -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("ERROR: failed to inspect staged files.", file=sys.stderr)
        print(proc.stderr.strip(), file=sys.stderr)
        raise SystemExit(2)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    files = staged_files()
    if not files:
        return 0

    normalized = {normalize(p) for p in files}
    changed_code_files = [p for p in files if is_code_change(p)]
    if not changed_code_files:
        return 0

    missing = [f for f in sorted(REQUIRED_MEMORY_FILES) if f not in normalized]
    if not missing:
        return 0

    print("ERROR: AI memory check failed.")
    print("Code/config/CI/build changes are staged, but required memory files are missing:")
    for item in missing:
        print(f"  - {item}")
    print("")
    print("Staged files that triggered this rule:")
    for item in changed_code_files[:15]:
        print(f"  - {item}")
    if len(changed_code_files) > 15:
        print(f"  - ... and {len(changed_code_files) - 15} more")
    print("")
    print("Fix:")
    print("  1) Update .ai/STATE.md and .ai/TASKS.md")
    print("  2) Stage them")
    print("  3) Commit again")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
