from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_path(path: Path) -> None:
    if not path.exists():
        return
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    install_root = Path(__file__).resolve().parent

    _prepend_path(install_root / "app" / "src")
    _prepend_path(install_root / "app")
    _prepend_path(install_root / "runtime" / "env")
    _prepend_path(install_root / "runtime" / "env" / "Lib" / "site-packages")

    os.environ.setdefault("EZFRAMES_INSTALL_ROOT", str(install_root))
    os.environ.setdefault("EZFRAMES_SOURCE_ROOT", str(install_root))

    from ezframes.launcher.cli import main as launcher_main

    return int(launcher_main())


if __name__ == "__main__":
    raise SystemExit(main())
