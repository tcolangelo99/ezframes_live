from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ezframes.common.state import InstallState, load_install_state, save_install_state


class StateTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = InstallState(
                installed_version="3.0.0",
                asset_checksums={"a.zip": "abc"},
                migration_completed=True,
                legacy_source_paths=["C:/old"],
            )
            save_install_state(root, state)
            loaded = load_install_state(root)
            self.assertEqual(loaded.installed_version, "3.0.0")
            self.assertEqual(loaded.asset_checksums["a.zip"], "abc")
            self.assertTrue(loaded.migration_completed)
            self.assertEqual(loaded.legacy_source_paths[0], "C:/old")


if __name__ == "__main__":
    unittest.main()

