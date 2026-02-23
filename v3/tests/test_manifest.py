from __future__ import annotations

import unittest

from ezframes.common.config import AppPaths, RuntimeConfig
from ezframes.launcher.update_service import UpdateService


class ManifestTests(unittest.TestCase):
    def test_parse_manifest(self) -> None:
        svc = UpdateService(AppPaths.default(), RuntimeConfig.from_env())
        manifest = svc._parse_manifest(
            {
                "schema_version": "1",
                "app_version": "3.0.0",
                "min_launcher_version": "3.0.0",
                "published_at": "2026-02-22T00:00:00Z",
                "notes_url": "https://example.com",
                "assets": [
                    {"name": "runtime-python-win64.zip", "url": "https://example.com/a.zip", "sha256": "00", "size": 1}
                ],
            }
        )
        self.assertEqual(manifest.app_version, "3.0.0")
        self.assertEqual(len(manifest.assets), 1)
        self.assertEqual(manifest.assets[0].name, "runtime-python-win64.zip")

    def test_launcher_version_compat(self) -> None:
        svc = UpdateService(AppPaths.default(), RuntimeConfig.from_env())
        manifest = svc._parse_manifest(
            {
                "schema_version": "1",
                "app_version": "3.0.0",
                "min_launcher_version": "99.0.0",
                "published_at": "2026-02-22T00:00:00Z",
                "notes_url": "https://example.com",
                "assets": [
                    {"name": "runtime-python-win64.zip", "url": "https://example.com/a.zip", "sha256": "00", "size": 1}
                ],
            }
        )
        self.assertFalse(svc.launcher_meets_min_version(manifest))


if __name__ == "__main__":
    unittest.main()
