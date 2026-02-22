# EzFrames v2 -> v3 Bridge

This folder contains the source for the temporary bridge executable used on legacy `master`.

- Legacy launcher (`launcher.exe`) still launches `ezcutie.exe`.
- `ezcutie.exe` is now a migration bridge that:
  1. Tries to launch installed v3 from `%LocalAppData%\EzFrames`.
  2. If missing, downloads latest bootstrap installer assets from GitHub Releases.
  3. Runs installer silently and launches v3.
  4. Falls back to `ezcutie_legacy.exe` if migration fails.

Release dependency:
- Latest release in `tcolangelo99/ezframes_live` must include:
  - `*_bootstrap_installer.exe`
  - optional `*_bootstrap_installer-*.bin`

Environment override (debug only):
- `EZFRAMES_MIGRATION_REPO=<owner/repo>` to point the bridge at a different repo.
