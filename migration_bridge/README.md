# EzFrames v2 -> v3 Bridge

This folder contains the source for the temporary bridge executable used on legacy `master`.

- Legacy launcher (`launcher.exe`) still launches `ezcutie.exe`.
- `ezcutie.exe` is now a migration bridge that:
  1. During legacy v2 handoff, always applies the latest v3 bootstrap installer from GitHub Releases.
  2. If launched outside legacy handoff, it can reuse an already-current `%LocalAppData%\EzFrames` runtime.
  3. Shows a top-most progress window during download/install so users can see migration status.
  4. Runs installer silently and launches v3.
  5. Exits with clear error messaging if migration fails (no legacy fallback).

Release dependency:
- Latest release in `tcolangelo99/ezframes_live` must include:
  - `*_bootstrap_installer.exe`
  - optional `*_bootstrap_installer-*.bin`

Environment override (debug only):
- `EZFRAMES_MIGRATION_REPO=<owner/repo>` to point the bridge at a different repo.
