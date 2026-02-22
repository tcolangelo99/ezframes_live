# EzFrames v3 Migration Notes

## What Was Implemented

- New package layout under `src/ezframes`:
  - `launcher/` (`auth_service`, `update_service`, `runtime_installer`, `migration_service`, `process_service`, `cli`)
  - `app/` (`motion_detection`, `video_io`, `mask_integration`, `rife_interpolation`, `ui_controller`)
  - `cutie_lite/` (native prompt/propagation/export flow without Qt)
  - `common/` (paths, state, logging, hashing, manifest types)
- Manifest-driven update model.
- Local install path model based on `%LocalAppData%\EzFrames`.
- Install state persistence (`state/install_state.v1.json`).
- AWS auth calls retained and hardened with retries/timeouts.
- Windows credential manager integration for minimal session marker.
- Inno Setup thin installer template in `packaging/inno`.
- Release scripts in `scripts/release`.
- App launch mode:
  - `python -m ezframes.app` defaults to the refactored v3 UI (`EZFRAMES_APP_MODE=new`).
  - Legacy parity exists only as fallback (`EZFRAMES_APP_MODE=legacy`).

## What Was Not Touched

- No GitHub repo/release was modified.
- No AWS infrastructure or endpoint behavior was changed.
- Legacy files (`launcher.py`, `ezcutie.py`, etc.) remain present for rollback.

## Run (Local Dev)

```powershell
pip install -e .
python -m ezframes.launcher --check-only --skip-update
python -m ezframes.app                     # new UI mode (default)
$env:EZFRAMES_APP_MODE='legacy'; python -m ezframes.app   # legacy fallback
```
