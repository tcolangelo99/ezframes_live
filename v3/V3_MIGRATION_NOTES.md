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
- One-time launch tickets with nonce replay protection.
- Inno Setup thin installer template in `packaging/inno`.
- Release scripts in `scripts/release`.
- App launch mode:
  - `python -m ezframes.app` runs the refactored v3 UI only.
  - Legacy mode was removed from the runtime path.

## What Was Not Touched

- No GitHub repo/release was modified.
- No AWS infrastructure or endpoint behavior was changed.
- Legacy files (`launcher.py`, `ezcutie.py`, etc.) remain present for rollback.

## Run (Local Dev)

```powershell
pip install -e .
python -m ezframes.launcher --check-only
python -m ezframes.app
```
