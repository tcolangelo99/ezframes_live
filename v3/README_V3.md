# EzFrames v3 Runtime

This is the v3 migration scaffold for EzFrames:

- No PyInstaller runtime freezing.
- Installer-managed Python runtime in `%LocalAppData%\EzFrames`.
- User-facing launcher: `EzFramesLauncher.exe` (installed by Inno and used by shortcuts).
- Internal launcher module: `pythonw -m ezframes.launcher`.
- Main app module: `python -m ezframes.app` (normally launched by launcher with auth ticket).
- Native Cutie-lite entrypoint: `python -m ezframes.cutie_lite` (no Qt interactive demo).

Legacy mode is removed from the v3 runtime path.

See also:
- `SECURITY.md` for public security controls and disclosure guidance.
- `docs/INTERNAL_SECURITY_RUNBOOK_TEMPLATE.md` for private runbook structure (template only).

## Documentation Policy

- Public docs in this repo should describe architecture and security controls.
- Do not commit secrets, private keys, production credentials, or incident-response runbooks.
- Keep sensitive operational details in private/internal documentation outside this repo.

## Local Dev

```powershell
cd master
..\venv\Scripts\python.exe -m pip install -e .
set PYTHONPATH=src
..\venv\Scripts\python.exe -m ezframes.launcher --check-only
```

Direct launcher entrypoint (developer/internal):

```powershell
pythonw -m ezframes.launcher
```

## Notes

- AWS endpoints are unchanged and configurable via env vars.
- App launch is now ticket-gated by the launcher. Direct app invocation is blocked unless a valid launcher-issued ticket is present.
- Release/update source is manifest-driven and designed for GitHub Releases.
- Manifest authenticity is enforced with Ed25519 signature verification (trusted public key embedded in launcher/app code).
- Manifest and asset URLs are restricted to trusted hosts, with schema/version validation before apply.
- Update flow is forward-only (downgrades and non-forward updates are rejected).
- Update flow is mandatory and promptless for required versions:
  - if manifest target is newer, launcher applies update automatically;
  - if `min_launcher_version` is newer than current launcher, launcher still force-updates and relaunches.
- Launch tickets are one-time-use and nonce-protected to prevent replay.
- Signing keys are stored in Windows Credential Manager when available, with DPAPI-protected file fallback.
- Interrupted updates self-heal on next launch by restoring `.bak` targets before continuing.
- Production launch ignores `EZFRAMES_MANIFEST_URL` unless `EZFRAMES_DEV_ALLOW_MANIFEST_OVERRIDE=1` is explicitly set.
- `EZFRAMES_SOURCE_ROOT` can be set explicitly in development if your working directory is not the repo root.
- `Track with Cutie` uses native Cutie-lite only.
- First-run migration copies data from legacy installs, then cleans known legacy install roots (`%ProgramFiles%\\ezframes_live` and `Documents\\ezframes`) to remove old runtime artifacts.
- Launcher now self-repairs user shortcuts (`Start Menu`/`Desktop`) to point to `EzFramesLauncher.exe` and can stage the EXE from `app/` after updater-only installs.
- Managed runtime layout supports:
  - `runtime/python` (base CPython)
  - `runtime/env/Lib/site-packages` (portable dependency bundle)
  - `app/` (ezframes package payload)

## Entitlements (Free vs Pro)

Launcher auth now supports two launch ticket statuses:
- `ACTIVE` -> Pro tier
- `FREE` -> Free tier

Free tier behavior:
- Motion method is locked to `Absdiff`.
- Feature detector is locked to `ORB`.
- Full tuning sliders remain available for Absdiff + ORB pipeline.
- Pro-only actions are disabled: `Track with Cutie`, `Interpolate Last Output`.
- Pro-only outputs are disabled: `Use Cutie Masks`, `Output Masked Video (Alpha)`, `Output Image Sequence`, `Debug Output`.

Pro tier behavior:
- All current app features are enabled.

Operational flags:
- `EZFRAMES_ALLOW_FREE_TIER=0` disables launcher-side "Continue Free" button.
- App tier is passed via launcher ticket validation and exposed as `EZFRAMES_ENTITLEMENT` (`free`/`pro`) for runtime gating.
- During mandatory updates, a topmost updater window is always shown and remains visible through relaunch handoff.

## Production Build (No PyInstaller)

```powershell
cd master

# Prepare runtime/python folder from local CPython install
powershell .\scripts\release\prepare_runtime_python.ps1 -PyVersion 3.11

# Prepare runtime/env dependencies (CUDA-capable torch without CUDA toolkit install)
powershell .\scripts\release\build_runtime_env.ps1 -PythonExe .\runtime_python\python.exe -TorchFlavor cu124 -Slim

# Build release zips + manifest
powershell .\scripts\release\build_release.ps1 -Version 3.0.0 -RepoUrl https://github.com/tcolangelo99/ezframes -RuntimePythonDir .\runtime_python -RuntimeEnvDir .\runtime_env

# Build bootstrap payload for Inno installer
powershell .\scripts\release\build_bootstrap_payload.ps1 -RuntimePythonDir .\runtime_python -RuntimeEnvDir .\runtime_env
```

Upload the generated assets and `manifest.v1.json` to the matching GitHub release tag.

Manifest signing is required for production releases:

```powershell
$env:EZFRAMES_MANIFEST_SIGNING_KEY="<base64-raw-ed25519-private-key>"
$env:EZFRAMES_MANIFEST_KEY_ID="ezframes-prod-2026-02"
```

Build installer (Inno Setup compiler `ISCC.exe`):

```powershell
ISCC .\packaging\inno\ezframes_bootstrap.iss /DMyAppVersion=3.0.0
```

If installer payload exceeds 2.1GB, Inno emits a spanned set. Ship all generated files together:
- `ezframes_v3_bootstrap_installer.exe`
- `ezframes_v3_bootstrap_installer-*.bin`

### CUDA Notes

- Users still need a compatible NVIDIA driver.
- Users do **not** need CUDA Toolkit installed when using PyTorch CUDA wheels (`cu12x`) in `runtime/env`.
- OpenCV CUDA acceleration requires a CUDA-enabled OpenCV build; default `opencv-python` wheels are CPU-only.
- RIFE and Cutie-lite now auto-fallback to CPU if CUDA init/runtime fails.
- You can force RIFE CPU mode with `EZFRAMES_RIFE_DEVICE=cpu`.

### Cutie-lite Controls

- Click: `Left` add, `Right` remove.
- Playback: `Previous`, `Play/Pause`, `Next` (also `Left`/`Right`/`Space` keys).
- Tracking: `Propagate Backward` and `Propagate Forward` are separate actions.
- Export: `Export Masks` writes `binary_masks` without forcing immediate tracking.
- Object target: `Object ID` spinner + `Clear Object` + `Reset Prompts`.
