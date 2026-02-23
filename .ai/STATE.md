# STATE

## Project Overview

EzFrames currently runs a v3 runtime architecture from `v3/` (launcher, app, updater, release tooling), while root-level v2 binaries remain only as a migration bridge so legacy users can update safely to the latest v3 release.

## How To Run

- Source/dev mode (from repo root):
  - PowerShell: `$env:PYTHONPATH='v3/src'; py -3.11 -m ezframes.launcher --check-only`
- Installed/bootstrap mode (if runtime payload is present):
  - PowerShell: `.\v3\runtime_bootstrap\EzFramesLauncher.exe`

## How To Test

- PowerShell: `$env:PYTHONPATH='v3/src'; py -3.11 -m unittest discover -s v3/tests`
- Current status on this machine: command fails unless `cryptography` is installed in the active Python environment.

## Current Architecture

- `v3/src/ezframes/launcher/`: auth, update, runtime install/apply, migration, process launch.
- `v3/src/ezframes/app/`: main EzFrames UI and processing pipeline.
- `v3/src/ezframes/cutie_lite/`: native Cutie-lite tracking/propagation/export flow.
- `v3/scripts/release/` + `v3/packaging/`: release assets, manifest signing, launcher stub, Inno installer.
- Root `launcher.exe` and `ezcutie.exe`: legacy v2 bridge binaries used for v2 to v3 migration handoff.

## What Works

- Mandatory v3 manifest-driven updates from GitHub Releases.
- v2 to v3 bridge path with progress UI and installer handoff.
- Launcher auth supports both Pro and Free tier entitlements.
- Local/runtime security hardening and signed manifest verification are in place.
- Public `.ai` memory files remain sanitized; sensitive operational data is routed to local-only `.ai/PRIVATE_NOTES.md`.

## In Progress

- Robustness hardening backlog has been queued in `.ai/TASKS.md` and is ready for implementation sequencing.
- Immediate priority group:
  - Harden v2 to v3 bridge integrity verification.
  - Add CI for launcher/app tests plus release sanity checks.
  - Add root `.gitignore` and artifact cleanup policy.

## Known Issues / Tech Debt

- Repo root contains generated build artifacts (`build/`, `dist/`, installer binaries/spec files) that should stay out of committed source changes.
- Root contributor docs are minimal; primary operational docs currently live in `v3/README_V3.md`.
- Remaining hardening items are tracked in `.ai/TASKS.md` under `NEXT` and `LATER` (interpreter strictness, disk-space preflight, log rotation, endpoint/channel separation, key rotation, migration matrix automation).

## Last Updated

- 2026-02-23: Added root ignore policy and local-only private notes path for sensitive operational details.
