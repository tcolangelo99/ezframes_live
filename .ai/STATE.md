# STATE

## Project Overview

EzFrames runs from the v3 runtime architecture under `v3/` (launcher, app, updater, release tooling). Root-level v2 binaries remain only as a migration bridge so legacy users can update safely to current v3 releases. The canonical git source is `G:\AFTER EFFECTS\EZFRAMES_PROD\ezframes`; `G:\AFTER EFFECTS\ezframes2.0\master` is staging-only mirror and must be synchronized from canonical using `scripts/sync_staging.py`.

## How To Run

- Source/dev check from repo root:
  - PowerShell: `cd v3; $env:PYTHONPATH='src'; py -3.11 -m ezframes.launcher --check-only`
- Source/dev full launch:
  - PowerShell: `cd v3; $env:PYTHONPATH='src'; py -3.11 -m ezframes.launcher`
- Installed/bootstrap launch:
  - PowerShell: `& "$env:LOCALAPPDATA\EzFrames\EzFramesLauncher.exe"`

## How To Test

- Test command:
  - PowerShell: `cd v3; $env:PYTHONPATH='src'; py -3.11 -m unittest discover -s tests`
- Last known result on this machine (2026-02-25):
  - Launcher check-only: pass (`exit 0`)
  - Unit tests: fail (`exit 1`) with 2 errors in `tests/test_manifest.py` due strict trusted-host enforcement rejecting `example.com` fixtures.

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
- Critical restore state is available (AWS CLI, GitHub CLI auth file, manifest private key file path, local EzFrames runtime data).
- Workspace governance is documented in `.ai/WORKSPACE_GOVERNANCE.md` with canonical vs staging path ownership.
- Workspace naming and label mapping is canonicalized in `.ai/PATH_MAP.md`.
- `scripts/sync_staging.py` now provides deterministic `verify`/`sync --prune` mirror control for canonical `v3` -> staging `ezframes2.0\master`.
- `AGENTS.md` now includes a mandatory memory protocol (read order + required `.ai` updates) and explicit staging `sync` + `verify` closure rules.
- `scripts/audit_workspace_entropy.py` validates workspace root health and can enforce staging verify in one command.
- Duplicate root entropy controls are now in place via non-destructive marker policies:
  - `G:\AFTER EFFECTS\ezframes` marked as frozen duplicate (`AGENTS.md`, `README_FROZEN_DUPLICATE.md`).
  - `G:\AFTER EFFECTS\ezframes2.0\master` marked as non-canonical staging mirror (`AGENTS.md`, `README_STAGING_MIRROR.md`).
- `G:\AFTER EFFECTS\EZFRAMES_PROD\ezframes_v3runtime` is a linked git worktree checkout for branch `v3-runtime` (alternate checkout, not a separate canonical project root).

## In Progress

- Agent handoff documentation has been refreshed; execution priorities are in `.ai/TASKS.md`.
- Workspace entropy reduction phase is complete: staging docs are pointer stubs, managed source/config drift sync is enforced, and duplicate roots are policy-marked.
- Immediate focus for next implementation pass:
  - Fix manifest test fixtures to match trusted-host validation.
  - Finish root artifact cleanup and ignore coverage.
  - Add CI coverage for launcher/app and release smoke checks.

## Known Issues / Tech Debt

- `tests/test_manifest.py` currently fails against current trusted-host security behavior and must be updated.
- Repo has local/untracked runtime artifacts (for example `v3/.venv/`) that should not leak into commits.
- Root contributor docs are intentionally minimal; v3 operational detail lives in `v3/README_V3.md` and `.ai/HANDOFF.md`.
- Remaining hardening items stay tracked in `.ai/TASKS.md` (`NOW/NEXT/LATER`).

## Last Updated

- 2026-02-25: Refreshed run/test status, documented current test failures, and added handoff-ready state for next agent.
- 2026-02-25: Added workspace governance + staging sync tooling and reconciled staging code/config drift to canonical `v3`.
- 2026-02-25: Strengthened AGENTS memory protocol text and required `sync_staging.py` verify pass for managed v3 changes.
- 2026-02-25: Completed Plan 2 closure with non-destructive folder entropy controls and workspace audit script (`audit_workspace_entropy.py`), verified passing.
- 2026-02-25: Added `.ai/PATH_MAP.md` and wired AGENTS/operating docs to require consulting path map before planning or edits.
