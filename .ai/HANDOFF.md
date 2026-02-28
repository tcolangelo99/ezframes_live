# EzFrames Agent Handoff

## Read Order For Next Agent

1. `AGENTS.md`
2. `.ai/OPERATING_RULES.md`
3. `.ai/PATH_MAP.md`
4. `.ai/STATE.md`
5. `.ai/TASKS.md`
6. `.ai/WORKSPACE_GOVERNANCE.md`
7. `v3/README_V3.md`

## Current Snapshot (2026-02-28)

- Canonical repo: `G:\AFTER EFFECTS\EZFRAMES_PROD\ezframes`
- Active runtime: `v3/`
- v2 binaries remain for migration bridge only.
- Latest docs refresh includes run/test truth and execution order in `.ai/TASKS.md`.
- Canonical workspace naming map is now in `.ai/PATH_MAP.md`.
- Workspace governance + drift controls now exist (`.ai/WORKSPACE_GOVERNANCE.md`, `scripts/sync_staging.py`).
- Workspace entropy audit control now exists (`scripts/audit_workspace_entropy.py`).
- AGENTS memory protocol is now explicit about required `.ai` read order and post-change doc updates, plus staging mirror `sync` + `verify` closure.
- RIFE interpolation ffmpeg resolution was fixed in `v3/src/motion_interpolation.py` to prioritize shared runtime/app lookup (`resolve_ffmpeg`) so bundled assets work on clean machines.

## Path Relationships (Important)

- `G:\AFTER EFFECTS\EZFRAMES_PROD\ezframes`
  - The only canonical git repo (tracked, versioned, push/pull source of truth).
- `G:\AFTER EFFECTS\EZFRAMES_PROD\ezframes\v3`
  - Canonical v3 runtime subtree inside the repo.
- `G:\AFTER EFFECTS\ezframes2.0\master`
  - Local staging/build working copy, **not** a git repository.
  - Must be synchronized from canonical v3 with `py -3.11 scripts/sync_staging.py --mode sync --prune`.
  - Staging docs are pointer stubs that redirect to canonical docs.
  - Changes here are not published unless present in canonical repo.
- `G:\AFTER EFFECTS\ezframes`
  - Frozen duplicate root (legacy/local backup posture).
  - Marked with policy files to prevent accidental authoring drift.

## Mirror Commands

- Verify canonical v3 vs staging mirror:
  - `py -3.11 scripts/sync_staging.py --mode verify`
- Apply canonical -> staging sync (with managed-file prune):
  - `py -3.11 scripts/sync_staging.py --mode sync --prune`
- Last known result (2026-02-28):
  - Sync result: `copied=1`, then verify clean (`missing=0`, `mismatched=0`, `extras=0`)
  - Standalone verify result: clean (`missing=0`, `mismatched=0`, `extras=0`)
- Workspace entropy audit:
  - `py -3.11 scripts/audit_workspace_entropy.py --verify-staging`
  - Last result (2026-02-28): `AUDIT PASSED`

## Verified Commands

- Launcher check-only:
  - `cd v3; $env:PYTHONPATH='src'; py -3.11 -m ezframes.launcher --check-only`
  - Last result: `exit 0`
- Unit tests:
  - `cd v3; $env:PYTHONPATH='src'; py -3.11 -m unittest discover -s tests`
  - Last result: `exit 1` (2 manifest host-validation fixture errors)

## Immediate Priorities

1. Fix manifest test fixtures for trusted-host enforcement.
2. Complete artifact hygiene and ignore coverage (`v3/.venv`, build outputs, staging leftovers).
3. Add CI checks for launcher/app tests and release smoke flow.
4. Run post-restore release smoke validation and record results.

## Known Risks

- Manifest tests currently lag behind security behavior; this can hide real regressions.
- Local runtime/build artifacts may pollute commits without stricter hygiene.
- CI currently does not provide broad regression coverage for launcher/update/release paths.

## Sensitive Data Rule

- Do not store secrets in tracked docs.
- Use `.ai/PRIVATE_NOTES.md` for local-only secret locations/operational notes.

## Suggested First Session Plan (Next Agent)

1. Reproduce failing manifest tests.
2. Patch tests to assert current trusted-host policy.
3. Run full `v3/tests` suite.
4. Update `.ai/STATE.md` + `.ai/TASKS.md` with results.
5. Open/prepare PR with test fixes and updated docs.
