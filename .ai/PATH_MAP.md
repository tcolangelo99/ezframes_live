# EzFrames Path Map (Canonical)

This map defines human-readable workspace labels and their actual paths.
If any instruction conflicts with this file, this file wins.

## Canonical Labels

- `ezframes1.0_dev`
  - Path: `G:\AFTER EFFECTS\ezframes`
  - Purpose: legacy v1-era duplicate root (frozen).
  - Write policy: no persistent source/docs/policy authoring.

- `ezframes2.0_dev`
  - Path: `G:\AFTER EFFECTS\ezframes2.0`
  - Purpose: legacy v2-era local workspace + staging parent.
  - Write policy: no canonical authoring; keep as support workspace.

- `ezframes3.0_dev` (staging)
  - Path: `G:\AFTER EFFECTS\ezframes2.0\master`
  - Purpose: v3 staging mirror for pre-prod validation.
  - Write policy: mirror output only; refresh from canonical with sync scripts.

- `ezframes_prod`
  - Path: `G:\AFTER EFFECTS\EZFRAMES_PROD\ezframes`
  - Purpose: canonical git source of truth for EzFrames.
  - Write policy: all persistent source/docs/policy changes originate here.

## Additional Path (Often Confusing)

- `ezframes_v3runtime_worktree`
  - Path: `G:\AFTER EFFECTS\EZFRAMES_PROD\ezframes_v3runtime`
  - Purpose: git worktree (alternate checkout) of canonical repo, currently on branch `v3-runtime`.
  - Not a separate product/runtime root.
  - Keep only if you intentionally need simultaneous branch checkouts; otherwise retire after validation.

## Required Flow

1. Author in `ezframes_prod`.
2. For managed v3 scope, run:
   - `py -3.11 scripts/sync_staging.py --mode sync --prune`
   - `py -3.11 scripts/sync_staging.py --mode verify`
3. Before release/handoff, run:
   - `py -3.11 scripts/audit_workspace_entropy.py --verify-staging`
