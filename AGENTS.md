# Agent Operating Gate (EzFrames Canonical)

## Memory Protocol (Mandatory)

1. Before proposing or implementing any change, read in this order:
   - `.ai/OPERATING_RULES.md`
   - `.ai/PATH_MAP.md`
   - `.ai/STATE.md`
   - `.ai/TASKS.md`
   - `.ai/HANDOFF.md`
   - `.ai/WORKSPACE_GOVERNANCE.md`
2. After completing any meaningful code/config/behavior/CI/build/doc change:
   - Update `.ai/STATE.md` to reflect the true current system state.
   - Update `.ai/TASKS.md` to reflect true status and next actions.
   - Append or refresh a concise dated handoff note in `.ai/HANDOFF.md`.
3. If chat context conflicts with `.ai/*` files, `.ai/*` files are source of truth.
4. Never rely on chat history as authoritative memory.

## Canonical And Drift Control

1. Canonical EzFrames source is `G:\AFTER EFFECTS\EZFRAMES_PROD\ezframes`.
2. `G:\AFTER EFFECTS\ezframes2.0\master` is staging mirror only. Do not author persistent changes there.
3. For any change in managed v3 mirror scope (`v3/src`, `v3/scripts`, `v3/packaging`, `v3/tests`, root v3 sync-managed config files), run:
   - `py -3.11 scripts/sync_staging.py --mode sync --prune`
   - `py -3.11 scripts/sync_staging.py --mode verify`
4. If mirror verify reports drift, do not treat work as complete until resolved.
5. Before release or handoff, run:
   - `py -3.11 scripts/audit_workspace_entropy.py --verify-staging`
