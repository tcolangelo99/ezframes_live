# Workspace Governance

See `.ai/PATH_MAP.md` for canonical workspace labels and the human-facing naming model.

## Canonical Paths

- Canonical EzFrames repo (source of truth):
  - `G:\AFTER EFFECTS\EZFRAMES_PROD\ezframes`
- Canonical v3 runtime subtree:
  - `G:\AFTER EFFECTS\EZFRAMES_PROD\ezframes\v3`

## Staging Mirror Paths

- Staging root:
  - `G:\AFTER EFFECTS\ezframes2.0`
- Staging v3 mirror target:
  - `G:\AFTER EFFECTS\ezframes2.0\master`

## Active vs Frozen Roots

- Active for EzFrames runtime/release work:
  - `G:\AFTER EFFECTS\EZFRAMES_PROD\ezframes`
  - `G:\AFTER EFFECTS\ezframes2.0\master` (mirror output only)
- Frozen/legacy duplicate root:
  - `G:\AFTER EFFECTS\ezframes` (do not treat as canonical source)
- Out of scope for this governance:
  - `G:\AFTER EFFECTS\ezframes_plugin`

## Rules

1. Author source/docs/policy changes in canonical repo only.
2. Do not hand-edit staging mirror code/docs (`ezframes2.0\master`) for persistent changes.
3. Sync canonical v3 -> staging mirror using:
   - `py -3.11 scripts/sync_staging.py --mode sync --prune`
4. Verify mirror health using:
   - `py -3.11 scripts/sync_staging.py --mode verify`
5. Keep `.ai` memory canonical in `EZFRAMES_PROD\ezframes\.ai` only.
6. `G:\AFTER EFFECTS\ezframes_plugin` is not governed by this document.
7. Before release or handoff, run workspace audit:
   - `py -3.11 scripts/audit_workspace_entropy.py --verify-staging`

## Managed Mirror Scope

`scripts/sync_staging.py` manages these canonical `v3` paths:

- Root files:
  - `.gitignore`
  - `bootstrap_launcher.py`
  - `ezcutie_v3.py`
  - `launcher_v3.py`
  - `pyproject.toml`
  - `requirements_v3.txt`
- Directories:
  - `src/`
  - `scripts/`
  - `packaging/`
  - `tests/`

Markdown docs are intentionally excluded from automated code sync and should remain pointer stubs in staging.

## Safety Posture

- Folder entropy reduction is non-destructive: no automated folder moves/deletes outside mirror-managed file scope.
- Legacy/frozen roots are controlled through marker docs and AGENTS policy, not destructive cleanup.
