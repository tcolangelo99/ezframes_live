# EzFrames AI Operating Rules

## Core Contract

- Before starting any task, read `.ai/STATE.md` and `.ai/TASKS.md` (create them if missing).
- After any meaningful code/config/behavior/CI/build change, update both `.ai/STATE.md` and `.ai/TASKS.md`.
- Every task entry must contain Goal and Acceptance Criteria.
- Only mark a task done when acceptance criteria are met.
- Repository files are canonical memory. Chat context is not canonical.
- Never store secrets, private keys, credentials, or secret file paths in `.ai/*`.
- Store sensitive operational notes in local-only `.ai/PRIVATE_NOTES.md` (gitignored), not in tracked files.

## How To Operate

1. Read `AGENTS.md`, then this file, then `.ai/STATE.md` and `.ai/TASKS.md`.
2. Read `.ai/PATH_MAP.md` for canonical labels and path intent before planning.
3. Follow `.ai/WORKSPACE_GOVERNANCE.md` for canonical vs staging path rules.
4. Execute work.
5. Run checks relevant to the change.
6. Update `.ai/STATE.md` and `.ai/TASKS.md` before commit.

## Local Checks

- Tests:
  - PowerShell: `$env:PYTHONPATH='v3/src'; py -3.11 -m unittest discover -s v3/tests`
- Staging mirror drift check:
  - PowerShell: `py -3.11 scripts/sync_staging.py --mode verify`
- Workspace entropy audit (pre-release/handoff):
  - PowerShell: `py -3.11 scripts/audit_workspace_entropy.py --verify-staging`
- AI memory pre-commit hook:
  - Install once: `py -3.11 -m pip install pre-commit`
  - Install hooks once per clone: `py -3.11 -m pre_commit install`
  - Run hooks manually: `py -3.11 -m pre_commit run --all-files`
  - `--no-verify` exists for git commit, but use only for emergencies and follow up immediately with a compliant commit.
