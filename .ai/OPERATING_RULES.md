# EzFrames AI Operating Rules

## Core Contract

- Before starting any task, read `.ai/STATE.md` and `.ai/TASKS.md` (create them if missing).
- After any meaningful code/config/behavior/CI/build change, update both `.ai/STATE.md` and `.ai/TASKS.md`.
- Every task entry must contain Goal and Acceptance Criteria.
- Only mark a task done when acceptance criteria are met.
- Repository files are canonical memory. Chat context is not canonical.
- Never store secrets, private keys, credentials, or secret file paths in `.ai/*`.

## How To Operate

1. Read `AGENTS.md`, then this file, then `.ai/STATE.md` and `.ai/TASKS.md`.
2. Execute work.
3. Run checks relevant to the change.
4. Update `.ai/STATE.md` and `.ai/TASKS.md` before commit.

## Local Checks

- Tests:
  - PowerShell: `$env:PYTHONPATH='v3/src'; py -3.11 -m unittest discover -s v3/tests`
- AI memory pre-commit hook:
  - Install once: `py -3.11 -m pip install pre-commit`
  - Install hooks once per clone: `py -3.11 -m pre_commit install`
  - Run hooks manually: `py -3.11 -m pre_commit run --all-files`
  - `--no-verify` exists for git commit, but use only for emergencies and follow up immediately with a compliant commit.
