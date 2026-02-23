# TASKS

## Task Template

- Status: `[TODO]` / `[IN PROGRESS]` / `[DONE]`
- Goal:
- Acceptance Criteria:
- Owner:

## NOW

- [DONE] Adopt AI memory mechanism (files + enforcement)
  - Goal: Add durable project memory files and enforce updates locally and in CI when code/config/CI/build changes happen.
  - Acceptance Criteria:
    - `AGENTS.md`, `.ai/OPERATING_RULES.md`, `.ai/STATE.md`, `.ai/TASKS.md` exist and are populated.
    - Local pre-commit enforcement blocks code commits missing `.ai/STATE.md` + `.ai/TASKS.md`.
    - GitHub Actions enforces the same rule for PR/push.
  - Owner: Codex

## NEXT

- [TODO] Add/confirm root contributor README entrypoint
  - Goal: Ensure contributors immediately land on current v3 runtime docs and conventions.
  - Acceptance Criteria:
    - Root docs clearly link to `v3/README_V3.md`.
    - Startup/testing/update entrypoints are visible from repo root.
  - Owner: EzFrames Team

- [TODO] Add tests for AI memory check scripts
  - Goal: Prevent regressions in file-pattern enforcement logic.
  - Acceptance Criteria:
    - Unit tests cover positive/negative path matching and memory-file requirements.
    - Tests run in CI.
  - Owner: EzFrames Team

## LATER

- [TODO] Reduce committed/generated artifact noise in repo root
  - Goal: Keep source tree focused on code and release metadata, not transient build outputs.
  - Acceptance Criteria:
    - Build output policy documented.
    - Generated directories are ignored or moved to dedicated artifact paths.
  - Owner: EzFrames Team
