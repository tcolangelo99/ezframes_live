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

- [TODO] Harden v2 to v3 bridge integrity verification
  - Goal: Require signed-manifest and SHA256 verification in the v2 migrator before executing installer assets.
  - Acceptance Criteria:
    - Bridge validates manifest signature against embedded trusted key(s).
    - All downloaded installer assets are checksum-verified before execute.
    - Tampered asset test fails closed with user-facing error.
  - Owner: EzFrames Team

- [TODO] Add CI for launcher/app tests and release sanity
  - Goal: Block merges that break runtime startup, update flow, or release packaging basics.
  - Acceptance Criteria:
    - PR workflow runs automated tests for launcher/app modules.
    - PR workflow includes release-script smoke checks.
    - Failing checks block merge.
  - Owner: EzFrames Team

- [IN PROGRESS] Add root `.gitignore` and cleanup artifact tracking
  - Goal: Keep generated artifacts out of source control and reduce accidental noise in commits.
  - Acceptance Criteria:
    - Root `.gitignore` excludes build and packaging outputs/caches.
    - Local-only `.ai/PRIVATE_NOTES.md` is gitignored for sensitive operational notes.
    - Existing transient directories are cleaned from tracked state.
    - Build artifact policy is documented in repo docs.
  - Owner: EzFrames Team

## NEXT

- [TODO] Enforce strict production interpreter selection
  - Goal: Ensure release launcher only executes managed runtime Python unless explicit developer mode is enabled.
  - Acceptance Criteria:
    - Production launch path refuses fallback to system Python.
    - Developer override remains explicit and logged.
  - Owner: EzFrames Team

- [TODO] Add update disk-space preflight checks
  - Goal: Prevent partial/corrupt update applies due to insufficient local storage.
  - Acceptance Criteria:
    - Updater computes required space before download/apply.
    - User gets clear remediation message when space is insufficient.
  - Owner: EzFrames Team

- [TODO] Add log rotation and support bundle export
  - Goal: Improve diagnostics while capping local disk growth.
  - Acceptance Criteria:
    - Logs rotate with size/count limits.
    - Single-click support bundle export includes logs and sanitized state files.
  - Owner: EzFrames Team

- [TODO] Harden auth endpoint/channel separation
  - Goal: Avoid accidental production usage of test endpoints/configuration.
  - Acceptance Criteria:
    - Endpoint set is channel-bound and explicit.
    - Production channel cannot resolve testing endpoints.
  - Owner: EzFrames Team

## LATER

- [TODO] Add manifest key rotation strategy
  - Goal: Support secure key rollover without breaking existing installs.
  - Acceptance Criteria:
    - Multiple trusted key IDs supported for verification.
    - Rotation runbook documented and tested in staging.
  - Owner: EzFrames Team

- [TODO] Automate migration smoke matrix
  - Goal: Catch regressions across fresh install and legacy upgrade paths.
  - Acceptance Criteria:
    - Automated checks cover fresh install, v3 to latest, v2 bridge to latest, and non-CUDA startup.
    - Results are visible in CI/release gates.
  - Owner: EzFrames Team
