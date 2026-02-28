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

- [DONE] Refresh handoff docs and clarify canonical vs staging paths
  - Goal: Ensure next agent can unambiguously identify source-of-truth repo and staging copies.
  - Acceptance Criteria:
    - `.ai/HANDOFF.md` includes explicit relationship between `EZFRAMES_PROD\ezframes`, `...\ezframes\v3`, and `ezframes2.0\master`.
    - `.ai/STATE.md` states that `ezframes2.0\master` is staging-only and may drift.
    - Root docs include handoff pointer.
  - Owner: Codex

- [DONE] Add workspace drift governance and staging sync controls
  - Goal: Reduce documentation/source entropy across canonical and staging paths by enforcing a one-way mirror flow.
  - Acceptance Criteria:
    - Canonical governance doc exists with path ownership + sync commands.
    - Staging docs (`ezframes2.0\master`) are pointer stubs to canonical docs.
    - `scripts/sync_staging.py` supports `verify` and `sync --prune` for canonical `v3` -> staging `master`.
    - Initial verify shows clean mirror after sync (`missing=0`, `mismatched=0`, `extras=0`).
  - Owner: Codex

- [DONE] Harden AGENTS memory protocol and mirror closure requirements
  - Goal: Force agents to update canonical memory docs after changes and require explicit staging mirror closure for managed v3 work.
  - Acceptance Criteria:
    - Canonical `AGENTS.md` defines mandatory pre-read order for `.ai` memory/governance files.
    - Canonical `AGENTS.md` requires post-change updates to `.ai/STATE.md`, `.ai/TASKS.md`, and concise handoff note updates.
    - Canonical and staging/global AGENTS text requires `sync_staging.py --mode sync --prune` and `--mode verify` for managed v3 changes.
  - Owner: Codex

- [DONE] Add canonical path map and require agent consultation
  - Goal: Eliminate path ambiguity by defining explicit workspace labels and forcing agents to consult mapping before work.
  - Acceptance Criteria:
    - `.ai/PATH_MAP.md` exists and maps `ezframes1.0_dev`, `ezframes2.0_dev`, `ezframes3.0_dev`, and `ezframes_prod` to actual paths.
    - AGENTS instructions (canonical + workspace wrappers) explicitly require consulting path map.
    - Operating/handoff docs include path map in startup read order.
  - Owner: Codex

- [DONE] Fix interpolation ffmpeg discovery for clean-machine installs
  - Goal: Ensure "Interpolate Last Output" works without requiring a system ffmpeg install.
  - Acceptance Criteria:
    - `v3/src/motion_interpolation.py` resolves ffmpeg via shared runtime/app path logic first.
    - Fallback lookup includes install/source roots (`EZFRAMES_SOURCE_ROOT`, `EZFRAMES_INSTALL_ROOT`, module ancestors, cwd).
    - Managed v3 staging mirror closure commands pass after change (`sync --prune`, `verify`).
  - Owner: Codex

- [DONE] Roll out interpolation ffmpeg hotfix to updater clients
  - Goal: Publish a signed release carrying the interpolation ffmpeg path fix so clients receive it through the manifest updater.
  - Acceptance Criteria:
    - `v3.0.12` GitHub release exists with `app-bundle-v3.0.12.zip` and `manifest.v1.json`.
    - `releases/latest/download/manifest.v1.json` advertises `app_version` `3.0.12`.
    - Release notes mention the `ffmpeg not found` interpolation fix context.
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

- [DONE] Add pre-release staging mirror verification gate
  - Goal: Prevent shipping from stale staging copies by requiring an explicit canonical-to-staging verification step.
  - Acceptance Criteria:
    - Release checklist includes `py -3.11 scripts/sync_staging.py --mode verify`.
    - Any mismatch blocks release packaging until canonical sync is applied.
    - Verification result location is documented for handoff/review.
  - Owner: EzFrames Team

- [DONE] Complete non-destructive folder entropy controls
  - Goal: Reduce accidental edits across duplicate workspace roots without risky folder moves/deletes.
  - Acceptance Criteria:
    - `G:\AFTER EFFECTS\ezframes` has frozen-duplicate marker policy docs.
    - `G:\AFTER EFFECTS\ezframes2.0\master` has explicit staging-mirror marker docs.
    - `scripts/audit_workspace_entropy.py --verify-staging` passes and reports healthy layout.
  - Owner: Codex

- [IN PROGRESS] Add root `.gitignore` and cleanup artifact tracking
  - Goal: Keep generated artifacts out of source control and reduce accidental noise in commits.
  - Acceptance Criteria:
    - Root `.gitignore` excludes build and packaging outputs/caches.
    - Local-only `.ai/PRIVATE_NOTES.md` is gitignored for sensitive operational notes.
    - Existing transient directories are cleaned from tracked state.
    - Build artifact policy is documented in repo docs.
  - Owner: EzFrames Team

- [TODO] Fix manifest unit tests for trusted-host validation
  - Goal: Align `tests/test_manifest.py` fixtures with current trusted update host restrictions.
  - Acceptance Criteria:
    - `py -3.11 -m unittest discover -s v3/tests` passes manifest tests.
    - Tests assert host-validation behavior explicitly (allowed + blocked hosts).
  - Owner: EzFrames Team

- [TODO] Run post-restore release smoke validation
  - Goal: Confirm rebuild environment can execute check-only launch and release scripts end-to-end after machine restore.
  - Acceptance Criteria:
    - Launcher check-only succeeds from source on clean shell.
    - Release script dry-run checklist executed and results recorded in `.ai/HANDOFF.md`.
    - Any missing prereq/tooling is captured with remediation commands.
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

## Handoff Execution Order

1. Resolve manifest test failures (`NOW` item: trusted-host fixtures).
2. Finish artifact hygiene (`NOW` item: root ignore cleanup).
3. Add CI test/smoke gates (`NOW` item: launcher/app/release checks).
4. Execute post-restore smoke validation and document output.
5. Continue into `NEXT` hardening items.
