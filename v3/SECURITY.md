# Security Overview

## Scope

This document is public and intentionally excludes secrets and private operational procedures.

## Responsible Disclosure

If you find a security issue, report it privately to the project owner. Do not open a public issue with exploit details.

## Current Security Controls

- Signed update manifest:
  - `manifest.v1.json` is verified with Ed25519 before use.
  - Unknown key IDs or invalid signatures are rejected.
- Trusted update origins:
  - Manifest and asset URLs must resolve to trusted hosts.
  - Untrusted hosts or insecure schemes are rejected by default.
- Strict manifest validation:
  - Schema version is enforced.
  - Required fields and asset metadata are validated.
- Safe update extraction:
  - Archive traversal paths are blocked.
  - Absolute/drive paths are blocked.
  - Symlink entries are blocked.
- Forward-only updates:
  - Downgrades and non-forward updates are rejected.
- Launcher-gated app execution:
  - Main app requires a launcher-issued launch ticket.
  - Ticket validation enforces signature, expiry, and status.
  - Ticket nonces are one-time-use to prevent replay.
- Credential protection:
  - Sensitive launcher secrets are stored via Windows Credential Manager when available.
  - DPAPI-protected file fallback is used when Credential Manager is unavailable.

## Out Of Scope For Public Docs

- Private signing key material and key backup/rotation internals.
- Internal incident response and rollback playbooks.
- Abuse-detection thresholds and anti-tamper heuristics.
