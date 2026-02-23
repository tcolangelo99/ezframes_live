# Internal Security Runbook (Template)

This template is safe to keep in the repo.  
Fill a private copy outside the public repo with real values.

## Private Values (Do Not Commit)

- Manifest signing private key location
- Recovery key backups
- Credential rotation contacts
- Emergency release credentials

## Incident Response Checklist

1. Triage report and reproduce.
2. Assess blast radius and affected versions.
3. Revoke/rotate impacted credentials or keys.
4. Ship fixed release and update manifest.
5. Publish user-facing advisory.
6. Record postmortem actions.

## Key Rotation Checklist

1. Generate new Ed25519 key pair.
2. Update trusted public key in launcher code.
3. Sign next release with new key ID.
4. Verify update flow from previous release.
5. Deprecate old key after migration window.
