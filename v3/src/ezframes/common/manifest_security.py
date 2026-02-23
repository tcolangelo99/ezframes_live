from __future__ import annotations

import base64
import json
from pathlib import PurePosixPath
from typing import Iterable
from urllib.parse import urlparse

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


MANIFEST_SCHEMA_VERSION = "1"
MANIFEST_SIGNATURE_ALG = "ed25519"
DEFAULT_MANIFEST_KEY_ID = "ezframes-prod-2026-02"

# Trusted release hosts for manifest and redirected GitHub release assets.
TRUSTED_RELEASE_HOSTS: tuple[str, ...] = (
    "github.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
    "github-releases.githubusercontent.com",
)

# Public verification keys embedded in launcher/app code.
MANIFEST_PUBLIC_KEYS: dict[str, str] = {
    DEFAULT_MANIFEST_KEY_ID: "rvYa9/XJrcdy+Za0p7e3Ll8Uf6n6wBztNzZ1V0IICvI=",
}

_CANONICAL_MANIFEST_FIELDS = (
    "schema_version",
    "app_version",
    "min_launcher_version",
    "published_at",
    "notes_url",
    "assets",
)


def _normalize_host(host: str) -> str:
    return str(host or "").strip().lower().rstrip(".")


def _is_allowed_host(host: str, allowed_hosts: Iterable[str]) -> bool:
    normalized = _normalize_host(host)
    if not normalized:
        return False
    allowed = {_normalize_host(v) for v in allowed_hosts}
    if normalized in allowed:
        return True
    return any(normalized.endswith("." + entry) for entry in allowed)


def validate_trusted_url(url: str, allowed_hosts: Iterable[str], allow_http: bool = False) -> None:
    parsed = urlparse(str(url))
    scheme = (parsed.scheme or "").lower()
    if scheme != "https" and not allow_http:
        raise ValueError(f"Untrusted URL scheme for update asset: {url}")
    host = parsed.hostname or ""
    if not _is_allowed_host(host, allowed_hosts):
        raise ValueError(f"Untrusted update host: {host or '<none>'}")


def _canonical_assets(raw_assets: object) -> list[dict[str, object]]:
    if not isinstance(raw_assets, list):
        raise ValueError("Manifest assets must be a list.")
    if not raw_assets:
        raise ValueError("Manifest assets list is empty.")

    output: list[dict[str, object]] = []
    names: set[str] = set()
    for idx, raw in enumerate(raw_assets):
        if not isinstance(raw, dict):
            raise ValueError(f"Manifest asset at index {idx} is not an object.")
        missing = [k for k in ("name", "url", "sha256", "size") if k not in raw]
        if missing:
            raise ValueError(f"Manifest asset at index {idx} missing fields: {missing}")

        name = str(raw["name"]).strip()
        if not name:
            raise ValueError(f"Manifest asset at index {idx} has empty name.")
        if name in names:
            raise ValueError(f"Manifest has duplicate asset name: {name}")
        names.add(name)

        output.append(
            {
                "name": name,
                "url": str(raw["url"]).strip(),
                "sha256": str(raw["sha256"]).strip().lower(),
                "size": int(raw["size"]),
            }
        )
    return output


def manifest_signing_payload(manifest: dict) -> dict:
    missing = [k for k in _CANONICAL_MANIFEST_FIELDS if k not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required fields for signature: {missing}")

    schema_version = str(manifest["schema_version"]).strip()
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema version {schema_version!r}; expected {MANIFEST_SCHEMA_VERSION!r}."
        )

    payload = {
        "schema_version": schema_version,
        "app_version": str(manifest["app_version"]).strip(),
        "min_launcher_version": str(manifest["min_launcher_version"]).strip(),
        "published_at": str(manifest["published_at"]).strip(),
        "notes_url": str(manifest["notes_url"]).strip(),
        "assets": _canonical_assets(manifest["assets"]),
    }
    return payload


def canonical_manifest_bytes(manifest: dict) -> bytes:
    payload = manifest_signing_payload(manifest)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _decode_private_key(value: str) -> Ed25519PrivateKey:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Manifest signing key is empty.")

    if text.startswith("-----BEGIN"):
        key = serialization.load_pem_private_key(text.encode("utf-8"), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("Manifest signing key must be an Ed25519 private key.")
        return key

    try:
        raw = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise ValueError("Manifest signing key must be PEM or base64-encoded raw Ed25519 key.") from exc
    if len(raw) != 32:
        raise ValueError("Base64 manifest signing key must decode to 32 bytes.")
    return Ed25519PrivateKey.from_private_bytes(raw)


def sign_manifest(manifest: dict, private_key_value: str, key_id: str) -> dict:
    payload = canonical_manifest_bytes(manifest)
    key = _decode_private_key(private_key_value)
    signature = key.sign(payload)

    signed = dict(manifest)
    signed["signature_alg"] = MANIFEST_SIGNATURE_ALG
    signed["signature_key_id"] = str(key_id).strip()
    signed["signature"] = base64.b64encode(signature).decode("ascii")
    return signed


def verify_manifest_signature(manifest: dict) -> None:
    alg = str(manifest.get("signature_alg", "")).strip().lower()
    key_id = str(manifest.get("signature_key_id", "")).strip()
    signature_b64 = str(manifest.get("signature", "")).strip()

    if alg != MANIFEST_SIGNATURE_ALG:
        raise ValueError(f"Unsupported manifest signature algorithm: {alg or '<missing>'}")
    if not key_id:
        raise ValueError("Manifest signature_key_id is missing.")
    if not signature_b64:
        raise ValueError("Manifest signature is missing.")

    key_b64 = MANIFEST_PUBLIC_KEYS.get(key_id)
    if not key_b64:
        raise ValueError(f"Manifest key ID {key_id!r} is not trusted.")

    try:
        signature = base64.b64decode(signature_b64, validate=True)
    except Exception as exc:
        raise ValueError("Manifest signature is not valid base64.") from exc

    try:
        pub_raw = base64.b64decode(key_b64, validate=True)
    except Exception as exc:
        raise ValueError(f"Embedded public key for {key_id!r} is invalid.") from exc

    if len(pub_raw) != 32:
        raise ValueError(f"Embedded public key for {key_id!r} must be 32 bytes.")

    payload = canonical_manifest_bytes(manifest)
    pub = Ed25519PublicKey.from_public_bytes(pub_raw)
    try:
        pub.verify(signature, payload)
    except Exception as exc:
        raise ValueError("Manifest signature verification failed.") from exc


def validate_archive_member_path(member_name: str) -> PurePosixPath:
    # Normalize as posix to avoid platform-dependent traversal quirks.
    normalized = str(member_name or "").replace("\\", "/").strip()
    if not normalized:
        raise ValueError("Archive contains an empty path entry.")

    path = PurePosixPath(normalized)
    parts = path.parts
    if not parts:
        raise ValueError("Archive path entry has no parts.")
    if path.is_absolute():
        raise ValueError(f"Archive entry is absolute path: {member_name}")
    if any(part in {"..", ""} for part in parts):
        raise ValueError(f"Archive entry contains traversal segment: {member_name}")
    if ":" in parts[0]:
        raise ValueError(f"Archive entry contains drive designator: {member_name}")
    return path
