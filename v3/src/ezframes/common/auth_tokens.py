from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import json
import logging
import os
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

try:
    import win32cred  # type: ignore
except Exception:  # pragma: no cover
    win32cred = None


log = logging.getLogger(__name__)

_CRED_PREFIX = "EzFrames/v3/"
_SIGNING_KEY_TARGET = "auth-signing-key"
_LEGACY_SIGNING_KEY_FILE = "auth_signing_key.bin"
_SECURE_SIGNING_KEY_FILE = "auth_signing_key.dpapi"
_USED_NONCES_FILE = "used_launch_nonces.v1.json"
_TICKET_SCHEMA_VERSION = "1"
_TICKET_PREFIX = "launch_ticket"
_ALLOWED_TICKET_STATUSES = {"ACTIVE", "FREE"}


@dataclass(frozen=True)
class TicketValidation:
    valid: bool
    email: str | None = None
    status: str | None = None
    reason: str | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _b64_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    padded = value + "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _read_cred_secret(target: str) -> bytes | None:
    if win32cred is None:
        return None
    try:
        cred = win32cred.CredRead(_CRED_PREFIX + target, win32cred.CRED_TYPE_GENERIC, 0)
    except Exception:
        return None
    blob = cred.get("CredentialBlob", b"")
    if isinstance(blob, bytes):
        text = blob.decode("utf-16-le", errors="ignore")
    else:
        text = str(blob)
    text = text.strip()
    if not text:
        return None
    try:
        return _b64_decode(text)
    except Exception:
        return None


def _write_cred_secret(target: str, secret: bytes) -> bool:
    if win32cred is None:
        return False
    encoded = _b64_encode(secret).encode("utf-16-le")
    cred = {
        "Type": win32cred.CRED_TYPE_GENERIC,
        "TargetName": _CRED_PREFIX + target,
        "UserName": "ezframes",
        "CredentialBlob": encoded,
        # User-level persistence avoids admin requirements.
        "Persist": win32cred.CRED_PERSIST_LOCAL_MACHINE,
        "Comment": "EzFrames signing key",
    }
    try:
        win32cred.CredWrite(cred, 0)
        return True
    except Exception:
        return False


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _blob_from_bytes(value: bytes) -> tuple[_DATA_BLOB, ctypes.Array]:
    if not value:
        value = b"\x00"
    buffer = ctypes.create_string_buffer(value)
    blob = _DATA_BLOB(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    return blob, buffer


def _dpapi_protect(raw: bytes) -> bytes:
    if os.name != "nt":
        raise RuntimeError("DPAPI protection is only available on Windows.")

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32

    in_blob, in_buf = _blob_from_bytes(raw)
    out_blob = _DATA_BLOB()
    if not crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(out_blob),
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)


def _dpapi_unprotect(raw: bytes) -> bytes:
    if os.name != "nt":
        raise RuntimeError("DPAPI unprotection is only available on Windows.")

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32

    in_blob, in_buf = _blob_from_bytes(raw)
    out_blob = _DATA_BLOB()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(out_blob),
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)


def _legacy_secret_file_path(state_dir: Path) -> Path:
    return state_dir / _LEGACY_SIGNING_KEY_FILE


def _secure_secret_file_path(state_dir: Path) -> Path:
    return state_dir / _SECURE_SIGNING_KEY_FILE


def _read_legacy_secret_file(state_dir: Path) -> bytes | None:
    path = _legacy_secret_file_path(state_dir)
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="ascii").strip()
    except Exception:
        return None
    if not content:
        return None
    try:
        return _b64_decode(content)
    except Exception:
        return None


def _remove_legacy_secret_file(state_dir: Path) -> None:
    path = _legacy_secret_file_path(state_dir)
    if not path.exists():
        return
    try:
        path.unlink()
    except Exception:
        log.warning("Failed to remove legacy signing key file: %s", path)


def _read_secure_secret_file(state_dir: Path) -> bytes | None:
    path = _secure_secret_file_path(state_dir)
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="ascii").strip()
    except Exception:
        return None
    if not content.startswith("DPAPI1:"):
        return None
    encoded = content.split(":", 1)[1].strip()
    if not encoded:
        return None
    try:
        protected = _b64_decode(encoded)
        return _dpapi_unprotect(protected)
    except Exception:
        return None


def _write_secure_secret_file(state_dir: Path, secret: bytes) -> bool:
    path = _secure_secret_file_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        protected = _dpapi_protect(secret)
        data = "DPAPI1:" + _b64_encode(protected)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(data, encoding="ascii")
        tmp.replace(path)
        return True
    except Exception:
        return False


def load_or_create_signing_key(state_dir: Path) -> bytes:
    key = _read_cred_secret(_SIGNING_KEY_TARGET)
    if key:
        return key

    key = _read_secure_secret_file(state_dir)
    if key:
        _write_cred_secret(_SIGNING_KEY_TARGET, key)
        return key

    key = _read_legacy_secret_file(state_dir)
    if key:
        persisted = _write_cred_secret(_SIGNING_KEY_TARGET, key)
        if not persisted:
            persisted = _write_secure_secret_file(state_dir, key)
        if not persisted:
            raise RuntimeError("Failed to migrate signing key into secure storage.")
        _remove_legacy_secret_file(state_dir)
        return key

    key = os.urandom(32)
    persisted = _write_cred_secret(_SIGNING_KEY_TARGET, key)
    if not persisted:
        persisted = _write_secure_secret_file(state_dir, key)
    if not persisted:
        raise RuntimeError("Failed to persist signing key in secure storage.")
    return key


def _sign_fields(key: bytes, *fields: str) -> str:
    payload = "\x1f".join(fields).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def sign_cache_record(state_dir: Path, email: str, date_str: str, status: str) -> str:
    key = load_or_create_signing_key(state_dir)
    return _sign_fields(
        key,
        "cache.v1",
        email.strip().lower(),
        date_str.strip(),
        status.strip().upper(),
    )


def verify_cache_record(state_dir: Path, email: str, date_str: str, status: str, signature: str) -> bool:
    if not signature:
        return False
    expected = sign_cache_record(state_dir, email, date_str, status)
    return hmac.compare_digest(expected, signature.strip())


def issue_launch_ticket(
    state_dir: Path,
    email: str,
    status: str = "ACTIVE",
    ttl_seconds: int = 180,
) -> Path:
    key = load_or_create_signing_key(state_dir)
    now = _utc_now()
    ttl = max(30, int(ttl_seconds))
    expires = now + timedelta(seconds=ttl)
    nonce = _b64_encode(os.urandom(18))

    email_norm = email.strip().lower()
    status_norm = status.strip().upper()
    if status_norm not in _ALLOWED_TICKET_STATUSES:
        allowed = ", ".join(sorted(_ALLOWED_TICKET_STATUSES))
        raise ValueError(f"Unsupported launch status '{status_norm}'. Allowed: {allowed}.")

    issued_at = now.isoformat()
    expires_at = expires.isoformat()
    signature = _sign_fields(
        key,
        "launch.v1",
        email_norm,
        status_norm,
        issued_at,
        expires_at,
        nonce,
    )

    state_dir.mkdir(parents=True, exist_ok=True)
    for stale in state_dir.glob(f"{_TICKET_PREFIX}.*.v1.json"):
        try:
            stale.unlink()
        except Exception:
            pass

    ticket = {
        "schema_version": _TICKET_SCHEMA_VERSION,
        "email": email_norm,
        "status": status_norm,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "nonce": nonce,
        "signature": signature,
    }
    ticket_path = state_dir / f"{_TICKET_PREFIX}.{nonce[:12]}.v1.json"
    tmp = ticket_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(ticket, fh, indent=2, sort_keys=True)
    tmp.replace(ticket_path)
    return ticket_path


def _parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _used_nonces_path(state_dir: Path) -> Path:
    return state_dir / _USED_NONCES_FILE


def _load_used_nonces(state_dir: Path) -> dict[str, str]:
    path = _used_nonces_path(state_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Used nonce registry is unreadable: {path}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"Used nonce registry has invalid structure: {path}")

    out: dict[str, str] = {}
    for key, value in raw.items():
        nonce = str(key).strip()
        expiry = str(value).strip()
        if nonce and expiry:
            out[nonce] = expiry
    return out


def _save_used_nonces(state_dir: Path, entries: dict[str, str]) -> None:
    path = _used_nonces_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _prune_used_nonces(entries: dict[str, str], now: datetime) -> dict[str, str]:
    pruned: dict[str, str] = {}
    for nonce, expires_raw in entries.items():
        expires_dt = _parse_utc(expires_raw)
        if expires_dt is None:
            continue
        if expires_dt >= now:
            pruned[nonce] = expires_dt.isoformat()
    return pruned


def validate_launch_ticket(
    state_dir: Path,
    ticket_path: Path,
    consume: bool = True,
    clock_skew_seconds: int = 60,
) -> TicketValidation:
    if not ticket_path.exists():
        return TicketValidation(valid=False, reason="Launch ticket not found.")

    try:
        with ticket_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return TicketValidation(valid=False, reason="Launch ticket is unreadable.")

    required = {"schema_version", "email", "status", "issued_at", "expires_at", "nonce", "signature"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        return TicketValidation(valid=False, reason=f"Launch ticket missing fields: {', '.join(missing)}")

    if str(payload.get("schema_version", "")).strip() != _TICKET_SCHEMA_VERSION:
        return TicketValidation(valid=False, reason="Unsupported launch ticket schema.")

    email = str(payload.get("email", "")).strip().lower()
    status = str(payload.get("status", "")).strip().upper()
    issued_at = str(payload.get("issued_at", "")).strip()
    expires_at = str(payload.get("expires_at", "")).strip()
    nonce = str(payload.get("nonce", "")).strip()
    signature = str(payload.get("signature", "")).strip()
    if not email or not issued_at or not expires_at or not nonce or not signature:
        return TicketValidation(valid=False, reason="Launch ticket contains empty required fields.")

    key = load_or_create_signing_key(state_dir)
    expected = _sign_fields(
        key,
        "launch.v1",
        email,
        status,
        issued_at,
        expires_at,
        nonce,
    )
    if not hmac.compare_digest(expected, signature):
        return TicketValidation(valid=False, reason="Launch ticket signature mismatch.")

    issued_dt = _parse_utc(issued_at)
    expires_dt = _parse_utc(expires_at)
    if issued_dt is None or expires_dt is None:
        return TicketValidation(valid=False, reason="Launch ticket timestamp is invalid.")

    now = _utc_now()
    skew = timedelta(seconds=max(0, int(clock_skew_seconds)))
    if issued_dt - now > skew:
        return TicketValidation(valid=False, reason="Launch ticket issued time is invalid.")
    if now - expires_dt > skew:
        return TicketValidation(valid=False, reason="Launch ticket expired.")
    if status not in _ALLOWED_TICKET_STATUSES:
        allowed = ", ".join(sorted(_ALLOWED_TICKET_STATUSES))
        return TicketValidation(valid=False, reason=f"Unsupported launch status '{status}'. Allowed: {allowed}.")

    try:
        used_nonces = _prune_used_nonces(_load_used_nonces(state_dir), now=now)
    except Exception as exc:
        return TicketValidation(valid=False, reason=f"Launch ticket nonce registry error: {exc}")
    if nonce in used_nonces:
        return TicketValidation(valid=False, reason="Launch ticket nonce was already used.")

    if consume:
        used_nonces[nonce] = expires_dt.isoformat()
        _save_used_nonces(state_dir, used_nonces)
        try:
            ticket_path.unlink()
        except Exception:
            log.warning("Failed to consume launch ticket: %s", ticket_path)

    return TicketValidation(valid=True, email=email, status=status)


def validate_launch_ticket_from_env(
    state_dir: Path,
    env: Mapping[str, str] | None = None,
    consume: bool = True,
) -> TicketValidation:
    env_map = env if env is not None else os.environ
    ticket_raw = env_map.get("EZFRAMES_LAUNCH_TICKET", "").strip()
    if not ticket_raw:
        return TicketValidation(valid=False, reason="Missing launch ticket.")
    return validate_launch_ticket(state_dir, Path(ticket_raw), consume=consume)
