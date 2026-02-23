from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ezframes.common.auth_tokens import (
    issue_launch_ticket,
    sign_cache_record,
    verify_cache_record,
)
from ezframes.common.config import AppPaths, RuntimeConfig

try:
    import win32cred  # type: ignore
except Exception:  # pragma: no cover - optional import at runtime
    win32cred = None


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthConfig:
    register_url: str = "https://pk4xdhzw1h.execute-api.eu-north-1.amazonaws.com/testing/RegisterUser"
    subscription_url: str = "https://5d2d56nba4.execute-api.eu-north-1.amazonaws.com/v1/check-subscription"
    cache_ttl_days: int = 7
    launch_ticket_ttl_seconds: int = 180


@dataclass(frozen=True)
class AuthSession:
    email: str
    subscription_status: str
    source: str


class AuthClient:
    def __init__(self, runtime: RuntimeConfig, cfg: AuthConfig | None = None):
        self.runtime = runtime
        self.cfg = cfg or AuthConfig()
        self.session = requests.Session()
        retry = Retry(
            total=runtime.max_retries,
            connect=runtime.max_retries,
            read=runtime.max_retries,
            status=runtime.max_retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @property
    def _timeout(self) -> tuple[int, int]:
        return (self.runtime.connect_timeout_seconds, self.runtime.read_timeout_seconds)

    def sign_in(self, email: str, password: str) -> bool:
        payload = {"email": email, "password": password}
        resp = self.session.post(self.cfg.register_url, json=payload, timeout=self._timeout)
        if resp.status_code != 200:
            log.warning("Sign-in failed with status %s", resp.status_code)
            return False
        data = resp.json()
        return data.get("message") == "Sign-in successful"

    def check_subscription_status(self, email: str, password: str) -> str:
        payload = {"email": email, "password": password}
        resp = self.session.post(self.cfg.subscription_url, json=payload, timeout=self._timeout)
        if resp.status_code != 200:
            log.warning("Subscription check failed with status %s", resp.status_code)
            return "INACTIVE"
        data = resp.json()
        return str(data.get("subscription_status", "INACTIVE")).upper()


class CredentialStore:
    PREFIX = "EzFrames/v3/"

    def write(self, key: str, username: str, secret: str) -> None:
        if win32cred is None:
            return
        blob = secret.encode("utf-16-le")
        cred = {
            "Type": win32cred.CRED_TYPE_GENERIC,
            "TargetName": self.PREFIX + key,
            "UserName": username,
            "CredentialBlob": blob,
            "Persist": win32cred.CRED_PERSIST_LOCAL_MACHINE,
            "Comment": "EzFrames session data",
        }
        win32cred.CredWrite(cred, 0)

    def read(self, key: str) -> tuple[str, str] | None:
        if win32cred is None:
            return None
        try:
            cred = win32cred.CredRead(self.PREFIX + key, win32cred.CRED_TYPE_GENERIC, 0)
        except Exception:
            return None
        username = str(cred.get("UserName", ""))
        secret_blob = cred.get("CredentialBlob", b"")
        if isinstance(secret_blob, bytes):
            secret = secret_blob.decode("utf-16-le", errors="ignore")
        else:
            secret = str(secret_blob)
        return username, secret


class AuthService:
    def __init__(self, paths: AppPaths, runtime: RuntimeConfig, cfg: AuthConfig | None = None):
        self.paths = paths
        self.cfg = cfg or AuthConfig(
            cache_ttl_days=int(os.environ.get("EZFRAMES_AUTH_CACHE_TTL_DAYS", "7")),
            launch_ticket_ttl_seconds=int(os.environ.get("EZFRAMES_LAUNCH_TICKET_TTL_SECONDS", "180")),
        )
        self.client = AuthClient(runtime, self.cfg)
        self.creds = CredentialStore()
        self.db_path = self.paths.state_dir / "auth_cache.db"
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS subscriptions (
                    user_email TEXT PRIMARY KEY,
                    last_check_date TEXT NOT NULL,
                    subscription_status TEXT NOT NULL,
                    signature TEXT NOT NULL DEFAULT ''
                )
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(subscriptions)").fetchall()
                if len(row) > 1
            }
            if "signature" not in columns:
                conn.execute("ALTER TABLE subscriptions ADD COLUMN signature TEXT NOT NULL DEFAULT ''")
            conn.commit()

    def _cache_status(self, email: str, status: str) -> None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        signature = sign_cache_record(self.paths.state_dir, email, date_str, status)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO subscriptions (user_email, last_check_date, subscription_status, signature)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_email) DO UPDATE SET
                    last_check_date=excluded.last_check_date,
                    subscription_status=excluded.subscription_status,
                    signature=excluded.signature
                """,
                (email, date_str, status, signature),
            )
            conn.commit()

    def _cached_active_session(self) -> AuthSession | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT user_email, last_check_date, subscription_status, signature FROM subscriptions LIMIT 1"
            ).fetchone()
        if not row:
            return None
        email, date_str, status, signature = row
        email = str(email).strip().lower()
        date_str = str(date_str).strip()
        status = str(status).strip().upper()
        signature = str(signature or "").strip()

        if status != "ACTIVE":
            return None
        if not verify_cache_record(self.paths.state_dir, email, date_str, status, signature):
            log.warning("Auth cache signature invalid; forcing re-authentication.")
            return None
        last = datetime.strptime(date_str, "%Y-%m-%d")
        if datetime.now() - last > timedelta(days=self.cfg.cache_ttl_days):
            return None
        return AuthSession(email=email, subscription_status=status, source="cache")

    def ensure_subscription_interactive(self) -> AuthSession | None:
        cached = self._cached_active_session()
        if cached:
            log.info("Subscription cache valid for %s", cached.email)
            return cached

        # GUI prompt on demand to keep pythonw UX friendly.
        import tkinter as tk
        from tkinter import messagebox, simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        email = simpledialog.askstring("EzFrames Login", "Email:", parent=root)
        if not email:
            root.destroy()
            return None
        password = simpledialog.askstring("EzFrames Login", "Password:", show="*", parent=root)
        if not password:
            root.destroy()
            return None

        try:
            if not self.client.sign_in(email, password):
                messagebox.showerror("Login Failed", "Invalid credentials or network error.")
                root.destroy()
                return None
            status = self.client.check_subscription_status(email, password)
            self._cache_status(email, status)
            if status != "ACTIVE":
                messagebox.showwarning("Subscription Inactive", "Your subscription is not active.")
                root.destroy()
                return None

            # Store minimal session marker in credential manager (no AWS changes).
            session_payload = json.dumps({"email": email, "status": status})
            self.creds.write("session", email, session_payload)
            messagebox.showinfo("Login Successful", "Access granted.")
            return AuthSession(email=email.strip().lower(), subscription_status="ACTIVE", source="aws")
        except requests.RequestException as exc:
            log.exception("Auth request failed: %s", exc)
            messagebox.showerror("Network Error", str(exc))
            return None
        finally:
            root.destroy()

    def issue_launch_ticket(self, session: AuthSession) -> Path:
        return issue_launch_ticket(
            self.paths.state_dir,
            email=session.email,
            status=session.subscription_status,
            ttl_seconds=self.cfg.launch_ticket_ttl_seconds,
        )
