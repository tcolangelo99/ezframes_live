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
    cache_ttl_days: int = 0
    launch_ticket_ttl_seconds: int = 180
    allow_free_tier: bool = True


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
            cache_ttl_days=int(os.environ.get("EZFRAMES_AUTH_CACHE_TTL_DAYS", "0")),
            launch_ticket_ttl_seconds=int(os.environ.get("EZFRAMES_LAUNCH_TICKET_TTL_SECONDS", "180")),
            allow_free_tier=os.environ.get("EZFRAMES_ALLOW_FREE_TIER", "1").strip() not in {"0", "false", "False"},
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
        if int(self.cfg.cache_ttl_days) <= 0:
            return None
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

    def _icon_candidates(self) -> list[Path]:
        candidates: list[Path] = [
            self.paths.assets_dir / "icons" / "ezframes_icon.ico",
            self.paths.assets_dir / "icons" / "launcher_icon.ico",
            self.paths.install_root / "icons" / "ezframes_icon.ico",
        ]
        for root in self.paths.source_roots():
            candidates.append(root / "icons" / "ezframes_icon.ico")
            candidates.append(root / "ezframes_icon.ico")
            candidates.append(root / "launcher_icon.ico")
        return candidates

    def _apply_window_icon(self, window) -> None:
        for candidate in self._icon_candidates():
            try:
                if candidate.exists():
                    window.iconbitmap(str(candidate))
                    return
            except Exception:
                continue

    def _last_cached_email(self) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT user_email FROM subscriptions ORDER BY last_check_date DESC LIMIT 1"
                ).fetchone()
            if not row:
                return ""
            return str(row[0]).strip().lower()
        except Exception:
            return ""

    def ensure_subscription_interactive(self) -> AuthSession | None:
        cached = self._cached_active_session()
        if cached:
            log.info("Subscription cache valid for %s", cached.email)
            return cached

        # Visible launcher-style auth window for pythonw UX.
        import tkinter as tk
        import tkinter.font as tkfont

        def pick_font(candidates: list[str], fallback: str = "Segoe UI") -> str:
            try:
                available = {name.lower() for name in tkfont.families()}
            except Exception:
                available = set()
            if not available:
                return candidates[0] if candidates else fallback
            for candidate in candidates:
                if candidate.lower() in available:
                    return candidate
            return fallback

        colors = {
            "bg": "#1A1A1A",
            "panel": "#242424",
            "panel_alt": "#313131",
            "text": "#FFFFFF",
            "muted": "#BCC7D0",
            "accent": "#00F0FF",
            "primary": "#175482",
            "primary_hover": "#2A6D9E",
            "secondary": "#394955",
            "secondary_hover": "#6D8B93",
            "danger": "#D16A74",
        }
        body_font = pick_font(["Segoe UI", "Calibri", "Arial"])
        accent_font = pick_font(["WWE Raw", "Segoe UI Semibold", "Segoe UI"], fallback=body_font)

        root = tk.Tk()
        root.title("EzFrames Launcher")
        root.attributes("-topmost", True)
        root.resizable(False, False)
        root.geometry("470x320")
        root.configure(bg=colors["bg"])
        self._apply_window_icon(root)

        frame = tk.Frame(
            root,
            bg=colors["panel"],
            highlightbackground=colors["panel_alt"],
            highlightthickness=1,
            bd=0,
        )
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        title_lbl = tk.Label(
            frame,
            text="EzFrames",
            bg=colors["panel"],
            fg=colors["text"],
            anchor="w",
            font=(accent_font, 24),
        )
        title_lbl.pack(fill="x", padx=12, pady=(10, 0))

        title2_lbl = tk.Label(
            frame,
            text="Sign in for Pro features",
            bg=colors["panel"],
            fg=colors["accent"],
            anchor="w",
            font=(body_font, 11, "bold"),
        )
        title2_lbl.pack(fill="x", padx=12, pady=(2, 0))

        subtitle_lbl = tk.Label(
            frame,
            text="You can continue in Free mode (Absdiff + ORB only) without signing in.",
            bg=colors["panel"],
            fg=colors["muted"],
            anchor="w",
            wraplength=390,
            justify="left",
            font=(body_font, 10),
        )
        subtitle_lbl.pack(fill="x", padx=12, pady=(4, 10))

        email_lbl = tk.Label(
            frame,
            text="Email",
            bg=colors["panel"],
            fg=colors["text"],
            anchor="w",
            font=(body_font, 10, "bold"),
        )
        email_lbl.pack(fill="x", padx=12, pady=(0, 2))
        email_var = tk.StringVar(value=self._last_cached_email())
        email_entry = tk.Entry(
            frame,
            textvariable=email_var,
            bg=colors["panel_alt"],
            fg=colors["text"],
            insertbackground=colors["text"],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=colors["panel_alt"],
            highlightcolor=colors["accent"],
            font=(body_font, 10),
        )
        email_entry.pack(fill="x", padx=12, ipady=6)

        password_lbl = tk.Label(
            frame,
            text="Password",
            bg=colors["panel"],
            fg=colors["text"],
            anchor="w",
            font=(body_font, 10, "bold"),
        )
        password_lbl.pack(fill="x", padx=12, pady=(8, 2))
        password_var = tk.StringVar(value="")
        password_entry = tk.Entry(
            frame,
            textvariable=password_var,
            show="*",
            bg=colors["panel_alt"],
            fg=colors["text"],
            insertbackground=colors["text"],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=colors["panel_alt"],
            highlightcolor=colors["accent"],
            font=(body_font, 10),
        )
        password_entry.pack(fill="x", padx=12, ipady=6)

        status_var = tk.StringVar(value="Enter your credentials.")
        status_lbl = tk.Label(
            frame,
            textvariable=status_var,
            bg=colors["panel"],
            fg=colors["muted"],
            anchor="w",
            justify="left",
            wraplength=430,
            font=(body_font, 10),
        )
        status_lbl.pack(fill="x", padx=12, pady=(10, 4))

        button_row = tk.Frame(frame, bg=colors["panel"])
        button_row.pack(fill="x", padx=12, pady=(4, 10))

        result: dict[str, AuthSession | None] = {"session": None}
        in_flight = {"value": False}

        def set_status(message: str, color: str = colors["muted"]) -> None:
            status_var.set(message)
            status_lbl.configure(fg=color)

        def set_controls(enabled: bool) -> None:
            state = tk.NORMAL if enabled else tk.DISABLED
            email_entry.configure(state=state)
            password_entry.configure(state=state)
            login_btn.configure(state=state)
            if continue_free_btn is not None:
                continue_free_btn.configure(state=state)
            cancel_btn.configure(state=state)

        def close_with_session(session: AuthSession | None) -> None:
            result["session"] = session
            if root.winfo_exists():
                root.destroy()

        def resolve_free_email() -> str:
            maybe_email = email_var.get().strip().lower()
            if maybe_email:
                return maybe_email
            return "free@local.ezframes"

        def on_login() -> None:
            if in_flight["value"]:
                return

            email = email_var.get().strip().lower()
            password = password_var.get().strip()
            if not email or not password:
                set_status("Email and password are required.", colors["danger"])
                return

            in_flight["value"] = True
            set_controls(False)
            set_status("Checking credentials...", colors["accent"])
            root.update_idletasks()

            try:
                if not self.client.sign_in(email, password):
                    set_status("Login failed. Check your credentials and try again.", colors["danger"])
                    return

                status = self.client.check_subscription_status(email, password)
                self._cache_status(email, status)
                if status != "ACTIVE":
                    set_status("Subscription inactive. Please renew your plan.", colors["danger"])
                    return

                session_payload = json.dumps({"email": email, "status": status})
                self.creds.write("session", email, session_payload)
                close_with_session(AuthSession(email=email, subscription_status="ACTIVE", source="aws"))
            except requests.RequestException as exc:
                log.exception("Auth request failed: %s", exc)
                set_status(f"Network error: {exc}", colors["danger"])
            except Exception as exc:
                log.exception("Unexpected auth error: %s", exc)
                set_status(f"Authentication failed: {exc}", colors["danger"])
            finally:
                in_flight["value"] = False
                if root.winfo_exists():
                    set_controls(True)

        def on_continue_free() -> None:
            close_with_session(AuthSession(email=resolve_free_email(), subscription_status="FREE", source="free"))

        def on_cancel() -> None:
            close_with_session(None)

        def make_button(parent, text: str, command, variant: str = "secondary") -> tk.Button:
            bg = colors["secondary"]
            if variant == "primary":
                bg = colors["primary"]
            return tk.Button(
                parent,
                text=text,
                command=command,
                bg=bg,
                fg=colors["text"],
                activebackground=colors["secondary_hover"] if variant == "secondary" else colors["primary_hover"],
                activeforeground=colors["text"],
                disabledforeground=colors["muted"],
                relief=tk.FLAT,
                bd=0,
                highlightthickness=1,
                highlightbackground=colors["panel_alt"],
                highlightcolor=colors["accent"],
                cursor="hand2",
                padx=12,
                pady=6,
                font=(body_font, 10, "bold"),
            )

        cancel_btn = make_button(button_row, "Cancel", on_cancel, variant="secondary")
        cancel_btn.pack(side="right")

        continue_free_btn = None
        if self.cfg.allow_free_tier:
            continue_free_btn = make_button(button_row, "Continue Free", on_continue_free, variant="secondary")
            continue_free_btn.pack(side="right", padx=(0, 8))
        login_btn = make_button(button_row, "Sign In", on_login, variant="primary")
        login_btn.pack(side="right", padx=(0, 8))

        root.bind("<Return>", lambda _e: on_login())
        root.bind("<Escape>", lambda _e: on_cancel())

        if email_var.get().strip():
            password_entry.focus_set()
        else:
            email_entry.focus_set()

        root.mainloop()
        return result["session"]

    def issue_launch_ticket(self, session: AuthSession) -> Path:
        return issue_launch_ticket(
            self.paths.state_dir,
            email=session.email,
            status=session.subscription_status,
            ttl_seconds=self.cfg.launch_ticket_ttl_seconds,
        )
