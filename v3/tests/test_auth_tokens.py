from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ezframes.common.auth_tokens import (
    issue_launch_ticket,
    sign_cache_record,
    validate_launch_ticket,
    verify_cache_record,
)


class AuthTokenTests(unittest.TestCase):
    def test_cache_signature_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            sig = sign_cache_record(state_dir, "user@example.com", "2026-02-22", "ACTIVE")
            self.assertTrue(verify_cache_record(state_dir, "user@example.com", "2026-02-22", "ACTIVE", sig))
            self.assertFalse(verify_cache_record(state_dir, "user@example.com", "2026-02-22", "INACTIVE", sig))

    def test_launch_ticket_issue_and_consume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            ticket = issue_launch_ticket(state_dir, email="user@example.com", status="ACTIVE", ttl_seconds=120)
            self.assertTrue(ticket.exists())
            result = validate_launch_ticket(state_dir, ticket, consume=True)
            self.assertTrue(result.valid)
            self.assertEqual(result.email, "user@example.com")
            self.assertFalse(ticket.exists())

    def test_launch_ticket_tamper_detected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            ticket = issue_launch_ticket(state_dir, email="user@example.com", status="ACTIVE", ttl_seconds=120)
            payload = json.loads(ticket.read_text(encoding="utf-8"))
            payload["email"] = "attacker@example.com"
            ticket.write_text(json.dumps(payload), encoding="utf-8")

            result = validate_launch_ticket(state_dir, ticket, consume=False)
            self.assertFalse(result.valid)
            self.assertIn("signature", (result.reason or "").lower())


if __name__ == "__main__":
    unittest.main()
