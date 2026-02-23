from __future__ import annotations

import os
from dataclasses import dataclass


FREE = "free"
PRO = "pro"


@dataclass(frozen=True)
class Entitlement:
    tier: str

    @property
    def is_pro(self) -> bool:
        return self.tier == PRO

    @property
    def label(self) -> str:
        return "Pro" if self.is_pro else "Free"


def normalize_tier(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"pro", "paid", "premium", "active"}:
        return PRO
    return FREE


def resolve_entitlement_from_env(default_tier: str = FREE) -> Entitlement:
    raw = os.environ.get("EZFRAMES_ENTITLEMENT", "").strip()
    if not raw:
        raw = default_tier
    return Entitlement(tier=normalize_tier(raw))
