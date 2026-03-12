"""Helpers for loading demo seed data from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_SEED_DIR = PROJECT_ROOT / "data" / "demo_seed"


def _load_json(filename: str) -> Any:
    path = DEMO_SEED_DIR / filename
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_seed_accounts() -> Dict[str, Dict[str, Any]]:
    return _load_json("accounts.json")


def load_seed_subscriptions() -> Dict[str, Dict[str, Any]]:
    return _load_json("subscriptions.json")


def load_seed_invoices() -> Dict[str, List[Dict[str, Any]]]:
    return _load_json("invoices.json")


def load_seed_tickets() -> List[Dict[str, Any]]:
    return _load_json("tickets.json")


def load_seed_bundle() -> Dict[str, Any]:
    return {
        "accounts": load_seed_accounts(),
        "subscriptions": load_seed_subscriptions(),
        "invoices": load_seed_invoices(),
        "tickets": load_seed_tickets(),
    }

