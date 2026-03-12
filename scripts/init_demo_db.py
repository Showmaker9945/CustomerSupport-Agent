"""Initialize business tables and load curated demo data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.repositories import seed_demo_data
from src.db.session import reset_database_connection


def main() -> None:
    reset_database_connection()
    stats = seed_demo_data(clear_existing=True)
    print("Demo business data initialized.")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
