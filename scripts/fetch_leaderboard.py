"""Fetch a complete, paginated leaderboard snapshot for the 2026 challenge."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


COMPETITION_ID = "bb3693e1-26bc-4a9e-8619-4fe78b4eab0c"
URL = f"https://datacomp.opensky-network.org/api/competitions/{COMPETITION_ID}/leaderboard"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "external" / "leaderboard_latest.json",
    )
    args = parser.parse_args()

    items: list[dict] = []
    cursor: str | None = None
    while True:
        response = requests.get(URL, params={"cursor": cursor} if cursor else None, timeout=30)
        response.raise_for_status()
        page = response.json()
        items.extend(page.get("items", []))
        cursor = page.get("nextCursor")
        if not cursor:
            break

    payload = {
        "competitionId": COMPETITION_ID,
        "retrievedAt": datetime.now(timezone.utc).isoformat(),
        "source": URL,
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(items)} submissions to {args.output}")


if __name__ == "__main__":
    main()
