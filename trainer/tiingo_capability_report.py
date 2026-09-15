from __future__ import annotations

import json
from pathlib import Path

from trainer.providers.base import ProviderError
from trainer.providers.tiingo import TiingoClient, parse_tiingo_timestamp


def probe(name, operation):
    try:
        rows = operation()
        timestamps = [
            parse_tiingo_timestamp(row) for row in rows
            if row.get("date") or row.get("publishedDate")
        ]
        return {
            "status": "AVAILABLE",
            "record_count": len(rows),
            "earliest_timestamp": min(timestamps, default=None),
            "latest_timestamp": max(timestamps, default=None),
            "fields": sorted(rows[0]) if rows else [],
        }
    except ProviderError as exc:
        return {"status": "UNAVAILABLE", "error": str(exc)}


def build_report(client: TiingoClient) -> dict:
    return {
        "provider": client.provider_name,
        "test_date": "2018-01-02",
        "freeze_timestamp": "2018-01-02T07:00:00-05:00",
        "capabilities": {
            "daily": probe("daily", lambda: client.get_daily_prices(
                "SPY", "2018-01-02", "2018-01-02"
            )),
            "premarket_intraday": probe(
                "premarket_intraday",
                lambda: client.get_intraday_prices(
                    "SPY",
                    "2018-01-02T04:00:00-05:00",
                    "2018-01-02T07:00:00-05:00",
                ),
            ),
            "regular_session_intraday": probe(
                "regular_session_intraday",
                lambda: client.get_intraday_prices(
                    "SPY",
                    "2018-01-02T09:30:00-05:00",
                    "2018-01-02T16:00:00-05:00",
                ),
            ),
            "news": probe("news", lambda: client.get_news(
                ["SPY"], "2018-01-02", "2018-01-02"
            )),
        },
    }


def main() -> None:
    report = build_report(TiingoClient.from_environment())
    output = Path("reports/tiingo/capability_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["capabilities"]["daily"]["status"] != "AVAILABLE":
        raise SystemExit("Tiingo authentication/daily-price check failed.")


if __name__ == "__main__":
    main()
