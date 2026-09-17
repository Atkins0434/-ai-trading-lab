from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from trainer.providers.base import ProviderError
from trainer.providers.tiingo import TiingoClient, parse_tiingo_timestamp


MARKET_TIMEZONE = ZoneInfo("America/New_York")


def intraday_window(client, start_time: time, end_time: time):
    rows = client.get_intraday_prices(
        "SPY", "2018-01-02", "2018-01-03"
    )
    selected = []
    for row in rows:
        stamp = parse_tiingo_timestamp(row)
        observed = datetime.fromisoformat(
            stamp.replace("Z", "+00:00")
        ).astimezone(MARKET_TIMEZONE)
        if (
            observed.date().isoformat() == "2018-01-02"
            and start_time <= observed.time() <= end_time
        ):
            selected.append(row)
    return selected


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
        "freeze_timestamp": "2018-01-02T09:15:00-05:00",
        "capabilities": {
            "daily": probe("daily", lambda: client.get_daily_prices(
                "SPY", "2018-01-02", "2018-01-02"
            )),
            "premarket_intraday": probe(
                "premarket_intraday",
                lambda: intraday_window(
                    client, time(4, 0), time(9, 15)
                ),
            ),
            "regular_session_intraday": probe(
                "regular_session_intraday",
                lambda: intraday_window(
                    client, time(9, 30), time(16, 0)
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
