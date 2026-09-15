from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from trainer.historical_cache import HistoricalCache
from trainer.providers.tiingo import TiingoClient


MARKET_TIMEZONE = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and checksum Tiingo history without logging secrets."
    )
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--date", default="2018-01-02")
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def cache_tiingo_history(
    ticker: str,
    trading_date: str,
    cache_root: Path,
    *,
    force_refresh: bool = False,
) -> dict:
    target_date = date.fromisoformat(trading_date)
    daily_start = (target_date - timedelta(days=45)).isoformat()
    regular_start = datetime.combine(
        target_date,
        time(9, 30),
        MARKET_TIMEZONE,
    ).isoformat()
    regular_end = datetime.combine(
        target_date,
        time(16, 0),
        MARKET_TIMEZONE,
    ).isoformat()

    provider = TiingoClient.from_environment()
    cache = HistoricalCache(cache_root)
    daily = cache.get_daily_prices(
        provider,
        ticker,
        daily_start,
        trading_date,
        force_refresh=force_refresh,
    )
    regular = cache.get_intraday_prices(
        provider,
        ticker,
        regular_start,
        regular_end,
        purpose="OUTCOME_GRADING",
        force_refresh=force_refresh,
    )
    return {"daily": daily["manifest"], "regular": regular["manifest"]}


def main() -> None:
    load_dotenv()
    args = parse_args()
    manifests = cache_tiingo_history(
        ticker=args.ticker,
        trading_date=args.date,
        cache_root=args.cache_root,
        force_refresh=args.force_refresh,
    )
    for name, manifest in manifests.items():
        print(
            f"{name}: {manifest['record_count']} records; "
            f"sha256={manifest['content_sha256']}"
        )


if __name__ == "__main__":
    main()
