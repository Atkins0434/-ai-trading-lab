from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path

from trainer.historical_cache import HistoricalCache
from trainer.massive_alpha_snapshot import build_massive_alpha_snapshot
from trainer.providers.massive import MassiveClient
from trainer.research_scout_alpha import run_research_scout_alpha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the historical Research Scout Alpha probe.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--date", default="2026-09-14")
    parser.add_argument("--cache", type=Path, default=Path(".cache/historical"))
    parser.add_argument("--report", type=Path, default=Path("reports/massive/research_alpha.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = date.fromisoformat(args.date)
    start = (target - timedelta(days=45)).isoformat()
    client = MassiveClient.from_environment()
    cache = HistoricalCache(args.cache)
    daily = cache.get_daily_prices(client, args.ticker, start, args.date)["records"]
    intraday = cache.get_intraday_prices(
        client, args.ticker, start, args.date, purpose="ALPHA_SELECTION_BASELINE"
    )["records"]
    snapshot = build_massive_alpha_snapshot(
        args.ticker,
        args.date,
        daily,
        intraday,
        exchange="ARCA" if args.ticker.upper() == "SPY" else "UNKNOWN",
        eligible=False,
        eligibility_reasons=["DATA_PIPELINE_VALIDATION_SYMBOL"],
    )
    result = run_research_scout_alpha(snapshot)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
