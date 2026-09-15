from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer.providers.massive import MassiveClient
from trainer.universe_collector import RequestRateLimiter, collect_ticker_overviews, discover_common_stocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume the point-in-time Alpha universe collection.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/massive/research_universe.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limiter = RequestRateLimiter()
    client = MassiveClient.from_environment(before_request=limiter.wait)
    tickers = [ticker.upper() for ticker in (args.tickers or [])]
    if args.discover:
        tickers = discover_common_stocks(client, args.date)
    if not tickers:
        raise SystemExit("Provide --tickers or --discover.")
    manifest = collect_ticker_overviews(
        client, tickers, args.date, args.checkpoint, max_new=args.max_new
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
