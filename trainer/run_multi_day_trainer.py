from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer.multi_day_trainer import run_multi_day_trainer
from trainer.providers.massive import MassiveClient
from trainer.run_research_alpha_batch import DEFAULT_TICKERS
from trainer.universe_collector import RequestRateLimiter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable multi-day Research Scout Trainer replays.")
    parser.add_argument("--dates-csv", required=True, help="Comma-separated historical trading dates.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--threshold-pct", type=float)
    parser.add_argument("--exploration-top-k", type=int, default=0)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/trainer"))
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dates = [value.strip() for value in args.dates_csv.split(",") if value.strip()]
    limiter = RequestRateLimiter()
    client = MassiveClient.from_environment(before_request=limiter.wait)
    result = run_multi_day_trainer(
        client,
        args.tickers,
        dates,
        cache_root=args.cache,
        output_root=args.output_root,
        threshold_pct=args.threshold_pct,
        exploration_top_k=args.exploration_top_k,
        resume=not args.no_resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
