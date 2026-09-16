from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer.multi_day_trainer import run_multi_day_trainer
from trainer.providers.massive import MassiveClient
from trainer.rate_control import AdaptiveRateLimiter, RetryPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable multi-day Research Scout Trainer replays.")
    parser.add_argument("--dates-csv", default="", help="Comma-separated historical trading dates.")
    parser.add_argument("--start-date", help="First date for calendar-generated sessions.")
    parser.add_argument("--end-date", help="Last date for calendar-generated sessions.")
    parser.add_argument(
        "--universe-mode",
        required=True,
        choices=["ci_fixture", "historical_research"],
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--threshold-pct", type=float)
    parser.add_argument("--exploration-top-k", type=int, default=0)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/trainer"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=None,
        help=(
            "Optional runtime override; omit to use config/massive_plan.json."
        ),
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=["DEVELOPMENT", "VALIDATION", "HOLDOUT"],
        default=["DEVELOPMENT", "VALIDATION"],
    )
    parser.add_argument("--unlock-holdout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            raise SystemExit("--start-date and --end-date must be supplied together.")
        from trainer.trading_calendar import generate_trading_dates

        dates = generate_trading_dates(args.start_date, args.end_date)
    else:
        dates = [value.strip() for value in args.dates_csv.split(",") if value.strip()]
    if not dates:
        raise SystemExit("Provide --dates-csv or both --start-date and --end-date.")
    limiter = (
        AdaptiveRateLimiter()
        if args.requests_per_minute is None
        else AdaptiveRateLimiter(args.requests_per_minute)
    )
    client = MassiveClient.from_environment(
        rate_limiter=limiter,
        retry_policy=RetryPolicy(),
    )
    result = run_multi_day_trainer(
        client,
        args.tickers,
        dates,
        cache_root=args.cache,
        output_root=args.output_root,
        threshold_pct=args.threshold_pct,
        exploration_top_k=args.exploration_top_k,
        resume=not args.no_resume,
        max_workers=args.max_workers,
        allowed_partitions=args.partitions,
        holdout_unlocked=args.unlock_holdout,
        universe_mode=args.universe_mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
