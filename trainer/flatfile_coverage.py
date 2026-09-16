from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer.providers.massive_flatfiles import MassiveFlatFileStore
from trainer.trading_calendar import generate_trading_dates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report local Massive flat-file coverage without downloading."
    )
    parser.add_argument("--start", required=True, help="First trading date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Last trading date (YYYY-MM-DD).")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/flatfiles"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = MassiveFlatFileStore(object(), cache_root=args.cache_root)
    report = {
        "start_date": args.start,
        "end_date": args.end,
        **store.coverage_report(generate_trading_dates(args.start, args.end)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
