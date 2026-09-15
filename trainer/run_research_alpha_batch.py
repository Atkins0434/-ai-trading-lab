from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer.providers.massive import MassiveClient
from trainer.research_alpha_batch import run_massive_alpha_batch
from trainer.universe_collector import RequestRateLimiter


DEFAULT_TICKERS = ["AA", "AAL", "BBAI", "CHWY", "CLOV", "ETSY", "FUBO", "JOBY", "LUNR", "UPST"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one bounded Massive Free Research Scout Alpha batch."
    )
    parser.add_argument("--date", default="2026-09-14")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--threshold-pct", type=float)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("reports") / args.date
    limiter = RequestRateLimiter()
    client = MassiveClient.from_environment(before_request=limiter.wait)
    manifest = run_massive_alpha_batch(
        client,
        args.tickers,
        args.date,
        cache_root=args.cache,
        output_dir=output_dir,
        threshold_pct=args.threshold_pct,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
