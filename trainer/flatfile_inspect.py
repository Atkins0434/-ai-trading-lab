from __future__ import annotations

import argparse
import csv
from datetime import date, time
import gzip
import json
from pathlib import Path
from typing import Any

from trainer.flatfile_snapshot import load_flatfile_replay_config
from trainer.providers.massive_flatfiles import (
    MINUTE_AGGS_DATASET,
    SUPPORTED_DATASETS,
    MassiveFlatFileError,
    MassiveFlatFileStore,
    _parse_flatfile_timestamp,
    _row_value,
)


SESSION_BUCKETS = (
    ("04:00-07:00", time(4), time(7)),
    ("07:00-09:30", time(7), time(9, 30)),
    ("09:30-16:00", time(9, 30), time(16)),
    ("16:00-20:00", time(16), time(20)),
)


def inspect_cached_flatfile(
    dataset: str,
    trading_date: str,
    ticker: str,
    *,
    cache_root: Path,
) -> dict[str, Any]:
    """Inspect raw and ET-parsed timestamps for one ticker in a cached file."""
    if dataset not in SUPPORTED_DATASETS:
        raise MassiveFlatFileError(f"Unsupported Massive dataset: {dataset}")
    normalized_date = date.fromisoformat(trading_date).isoformat()
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise MassiveFlatFileError("ticker must be non-empty")

    store = MassiveFlatFileStore(object(), cache_root=cache_root)
    path = store.cache_path(dataset, normalized_date)
    if store.cache_status(dataset, normalized_date) != "PRESENT":
        raise MassiveFlatFileError(
            f"Cached flat file is not present and valid: {path}"
        )

    matches: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_ticker = (_row_value(row, "ticker", "symbol") or "").upper()
            if row_ticker != normalized_ticker:
                continue
            raw_timestamp = _row_value(row, "window_start", "timestamp", "t")
            if raw_timestamp is None:
                raise MassiveFlatFileError(
                    f"Cached row for {normalized_ticker} has no timestamp."
                )
            parsed = _parse_flatfile_timestamp(raw_timestamp)
            try:
                volume = float(_row_value(row, "volume", "v") or "nan")
            except ValueError as exc:
                raise MassiveFlatFileError(
                    f"Cached row for {normalized_ticker} has invalid volume."
                ) from exc
            matches.append({
                "raw_row": dict(row),
                "raw_timestamp": raw_timestamp,
                "parsed_et_timestamp": parsed.isoformat(),
                "parsed": parsed,
                "volume": volume,
            })

    matches.sort(key=lambda item: item["parsed"])
    buckets = {label: 0 for label, _, _ in SESSION_BUCKETS}
    for item in matches:
        observed = item["parsed"].time()
        for label, start, end in SESSION_BUCKETS:
            if start <= observed < end:
                buckets[label] += 1
                break

    def public_row(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "raw_row": item["raw_row"],
            "raw_timestamp": item["raw_timestamp"],
            "parsed_et_timestamp": item["parsed_et_timestamp"],
        }

    first = public_row(matches[0]) if matches else None
    last = public_row(matches[-1]) if matches else None
    positive_premarket = [
        public_row(item)
        for item in matches
        if item["parsed"].time() < time(9, 30) and item["volume"] > 0
    ][:3]
    return {
        "dataset": dataset,
        "trading_date": normalized_date,
        "ticker": normalized_ticker,
        "cached_path": str(path),
        "total_rows": len(matches),
        "first_row": first,
        "last_row": last,
        "session_row_counts": buckets,
        "first_three_positive_volume_before_09_30_et": positive_premarket,
    }


def parse_args() -> argparse.Namespace:
    config = load_flatfile_replay_config()
    parser = argparse.ArgumentParser(
        description="Inspect one ticker in a cached Massive flat file."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--date", required=True, dest="trading_date")
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--cache-root", type=Path, default=Path(config["cache_root"])
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_cached_flatfile(
        args.dataset,
        args.trading_date,
        args.ticker,
        cache_root=args.cache_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
