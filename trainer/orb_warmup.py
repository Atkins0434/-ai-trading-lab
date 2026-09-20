"""Warm the ORB opening-volume cache without retaining minute flat files."""
from __future__ import annotations

import argparse
from pathlib import Path

from trainer.orb import (
    load_orb_config,
    opening_cache_path,
    write_opening_volume_cache,
)
from trainer.providers.massive_flatfiles import (
    MINUTE_AGGS_DATASET,
    MassiveFlatFileStore,
)
from trainer.universe_builder import previous_trading_sessions


def warm_opening_volume_cache(
    end_date: str,
    *,
    flatfiles: MassiveFlatFileStore,
    cache_root: Path,
    sessions: int = 14,
    orb_version: str = "orb_v1.0",
) -> list[Path]:
    written: list[Path] = []
    for trading_date in previous_trading_sessions(end_date, sessions):
        destination = opening_cache_path(cache_root, trading_date)
        minute_path = flatfiles.cache_path(MINUTE_AGGS_DATASET, trading_date)
        metadata_path = flatfiles.metadata_path(minute_path)
        if not destination.is_file():
            flatfiles.ensure_day(MINUTE_AGGS_DATASET, trading_date)
            written.append(write_opening_volume_cache(
                flatfiles,
                trading_date,
                cache_root=cache_root,
                orb_version=orb_version,
            ))
        minute_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        print(
            f"[orb_warmup] trading_date={trading_date} "
            f"cache={destination} minute_file_deleted=true",
            flush=True,
        )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the prior-session ORB opening-volume cache."
    )
    parser.add_argument("--end", required=True)
    parser.add_argument("--cache-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_orb_config()
    warm_opening_volume_cache(
        args.end,
        flatfiles=MassiveFlatFileStore.from_environment(),
        cache_root=args.cache_root or Path(config["opening_volume_cache_root"]),
        sessions=int(config["opening_volume_lookback_sessions"]),
        orb_version=config["orb_version"],
    )


if __name__ == "__main__":
    main()
