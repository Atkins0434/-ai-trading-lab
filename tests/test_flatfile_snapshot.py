from __future__ import annotations

from datetime import date, datetime, time, timezone
import gzip
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from trainer.flatfile_snapshot import build_flatfile_snapshot
from trainer.providers.massive_flatfiles import (
    DAY_AGGS_DATASET,
    MINUTE_AGGS_DATASET,
    SOURCE,
    MassiveFlatFileStore,
)
from trainer.rate_control import load_massive_plan
from trainer.universe_builder import previous_trading_sessions


ET = ZoneInfo("America/New_York")
HEADER = "ticker,volume,open,close,high,low,window_start,transactions"


def _nanos(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _row(ticker: str, value: datetime, *, close: float, volume: int) -> str:
    return (
        f"{ticker},{volume},{close},{close},{close},{close},"
        f"{_nanos(value)},1"
    )


def _write_cached_csv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(("\n".join([HEADER, *rows]) + "\n").encode()))


def _manifest(trading_date: str) -> dict:
    cutoff = datetime.combine(
        date.fromisoformat(trading_date), time(0), tzinfo=ET
    ).isoformat()
    return {
        "replay_id": f"{trading_date}-flatfile-test",
        "trading_date": trading_date,
        "coverage_status": "complete",
        "ruleset": {"version": "universe_v1.0"},
        "massive_plan": load_massive_plan(),
        "universe_mode": "historical_research",
        "manifest_hash": "sha256:" + "a" * 64,
        "research_evidence": True,
        "promotion_eligible": True,
        "securities": [{
            "ticker": "AAL",
            "inclusion": True,
            "listing_venue": "XNAS",
            "prior_close": 10.0,
            "reference_data_as_of_timestamp": cutoff,
            "market_cap_usd": None,
        }],
    }


@pytest.mark.parametrize(
    ("trading_date", "premarket_utc", "expected_timestamp"),
    [
        (
            "2024-03-15",
            datetime(2024, 3, 15, 10, 30, tzinfo=timezone.utc),
            "2024-03-15T06:30:00-04:00",
        ),
        (
            "2024-01-16",
            datetime(2024, 1, 16, 11, 30, tzinfo=timezone.utc),
            "2024-01-16T06:30:00-05:00",
        ),
    ],
)
def test_nanosecond_flatfile_premarket_bar_respects_dst(
    tmp_path: Path,
    trading_date: str,
    premarket_utc: datetime,
    expected_timestamp: str,
):
    store = MassiveFlatFileStore(object(), cache_root=tmp_path)
    previous_date = previous_trading_sessions(trading_date, 1)[0]
    previous_midnight = datetime.combine(
        date.fromisoformat(previous_date), time(0), tzinfo=timezone.utc
    )
    _write_cached_csv(
        store.cache_path(DAY_AGGS_DATASET, previous_date),
        [_row("AAL", previous_midnight, close=10.0, volume=1_000_000)],
    )
    _write_cached_csv(
        store.cache_path(MINUTE_AGGS_DATASET, previous_date), []
    )
    regular_utc = datetime.combine(
        date.fromisoformat(trading_date), time(9, 45), tzinfo=ET
    ).astimezone(timezone.utc)
    _write_cached_csv(
        store.cache_path(MINUTE_AGGS_DATASET, trading_date),
        [
            _row("AAL", premarket_utc, close=10.1, volume=100),
            _row("AAL", regular_utc, close=10.2, volume=200),
        ],
    )

    result = build_flatfile_snapshot(
        trading_date,
        _manifest(trading_date),
        store,
        lookback_sessions=1,
    )

    bars = result.snapshot["securities"][0]["premarket_bars"]
    observed = next(bar for bar in bars if bar["timestamp"] == expected_timestamp)
    assert observed["source"] == SOURCE
    assert observed["session"] == "PREMARKET"
    assert observed["volume"] == 100
    assert result.padded_bar_statistics["by_ticker"]["AAL"]["premarket"] == 59
    assert result.padded_bar_statistics["premarket_padded_bar_count"] == 59
    assert result.padded_bar_statistics["premarket_bar_count"] == 60
    assert result.padded_bar_statistics["premarket_padding_share"] == pytest.approx(59 / 60)

