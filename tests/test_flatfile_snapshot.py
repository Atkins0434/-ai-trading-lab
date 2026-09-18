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


def _row(
    ticker: str,
    value: datetime,
    *,
    close: float,
    volume: int,
    transactions: int = 1,
) -> str:
    return (
        f"{ticker},{volume},{close},{close},{close},{close},"
        f"{_nanos(value)},{transactions}"
    )


def _ohlcv_row(
    ticker: str,
    value: datetime,
    *,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: int = 1_000_000,
) -> str:
    return (
        f"{ticker},{volume},{open_price},{close},{high},{low},"
        f"{_nanos(value)},1"
    )


def _write_cached_csv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(("\n".join([HEADER, *rows]) + "\n").encode()))


def _write_identity_csv(path: Path, rows: list[str]) -> None:
    header = HEADER + ",stable_security_id"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(("\n".join([header, *rows]) + "\n").encode()))


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
    ("trading_date", "last_admitted_utc", "freeze_utc", "expected_timestamp"),
    [
        (
            "2024-03-15",
            datetime(2024, 3, 15, 13, 14, tzinfo=timezone.utc),
            datetime(2024, 3, 15, 13, 15, tzinfo=timezone.utc),
            "2024-03-15T09:14:00-04:00",
        ),
        (
            "2024-01-16",
            datetime(2024, 1, 16, 14, 14, tzinfo=timezone.utc),
            datetime(2024, 1, 16, 14, 15, tzinfo=timezone.utc),
            "2024-01-16T09:14:00-05:00",
        ),
    ],
)
def test_nanosecond_flatfile_premarket_bar_respects_dst(
    tmp_path: Path,
    trading_date: str,
    last_admitted_utc: datetime,
    freeze_utc: datetime,
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
            _row("AAL", last_admitted_utc, close=10.1, volume=100),
            _row("AAL", freeze_utc, close=10.15, volume=150),
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
    assert len(bars) == 1
    assert all("09:15:00" not in bar["timestamp"] for bar in bars)
    assert result.bar_statistics["by_ticker"]["AAL"]["real_premarket"] == 1
    assert result.bar_statistics["by_ticker"]["AAL"]["real_premarket_60m"] == 1
    assert result.snapshot["securities"][0]["market_data"][
        "average_daily_volume"
    ]["value"] == 1_000_000


def test_regular_bars_are_sorted_and_duplicate_minutes_are_audited(
    tmp_path: Path,
):
    trading_date = "2024-03-15"
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
    at_1000 = datetime(2024, 3, 15, 10, 0, tzinfo=ET)
    at_1005 = datetime(2024, 3, 15, 10, 5, tzinfo=ET)
    _write_cached_csv(
        store.cache_path(MINUTE_AGGS_DATASET, trading_date),
        [
            _row("AAL", at_1005, close=10.5, volume=500),
            _row(
                "AAL", at_1000, close=10.1, volume=300, transactions=2
            ),
            _row(
                "AAL", at_1000, close=10.2, volume=200, transactions=5
            ),
        ],
    )

    result = build_flatfile_snapshot(
        trading_date,
        _manifest(trading_date),
        store,
        lookback_sessions=1,
    )

    path = result.outcome_bars["AAL"]
    assert [bar["timestamp"] for bar in path] == [
        "2024-03-15T10:00:00-04:00",
        "2024-03-15T10:05:00-04:00",
    ]
    assert path[0]["close"] == 10.2
    assert path[0]["volume"] == 200
    assert result.bar_statistics["by_ticker"]["AAL"]["real_regular"] == 2
    assert result.bar_statistics["duplicate_minute_rows"] == [
        {
            "ticker": "AAL",
            "timestamp": "2024-03-15T10:00:00-04:00",
            "kept_row": {
                "open": 10.2,
                "high": 10.2,
                "low": 10.2,
                "close": 10.2,
                "volume": 200.0,
                "transactions": 5,
            },
            "discarded_row": {
                "open": 10.1,
                "high": 10.1,
                "low": 10.1,
                "close": 10.1,
                "volume": 300.0,
                "transactions": 2,
            },
        }
    ]


def test_atr_14_uses_true_range_and_prior_close_gap(tmp_path: Path):
    trading_date = "2024-03-15"
    store = MassiveFlatFileStore(object(), cache_root=tmp_path)
    lookback_dates = previous_trading_sessions(trading_date, 20)
    active_dates = lookback_dates[-15:]

    for session_date in lookback_dates:
        session_midnight = datetime.combine(
            date.fromisoformat(session_date), time(0), tzinfo=timezone.utc
        )
        if session_date == active_dates[0]:
            rows = [_ohlcv_row(
                "AAL", session_midnight, open_price=100, high=101,
                low=99, close=100,
            )]
        elif session_date == active_dates[-1]:
            # The $10 prior-close gap dominates the $2 high-low range.
            rows = [_ohlcv_row(
                "AAL", session_midnight, open_price=109, high=110,
                low=108, close=109,
            )]
        elif session_date in active_dates:
            rows = [_ohlcv_row(
                "AAL", session_midnight, open_price=100, high=101,
                low=99, close=100,
            )]
        else:
            rows = []
        _write_cached_csv(
            store.cache_path(DAY_AGGS_DATASET, session_date), rows
        )
        _write_cached_csv(
            store.cache_path(MINUTE_AGGS_DATASET, session_date), []
        )
    _write_cached_csv(
        store.cache_path(MINUTE_AGGS_DATASET, trading_date), []
    )

    result = build_flatfile_snapshot(
        trading_date,
        _manifest(trading_date),
        store,
        lookback_sessions=20,
    )

    market_data = result.snapshot["securities"][0]["market_data"]
    assert market_data["atr_14_usd"]["value"] == pytest.approx(
        (13 * 2 + 10) / 14
    )
    assert market_data["atr_lookback_sessions"]["value"] == 14
    assert market_data["atr_sessions_used"]["value"] == 14
    assert "missing_reason" not in market_data["atr_14_usd"]


def test_atr_is_missing_with_reason_when_history_is_short(tmp_path: Path):
    trading_date = "2024-03-15"
    store = MassiveFlatFileStore(object(), cache_root=tmp_path)
    previous_date = previous_trading_sessions(trading_date, 1)[0]
    midnight = datetime.combine(
        date.fromisoformat(previous_date), time(0), tzinfo=timezone.utc
    )
    _write_cached_csv(
        store.cache_path(DAY_AGGS_DATASET, previous_date),
        [_row("AAL", midnight, close=10.0, volume=1_000_000)],
    )
    _write_cached_csv(store.cache_path(MINUTE_AGGS_DATASET, previous_date), [])
    _write_cached_csv(store.cache_path(MINUTE_AGGS_DATASET, trading_date), [])

    result = build_flatfile_snapshot(
        trading_date, _manifest(trading_date), store, lookback_sessions=1
    )

    atr = result.snapshot["securities"][0]["market_data"]["atr_14_usd"]
    assert atr["value"] is None
    assert atr["missing_reason"] == "INSUFFICIENT_ATR_HISTORY"


def test_security_id_join_separates_shared_ticker_and_quarantines_discontinuity(
    tmp_path: Path,
):
    trading_date = "2024-03-15"
    previous_date = previous_trading_sessions(trading_date, 1)[0]
    store = MassiveFlatFileStore(object(), cache_root=tmp_path)
    midnight = datetime.combine(
        date.fromisoformat(previous_date), time(0), tzinfo=timezone.utc
    )
    regular = datetime(2024, 3, 15, 10, 0, tzinfo=ET)
    manifest = _manifest(trading_date)
    template = manifest["securities"][0]
    manifest["securities"] = [
        {**template, "stable_security_id": "SEC-A", "prior_close": 10.0},
        {**template, "stable_security_id": "SEC-B", "prior_close": 100.0},
    ]
    _write_identity_csv(
        store.cache_path(DAY_AGGS_DATASET, previous_date),
        [
            _row("AAL", midnight, close=10.0, volume=1000) + ",SEC-A",
            _row("AAL", midnight, close=100.0, volume=1000) + ",SEC-B",
        ],
    )
    _write_identity_csv(
        store.cache_path(MINUTE_AGGS_DATASET, previous_date), []
    )
    _write_identity_csv(
        store.cache_path(MINUTE_AGGS_DATASET, trading_date),
        [
            _row("AAL", regular, close=10.0, volume=100) + ",SEC-A",
            _row("AAL", regular, close=100.0, volume=100) + ",SEC-B",
        ],
    )

    result = build_flatfile_snapshot(
        trading_date, manifest, store, lookback_sessions=1
    )

    assert result.outcome_bars["SEC-A"][0]["close"] == 10.0
    assert result.outcome_bars["SEC-B"][0]["close"] == 100.0
    assert set(result.bar_statistics["by_stable_security_id"]) == {
        "SEC-A", "SEC-B"
    }

    # A provider identity that is not in the point-in-time universe is dropped.
    _write_identity_csv(
        store.cache_path(MINUTE_AGGS_DATASET, trading_date),
        [
            _row("AAL", regular, close=10.0, volume=100) + ",SEC-A",
            _row(
                "AAL", regular.replace(minute=1), close=20.0, volume=100
            ) + ",SEC-A",
            _row("AAL", regular, close=50.0, volume=100) + ",UNKNOWN",
        ],
    )
    result = build_flatfile_snapshot(
        trading_date, manifest, store, lookback_sessions=1
    )
    assert result.bar_statistics["unresolved_bar_row_count"] == 1
    assert result.excluded_tickers == [{
        "ticker": "AAL",
        "reason": "BAR_DISCONTINUITY",
        "previous_timestamp": "2024-03-15T10:00:00-04:00",
        "current_timestamp": "2024-03-15T10:01:00-04:00",
        "previous_close": 10.0,
        "current_close": 20.0,
        "close_change_pct": 100.0,
    }]
