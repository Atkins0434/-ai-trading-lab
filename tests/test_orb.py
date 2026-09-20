from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from trainer.execution_costs import load_execution_costs
from trainer.orb import (
    _sized_results,
    build_orb_research,
    opening_cache_path,
    opening_volume_baseline,
    simulate_orb_signal,
)
from trainer.providers.massive_flatfiles import DAY_AGGS_DATASET, MINUTE_AGGS_DATASET
from trainer.universe_builder import previous_trading_sessions


ET = ZoneInfo("America/New_York")


def _bar(at: datetime, *, open=10.0, high=10.2, low=9.8, close=10.0):
    return {
        "timestamp": at.isoformat(),
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000.0,
        "source": "MASSIVE_FLATFILE",
        "session": "REGULAR",
    }


@pytest.mark.parametrize(
    ("available", "expected", "sessions_used"),
    ((14, 107.5, 14), (7, 104.0, 7), (3, None, 3)),
)
def test_opening_volume_baseline_uses_14_7_and_rejects_3_sessions(
    tmp_path: Path,
    available: int,
    expected: float | None,
    sessions_used: int,
):
    target = "2024-03-15"
    sessions = previous_trading_sessions(target, 14)
    for index, trading_date in enumerate(sessions[-available:], start=1):
        path = opening_cache_path(tmp_path, trading_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "tickers": {"AAL": {"opening_volume": 100 + index}},
        }))
    baseline, used = opening_volume_baseline(
        "AAL", target, cache_root=tmp_path,
        lookback_sessions=14, minimum_sessions=7,
    )
    assert used == sessions_used
    if expected is None:
        assert baseline is None
    else:
        assert baseline == pytest.approx(expected)


def test_buy_stop_triggers_0941_and_atr_stop_hits_1012_exactly():
    start = datetime(2024, 3, 15, 9, 35, tzinfo=ET)
    bars = [_bar(start + timedelta(minutes=index), high=10.49, low=10.2) for index in range(38)]
    bars[6] = _bar(start + timedelta(minutes=6), open=10.4, high=10.6, low=10.45, close=10.55)
    bars[37] = _bar(start + timedelta(minutes=37), open=10.2, high=10.3, low=10.0, close=10.1)
    result = simulate_orb_signal(
        bars,
        direction="LONG",
        first_candle_high=10.5,
        first_candle_low=9.9,
        atr_14_usd=5.0,
    )
    assert result["entry_timestamp"] == "2024-03-15T09:41:00-04:00"
    assert result["entry_price"] == 10.5
    assert result["exit_timestamp"] == "2024-03-15T10:12:00-04:00"
    assert result["exit_price"] == 10.0
    assert result["exit_reason"] == "STOP"
    assert result["r_multiple"] == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open():
    bars = [
        _bar(datetime(2024, 3, 15, 9, 41, tzinfo=ET), open=10.6, high=10.8, low=10.55),
        _bar(datetime(2024, 3, 15, 9, 42, tzinfo=ET), open=9.7, high=9.9, low=9.6),
    ]
    result = simulate_orb_signal(
        bars, direction="LONG", first_candle_high=10.5,
        first_candle_low=10.0, atr_14_usd=5.0,
    )
    assert result["stop_price"] == pytest.approx(10.1)
    assert result["exit_price"] == 9.7
    assert result["exit_reason"] == "STOP"


def test_not_triggered_and_no_entry_after_1530():
    quiet = [_bar(datetime(2024, 3, 15, 10, 0, tzinfo=ET), high=10.49)]
    assert simulate_orb_signal(
        quiet, direction="LONG", first_candle_high=10.5,
        first_candle_low=10.0, atr_14_usd=1.0,
    )["exit_reason"] == "NOT_TRIGGERED"
    late = [_bar(datetime(2024, 3, 15, 15, 31, tzinfo=ET), high=11.0)]
    assert simulate_orb_signal(
        late, direction="LONG", first_candle_high=10.5,
        first_candle_low=10.0, atr_14_usd=1.0,
    )["exit_reason"] == "NOT_TRIGGERED"


def _sizing_row(index: int, *, direction: str = "LONG"):
    entry = datetime(2024, 3, 15, 9, 40, tzinfo=ET)
    return {
        "stable_security_id": f"ID{index}",
        "selected": True,
        "selection_status": "SELECTED",
        "direction": direction,
        "risk_per_trade_pct": 1.0,
        "signal": {
            "trade_executed": True,
            "status": "TRIGGERED",
            "direction": direction,
            "entry_timestamp": entry.isoformat(),
            "entry_price": 10.0,
            "exit_timestamp": (entry + timedelta(hours=1)).isoformat(),
            "exit_price": 10.25 if direction == "LONG" else 9.75,
            "exit_reason": "SESSION_END",
            "stop_price": 9.75 if direction == "LONG" else 10.25,
            "stop_distance_usd": 0.25,
            "r_multiple": 1.0,
            "mfe_pct": 3.0,
            "mae_pct": -1.0,
            "gross_realized_return_pct": 2.5,
        },
    }


def test_paper_leverage_and_cash_exhaustion_and_cash_never_shorts():
    costs = load_execution_costs()
    paper_rows = [_sizing_row(index) for index in range(11)]
    paper = _sized_results(
        deepcopy(paper_rows), strategy_capital_usd=1000,
        leverage_cap=4.0, allow_shorts=True,
        exhausted_reason="LEVERAGE_CAP_EXCEEDED", costs=costs,
    )
    assert paper["ID10"]["status"] == "LEVERAGE_CAP_EXCEEDED"

    cash_rows = [_sizing_row(1), _sizing_row(2), _sizing_row(3), _sizing_row(4, direction="SHORT")]
    cash = _sized_results(
        deepcopy(cash_rows), strategy_capital_usd=1000,
        leverage_cap=1.0, allow_shorts=False,
        exhausted_reason="CASH_EXHAUSTED", costs=costs,
    )
    assert cash["ID3"]["status"] == "CASH_EXHAUSTED"
    assert cash["ID4"] == {"trade_executed": False, "status": "SHORT_NOT_ALLOWED"}
    assert not any(
        value.get("trade_executed") and value.get("direction") == "SHORT"
        for value in cash.values()
    )


def test_orb_research_does_not_mutate_scout_or_benchmark_outputs(tmp_path: Path):
    trading_date = "2024-03-15"
    prior_dates = previous_trading_sessions(trading_date, 15)
    daily = {
        prior_date: [{
            "ticker": "ORB",
            "stable_security_id": "SEC-ORB",
            "trading_date": prior_date,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "volume": 2_000_000,
            "transactions": 100,
            "source": "MASSIVE_FLATFILE",
            "session": "REGULAR",
        }]
        for prior_date in prior_dates
    }
    minute = [
        {
            "ticker": "ORB",
            "stable_security_id": "SEC-ORB",
            "timestamp": datetime(2024, 3, 15, 9, 30, tzinfo=ET).isoformat(),
            "open": 10.0,
            "high": 10.5,
            "low": 9.9,
            "close": 10.4,
            "volume": 500_000,
            "transactions": 100,
            "source": "MASSIVE_FLATFILE",
            "session": "REGULAR",
        },
        {
            "ticker": "ORB",
            "stable_security_id": "SEC-ORB",
            "timestamp": datetime(2024, 3, 15, 9, 35, tzinfo=ET).isoformat(),
            "open": 10.5,
            "high": 10.6,
            "low": 10.3,
            "close": 10.4,
            "volume": 100_000,
            "transactions": 50,
            "source": "MASSIVE_FLATFILE",
            "session": "REGULAR",
        },
    ]

    class FakeFlatFiles:
        def iter_bars(self, dataset, date_value, tickers=None):
            rows = minute if dataset == MINUTE_AGGS_DATASET else daily.get(date_value, [])
            yield from deepcopy(rows)

    cache_root = tmp_path / "opening"
    for prior_date in previous_trading_sessions(trading_date, 7):
        path = opening_cache_path(cache_root, prior_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "tickers": {"ORB": {"opening_volume": 100_000}},
        }))
    manifest = {
        "securities": [{
            "ticker": "ORB",
            "stable_security_id": "SEC-ORB",
            "security_type": "CS",
            "listing_venue": "XNAS",
            "reason_codes": ["MARKET_CAP_OUT_OF_RANGE"],
            "ticker_history": [],
        }],
    }
    scout = {
        "scout_id": "research_scout_alpha_v1.0",
        "candidates": [{"ticker": "ORB", "status": "SCORED", "research_selected": False}],
    }
    benchmark = {"benchmark_candidates": [{"ticker": "OTHER"}]}
    scout_before = json.dumps(scout, sort_keys=True, separators=(",", ":"))
    benchmark_before = json.dumps(benchmark, sort_keys=True, separators=(",", ":"))

    result = build_orb_research(
        trading_date,
        manifest,
        FakeFlatFiles(),
        scout,
        benchmark,
        strategy_capital_usd=2_500,
        output_path=tmp_path / "orb.csv",
        cache_root=cache_root,
    )

    assert json.dumps(scout, sort_keys=True, separators=(",", ":")) == scout_before
    assert json.dumps(benchmark, sort_keys=True, separators=(",", ":")) == benchmark_before
    assert result["orb_universe_count"] == 1
    assert result["ranked_candidates"][0]["outside_scout_band"] is True
