from __future__ import annotations

import csv
import json
from pathlib import Path

from trainer.scorable_outcomes import (
    concatenate_completed_outcomes,
    export_daily_outcomes,
)
from trainer.scorable_outcomes_schema import (
    eligible_outcomes_columns,
    METRIC_IDS,
    scorable_outcomes_columns,
)


PRIMARY = "execution_policy_v1.0"
ATR = "execution_policy_atr_v1.0"


def _security(ticker: str, count: int) -> dict:
    observation = lambda value: {"value": value}
    return {
        "ticker": ticker,
        "stable_security_id": f"sec-{ticker}",
        "market_data": {
            "real_bar_count_60m": observation(count),
            "real_bar_count": observation(count + 4),
            "previous_close": observation(9.5),
            "last_price": observation(10.0),
            "premarket_dollar_volume": observation(400_000.0),
            "average_daily_dollar_volume": observation(8_000_000.0),
        },
        "market_cap_usd": observation(50_000_000.0),
    }


def _candidate(
    ticker: str, status: str, *, selected: bool = False, rejected: bool = False
) -> dict:
    components = {
        metric_id: {
            "raw_value": (
                {"expansion_ratio": 1.5}
                if metric_id == "premarket_range_expansion"
                else {"price_vs_vwap_pct": 1.25}
                if metric_id == "price_vs_premarket_vwap"
                else {"aligned_positive_windows": 3}
                if metric_id == "price_volume_confirmation"
                else 2.5
            ),
            "score": 3,
        }
        for metric_id in METRIC_IDS
    }
    return {
        "ticker": ticker,
        "status": status,
        "shadow": status == "SHADOW_SCORED",
        "research_selected": selected,
        "selection_basis": "QUALIFYING_THRESHOLD" if selected else "NOT_SELECTED",
        "rejection_reasons": ["LIQUIDITY_BELOW_MINIMUM"] if rejected else [],
        "guardrails": {
            "aggregate_liquidity": {"action": "REJECT" if rejected else "PASS"}
        },
        "total_score": 36,
        "score_pct": 75.0,
        "reversal_score_pct": None if ticker == "NOPATH" else 72.5,
        "component_scores": components,
    }


def _execution(policy_id: str, realized: float = 0.0) -> dict:
    return {
        "policy_id": policy_id,
        "exit_reason": "SESSION_END" if realized else None,
        "realized_return_pct": realized,
        "capture_ratio": 0.5 if realized else None,
        "stop_distance_pct": 1.0 if realized else None,
    }


def _bars(ticker_index: int) -> list[dict]:
    opening = 10.0 + ticker_index
    return [
        {
            "timestamp": "2024-03-15T09:30:00-04:00",
            "open": opening, "high": opening * 1.05,
            "low": opening * 0.98, "close": opening * 1.02, "volume": 100,
        },
        {
            "timestamp": "2024-03-15T09:59:00-04:00",
            "open": opening * 1.02, "high": opening * 1.10,
            "low": opening, "close": opening * 1.08, "volume": 200,
        },
        {
            "timestamp": "2024-03-15T15:59:00-04:00",
            "open": opening * 1.08, "high": opening * 1.12,
            "low": opening * 0.97, "close": opening * 1.04, "volume": 300,
        },
    ]


def _artifacts() -> tuple[dict, dict, dict, dict]:
    tickers = ["SELECTED", "REJECTED", "SHADOW", "NOPATH", "UNSCORABLE"]
    snapshot = {
        "trading_date": "2024-03-15",
        "securities": [_security(ticker, 30 if ticker != "SHADOW" else 15) for ticker in tickers],
    }
    scout = {"candidates": [
        _candidate("SELECTED", "SCORED", selected=True),
        _candidate("REJECTED", "SCORED", rejected=True),
        _candidate("SHADOW", "SHADOW_SCORED"),
        _candidate("NOPATH", "SCORED"),
        {"ticker": "UNSCORABLE", "status": "NOT_SCORABLE"},
    ]}
    outcomes = []
    for index, ticker in enumerate(tickers):
        outcomes.append({
            "ticker": ticker,
            "intraday_path": [] if ticker == "NOPATH" else _bars(index),
            "execution_result": _execution(PRIMARY, 4.0 if ticker == "SELECTED" else 0.0),
        })
    outcome = {
        "outcomes": outcomes,
        "policy_comparisons": [{
            "policy_id": ATR,
            "executions": [{
                "ticker": "SELECTED", "cohort": "SCOUT_SELECTION",
                "execution_result": _execution(ATR, 3.0),
            }],
        }],
    }
    benchmark = {"benchmark_candidates": [
        {"ticker": "REJECTED", "benchmark_rank": 1},
        {"ticker": "SELECTED", "benchmark_rank": 2},
    ]}
    return snapshot, scout, outcome, benchmark


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_daily_exports_cover_scorable_and_eligible_universes(tmp_path: Path):
    scorable, eligible = export_daily_outcomes(tmp_path, *_artifacts())
    fields, rows = _read(scorable)
    assert fields == list(scorable_outcomes_columns((PRIMARY, ATR)))
    assert [row["ticker"] for row in rows] == [
        "NOPATH", "REJECTED", "SELECTED", "SHADOW"
    ]
    by_ticker = {row["ticker"]: row for row in rows}
    assert by_ticker["NOPATH"]["no_regular_session_path"] == "true"
    assert by_ticker["NOPATH"]["open_0930"] == ""
    assert by_ticker["SHADOW"]["shadow"] == "true"
    assert by_ticker["REJECTED"]["liquidity_action"] == "REJECT"
    assert by_ticker["REJECTED"]["rejection_reasons"] == "LIQUIDITY_BELOW_MINIMUM"
    assert by_ticker["SELECTED"]["return_0930_1000_pct"] == "8.000000"
    assert by_ticker["SELECTED"][f"{ATR}_realized_return_pct"] == "3.000000"
    assert by_ticker["SHADOW"]["premarket_range_expansion_raw"] == "1.500000"

    eligible_fields, eligible_rows = _read(eligible)
    assert eligible_fields == list(eligible_outcomes_columns((PRIMARY, ATR)))
    assert len(eligible_rows) == len(_artifacts()[0]["securities"])
    assert {row["ticker"] for row in eligible_rows} == {
        "SELECTED", "REJECTED", "SHADOW", "NOPATH", "UNSCORABLE"
    }


def test_daily_export_is_byte_deterministic(tmp_path: Path):
    scorable, eligible = export_daily_outcomes(tmp_path, *_artifacts())
    first = (scorable.read_bytes(), eligible.read_bytes())
    export_daily_outcomes(tmp_path, *_artifacts())
    assert (scorable.read_bytes(), eligible.read_bytes()) == first


def test_cumulative_counts_equal_sum_of_daily_counts(tmp_path: Path):
    expected = {"scorable_outcomes.csv": 0, "eligible_outcomes.csv": 0}
    for trading_date in ("2024-03-15", "2024-03-18"):
        day = tmp_path / "days" / trading_date
        artifacts = list(_artifacts())
        artifacts[0] = {**artifacts[0], "trading_date": trading_date}
        paths = export_daily_outcomes(day, *artifacts)
        for path in paths:
            expected[path.name.replace("_" + trading_date, "")] += len(_read(path)[1])
    cumulative = concatenate_completed_outcomes(
        tmp_path, ["2024-03-18", "2024-03-15"]
    )
    for path in cumulative:
        assert len(_read(path)[1]) == expected[path.name]
        dates = [row["trading_date"] for row in _read(path)[1]]
        assert dates == sorted(dates)
