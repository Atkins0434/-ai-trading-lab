from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trainer.price_volume import (
    PriceVolumeError,
    calculate_price_volume_metrics,
    score_thresholds,
)
from trainer.scout_engine import load_scout_config


def make_bars(count: int = 60) -> list[dict]:
    start = datetime(2018, 1, 2, 10, 0, tzinfo=timezone.utc)
    bars = []
    for index in range(count):
        close = 10.0 + index * 0.02
        bars.append(
            {
                "timestamp": (start + timedelta(minutes=index)).isoformat(),
                "open": close - 0.01,
                "high": close + 0.02,
                "low": close - 0.02,
                "close": close,
                "volume": 1000 + index * 50,
                "source": "TEST_FIXTURE",
            }
        )
    return bars


def make_security() -> dict:
    as_of = "2018-01-02T06:59:00-05:00"
    return {
        "market_data": {
            "previous_close": {"value": 9.5, "as_of_timestamp": as_of},
            "average_daily_range_pct": {
                "value": 1.0,
                "as_of_timestamp": as_of,
            },
        },
        "premarket_bars": make_bars(),
    }


def test_calculates_all_remaining_price_volume_metrics():
    metrics = calculate_price_volume_metrics(
        make_security(),
        load_scout_config(),
    )

    assert len(metrics) == 11
    assert set(metrics) == {
        "price_slope_15m",
        "price_slope_30m",
        "price_slope_60m",
        "volume_slope_15m",
        "volume_slope_30m",
        "volume_slope_60m",
        "premarket_gap_strength",
        "price_vs_premarket_vwap",
        "premarket_trend_consistency",
        "price_volume_confirmation",
        "premarket_range_expansion",
    }
    assert all(0 <= metric.score <= 4 for metric in metrics.values())
    assert metrics["price_slope_60m"].raw_value > 0
    assert metrics["premarket_gap_strength"].score == 4
    assert metrics["premarket_trend_consistency"].score == 4
    assert metrics["price_volume_confirmation"].score >= 3


def test_optional_baseline_inputs_remain_missing_not_zero():
    security = make_security()
    del security["market_data"]["previous_close"]
    del security["market_data"]["average_daily_range_pct"]

    metrics = calculate_price_volume_metrics(
        security,
        load_scout_config(),
    )

    assert "premarket_gap_strength" not in metrics
    assert "premarket_range_expansion" not in metrics


def test_non_contiguous_bars_are_rejected():
    security = make_security()
    security["premarket_bars"][30]["timestamp"] = (
        "2018-01-02T10:31:00+00:00"
    )

    with pytest.raises(PriceVolumeError, match="contiguous"):
        calculate_price_volume_metrics(security, load_scout_config())


def test_threshold_scoring_is_inclusive_and_monotonic():
    thresholds = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0}
    assert score_thresholds(0.99, thresholds) == 0
    assert score_thresholds(3.0, thresholds) == 3
    assert score_thresholds(9.0, thresholds) == 4

    with pytest.raises(PriceVolumeError, match="monotonic"):
        score_thresholds(3.0, {"1": 1, "2": 3, "3": 2, "4": 4})
