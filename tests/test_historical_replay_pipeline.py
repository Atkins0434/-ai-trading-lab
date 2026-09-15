from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trainer.historical_replay import run_single_day_replay
from trainer.replay_engine import ReplayError, load_historical_snapshot


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "2018-01-02" / "historical_snapshot.json"


def premarket_bars() -> list[dict]:
    start = datetime(2018, 1, 2, 10, 0, tzinfo=timezone.utc)
    result = []
    for index in range(60):
        close = 10.0 + index * 0.02
        result.append(
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
    return result


def outcome_bars() -> list[dict]:
    start = datetime(2018, 1, 2, 14, 30, tzinfo=timezone.utc)
    result = []
    for index in range(10):
        price = 12.0 + index * 0.03
        result.append(
            {
                "timestamp": (start + timedelta(minutes=index)).isoformat(),
                "open": price,
                "high": price + 0.04,
                "low": price - 0.04,
                "close": price + 0.02,
                "volume": 5000,
            }
        )
    return result


def enriched_snapshot() -> dict:
    snapshot = load_historical_snapshot(FIXTURE)
    snapshot["securities"] = [deepcopy(snapshot["securities"][0])]
    security = snapshot["securities"][0]
    observation = {
        "value": 9.5,
        "as_of_timestamp": "2018-01-01T16:00:00-05:00",
        "source": "TEST_FIXTURE",
        "provider_field": "previous_close",
    }
    security["market_data"]["previous_close"] = observation
    security["market_data"]["average_daily_range_pct"] = {
        **observation,
        "value": 1.0,
        "provider_field": "average_daily_range_pct",
    }
    security["premarket_bars"] = premarket_bars()
    return snapshot


def test_single_day_replay_runs_scout_trade_and_outcome_contract():
    result = run_single_day_replay(
        enriched_snapshot(),
        {"WINR": outcome_bars()},
        threshold_pct=35.0,
        strategy_capital=2500.0,
    )

    candidate = result["scout_output"]["candidates"][0]
    outcome = result["end_of_day_outcome"]["outcomes"][0]
    assert candidate["selected"] is True
    assert sum(
        component["status"] == "OBSERVED"
        for component in candidate["component_scores"].values()
    ) == 13
    assert outcome["selected"] is True
    assert outcome["execution_result"]["trade_executed"] is True
    assert outcome["execution_result"]["entry_timestamp"] == (
        "2018-01-02T14:30:00+00:00"
    )
    assert outcome["mfe_pct"] > 0
    assert outcome["maximum_capturable_move_pct"] > 0


def test_single_day_replay_rejects_future_premarket_bar():
    snapshot = enriched_snapshot()
    snapshot["securities"][0]["premarket_bars"][-1]["timestamp"] = (
        "2018-01-02T07:01:00-05:00"
    )

    with pytest.raises(ReplayError, match="premarket_bars"):
        run_single_day_replay(
            snapshot,
            {"WINR": outcome_bars()},
            threshold_pct=35.0,
        )
