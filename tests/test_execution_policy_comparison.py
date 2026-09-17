from __future__ import annotations

from pathlib import Path

import pytest

from trainer.outcome_grader import grade_replay_outcomes
from trainer.trade_engine import load_execution_policy, simulate_trade


ROOT = Path(__file__).resolve().parents[1]
ATR_POLICY_PATH = ROOT / "config" / "execution_policy_atr.json"


def _bars() -> list[dict]:
    return [
        {
            "timestamp": "2024-03-15T09:30:00-04:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1_000,
        },
        {
            "timestamp": "2024-03-15T09:31:00-04:00",
            "open": 100.5,
            "high": 102.0,
            "low": 99.8,
            "close": 101.5,
            "volume": 1_100,
        },
        {
            "timestamp": "2024-03-15T09:32:00-04:00",
            "open": 101.5,
            "high": 108.5,
            "low": 100.0,
            "close": 108.0,
            "volume": 1_200,
        },
    ]


def test_percent_policy_preserves_legacy_execution_values():
    policy = load_execution_policy()
    result = simulate_trade(
        "TEST", 100.0, _bars(), 2500.0, policy=policy
    )

    legacy_fields = {
        key: result[key]
        for key in (
            "ticker", "trade_executed", "entry_price",
            "position_size_shares", "position_value_usd",
            "exit_timestamp", "exit_price", "exit_reason",
            "realized_pnl_usd", "realized_return_pct",
            "highest_price_since_entry",
        )
    }
    assert legacy_fields == {
        "ticker": "TEST",
        "trade_executed": True,
        "entry_price": 100.0,
        "position_size_shares": 5,
        "position_value_usd": 500.0,
        "exit_timestamp": "2024-03-15T09:31:00-04:00",
        "exit_price": 99.99,
        "exit_reason": "TRAILING_STOP",
        "realized_pnl_usd": pytest.approx(-0.05),
        "realized_return_pct": pytest.approx(-0.01),
        "highest_price_since_entry": 101.0,
    }
    assert result["policy_id"] == "execution_policy_v1.0"
    assert result["exit_mode"] == "PERCENT"
    assert result["sizing_mode"] == "FIXED_FRACTION"


def test_atr_stop_holds_after_percent_stop_and_reaches_target():
    percent = simulate_trade("TEST", 100.0, _bars(), 2500.0)
    atr = simulate_trade(
        "TEST",
        100.0,
        _bars(),
        2500.0,
        policy=load_execution_policy(ATR_POLICY_PATH),
        atr_14_usd=2.0,
    )

    assert percent["exit_reason"] == "TRAILING_STOP"
    assert percent["exit_timestamp"] == "2024-03-15T09:31:00-04:00"
    assert atr["exit_reason"] == "PROFIT_TARGET"
    assert atr["exit_timestamp"] == "2024-03-15T09:32:00-04:00"
    assert atr["exit_price"] == 108.0
    assert atr["stop_distance_usd"] == 3.0
    assert atr["target_distance_usd"] == 8.0


def test_risk_sizing_shrinks_with_atr_and_respects_position_cap():
    policy = load_execution_policy(ATR_POLICY_PATH)
    low_atr = simulate_trade(
        "LOW", 100.0, _bars(), 10_000.0, policy=policy, atr_14_usd=0.1
    )
    high_atr = simulate_trade(
        "HIGH", 100.0, _bars(), 10_000.0, policy=policy, atr_14_usd=5.0
    )

    assert low_atr["position_size_shares"] == 20
    assert low_atr["position_value_usd"] == 2_000.0
    assert high_atr["position_size_shares"] == 13
    assert high_atr["position_size_shares"] < low_atr["position_size_shares"]
    assert high_atr["risk_usd_at_entry"] == 97.5


def test_atr_rejections_record_specific_reasons():
    policy = load_execution_policy(ATR_POLICY_PATH)
    unavailable = simulate_trade(
        "NONE", 100.0, _bars(), 2500.0, policy=policy, atr_14_usd=None
    )
    too_small = simulate_trade(
        "TINY", 100.0, _bars(), 2500.0, policy=policy, atr_14_usd=100.0
    )

    assert unavailable["exit_reason"] == "ENTRY_REJECTED"
    assert unavailable["entry_rejection_reason"] == "ATR_UNAVAILABLE"
    assert too_small["exit_reason"] == "ENTRY_REJECTED"
    assert too_small["entry_rejection_reason"] == "POSITION_TOO_SMALL"


def _snapshot() -> dict:
    observation = {
        "value": 2.0,
        "as_of_timestamp": "2024-03-14T16:00:00-04:00",
        "source": "TEST",
        "provider_field": "mean(true_range),14_sessions",
    }
    return {
        "replay_id": "policy-comparison",
        "trading_date": "2024-03-15",
        "scout_version": "research_scout_alpha_v1.0",
        "execution_policy_version": "execution_policy_v1.0_hypothetical",
        "universe_mode": "historical_research",
        "universe_manifest_hash": "sha256:" + "a" * 64,
        "universe_coverage": "complete",
        "research_evidence": True,
        "promotion_eligible": True,
        "securities": [{
            "ticker": "TEST",
            "market_data": {"atr_14_usd": observation},
        }],
    }


def test_comparison_policy_does_not_change_primary_outcomes(capsys):
    scout = {
        "candidates": [{
            "ticker": "TEST",
            "research_selected": True,
            "selection_basis": "QUALIFYING_THRESHOLD",
            "rank": 1,
            "score_pct": 77.1,
        }]
    }
    primary_only = grade_replay_outcomes(
        _snapshot(), scout, {"TEST": _bars()}, 2500.0
    )
    compared = grade_replay_outcomes(
        _snapshot(),
        scout,
        {"TEST": _bars()},
        2500.0,
        comparison_policy_paths=[ATR_POLICY_PATH],
    )

    assert primary_only["outcomes"] == compared["outcomes"]
    assert primary_only["policy_comparisons"] == []
    assert len(compared["policy_comparisons"]) == 1
    comparison = compared["policy_comparisons"][0]
    assert comparison["policy_id"] == "execution_policy_atr_v1.0"
    assert comparison["summary"]["net_realized_pnl_usd"] == 40.0
    assert {
        item["cohort"] for item in comparison["executions"]
    } == {"SCOUT_SELECTION", "TOP_10_MOVER"}
    captured = capsys.readouterr()
    assert "policy_id=execution_policy_v1.0" in captured.err
    assert "policy_id=execution_policy_atr_v1.0" in captured.err
