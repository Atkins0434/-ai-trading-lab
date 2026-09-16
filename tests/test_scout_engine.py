from copy import deepcopy
from pathlib import Path

import pytest

from trainer.replay_engine import load_historical_snapshot
from trainer.scout_engine import (
    evaluate_order_book_depth_guardrail,
    evaluate_spread_guardrail,
    load_feature_registry,
    load_scout_config,
    minimum_points_for_threshold,
    run_scout,
    score_security,
)


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = (
    ROOT
    / "fixtures"
    / "2018-01-02"
    / "historical_snapshot.json"
)


def get_result():
    snapshot = load_historical_snapshot(FIXTURE)
    return run_scout(snapshot, threshold_pct=85.0)


def by_ticker(result):
    return {
        candidate["ticker"]: candidate
        for candidate in result["candidates"]
    }


def test_expected_candidate_count():
    result = get_result()

    assert result["qualifying_candidate_count"] == 0


def test_partial_fixture_uses_fixed_120_point_contract():
    result = get_result()
    candidates = by_ticker(result)

    winr = candidates["WINR"]

    assert winr["eligible"] is True
    assert winr["selected"] is False
    assert winr["total_score"] == 7
    assert winr["maximum_possible_score"] == 120
    assert winr["score_pct"] == 7 / 120 * 100
    assert winr["threshold_points"] == 102
    assert winr["rank"] is None
    assert len(winr["component_scores"]) == 30


def test_missing_metric_is_not_observed_zero():
    winr = by_ticker(get_result())["WINR"]

    missing = winr["component_scores"]["price_slope_15m"]
    observed = winr["component_scores"]["relative_volume"]

    assert missing["status"] == "MISSING"
    assert missing["score"] is None
    assert observed["status"] == "OBSERVED"
    assert observed["score"] == 4


def test_percentage_thresholds_round_up_to_attainable_points():
    assert minimum_points_for_threshold(85) == 102
    assert minimum_points_for_threshold(96) == 116
    assert minimum_points_for_threshold(97) == 117
    assert minimum_points_for_threshold(98) == 118


def test_wide_is_rejected_for_spread():
    result = get_result()
    candidates = by_ticker(result)

    wide = candidates["WIDE"]

    assert wide["eligible"] is False
    assert wide["selected"] is False
    assert (
        wide["guardrails"]["spread"]["reason_code"]
        == "SPREAD_HARD_REJECT"
    )
    assert "SPREAD_HARD_REJECT" in wide["rejection_reasons"]


def test_thin_is_rejected_for_liquidity():
    result = get_result()
    candidates = by_ticker(result)

    thin = candidates["THIN"]

    assert thin["eligible"] is False
    assert thin["selected"] is False
    assert (
        thin["guardrails"]["liquidity"]["reason_code"]
        == "LIQUIDITY_BELOW_MINIMUM"
    )
    assert "LIQUIDITY_BELOW_MINIMUM" in thin["rejection_reasons"]


def test_meh_passes_gates_but_fails_score():
    result = get_result()
    candidates = by_ticker(result)

    meh = candidates["MEH"]

    assert meh["eligible"] is True
    assert meh["selected"] is False
    assert meh["score_pct"] < 85.0
    assert "BELOW_SELECTION_THRESHOLD" in meh["rejection_reasons"]


def test_only_selected_candidates_receive_rank():
    result = get_result()
    candidates = by_ticker(result)

    assert candidates["WINR"]["rank"] is None
    assert candidates["WIDE"]["rank"] is None
    assert candidates["THIN"]["rank"] is None
    assert candidates["MEH"]["rank"] is None


@pytest.mark.parametrize(
    ("policy", "expected_action", "expected_passed"),
    [
        ("NOT_EVALUATED", "NOT_EVALUATED", None),
        ("REJECT", "REJECT", False),
    ],
)
def test_missing_spread_quotes_follow_configured_policy(
    policy, expected_action, expected_passed
):
    config = load_scout_config()
    config["spread"]["missing_quotes_policy"] = policy

    result = evaluate_spread_guardrail({}, config)

    assert result["action"] == expected_action
    assert result["passed"] is expected_passed
    assert result["reason_code"] == "SPREAD_QUOTES_UNAVAILABLE"


@pytest.mark.parametrize(
    ("policy", "expected_action", "expected_passed"),
    [
        ("NOT_EVALUATED", "NOT_EVALUATED", None),
        ("REJECT", "REJECT", False),
    ],
)
def test_missing_order_book_depth_follows_configured_policy(
    policy, expected_action, expected_passed
):
    config = load_scout_config()
    config["order_book_depth"]["missing_depth_policy"] = policy

    result = evaluate_order_book_depth_guardrail({}, config)

    assert result["action"] == expected_action
    assert result["passed"] is expected_passed
    assert result["reason_code"] == "ORDER_BOOK_DEPTH_UNAVAILABLE"


def test_passing_candidate_with_only_missing_market_guardrails_is_selected():
    snapshot = load_historical_snapshot(FIXTURE)
    security = deepcopy(snapshot["securities"][0])
    security["market_data"].pop("bid", None)
    security["market_data"].pop("ask", None)

    candidate = score_security(
        security,
        load_scout_config(),
        threshold_pct=0,
        timestamp=snapshot["freeze_timestamp"],
        registry=load_feature_registry(),
    )

    assert candidate["selected"] is True
    assert candidate["guardrails"]["spread"]["action"] == "NOT_EVALUATED"
    assert candidate["guardrails"]["spread"]["passed"] is None
    assert candidate["guardrails"]["order_book_depth"]["action"] == (
        "NOT_EVALUATED"
    )
    assert "SPREAD_QUOTES_UNAVAILABLE" in candidate["reason_codes"]
    assert "ORDER_BOOK_DEPTH_UNAVAILABLE" in candidate["reason_codes"]
    assert "SPREAD_QUOTES_UNAVAILABLE" not in candidate["rejection_reasons"]
    assert "ORDER_BOOK_DEPTH_UNAVAILABLE" not in candidate["rejection_reasons"]
