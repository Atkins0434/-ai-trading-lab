from pathlib import Path

from trainer.replay_engine import load_historical_snapshot
from trainer.scout_engine import (
    minimum_points_for_threshold,
    run_scout,
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
