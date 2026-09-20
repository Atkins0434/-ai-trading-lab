from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import trainer.research_scout_alpha as research_scout_alpha
from trainer.massive_alpha_snapshot import build_massive_alpha_snapshot, real_premarket_bars
from trainer.research_scout_alpha import run_research_scout_alpha


ET = ZoneInfo("America/New_York")
METRIC_IDS = (
    "relative_volume", "price_slope_15m", "price_slope_30m",
    "price_slope_60m", "volume_slope_15m", "volume_slope_30m",
    "volume_slope_60m", "premarket_gap_strength",
    "price_vs_premarket_vwap", "premarket_trend_consistency",
    "price_volume_confirmation", "premarket_range_expansion",
)


def aggregate(timestamp: datetime, price: float, volume: float = 1000) -> dict:
    return {
        "t": int(timestamp.timestamp() * 1000),
        "o": price,
        "h": price + 0.05,
        "l": price - 0.05,
        "c": price + 0.02,
        "v": volume,
    }


def alpha_inputs() -> tuple[list[dict], list[dict]]:
    target = date(2026, 9, 14)
    trading_days = []
    cursor = target - timedelta(days=1)
    while len(trading_days) < 20:
        if cursor.weekday() < 5:
            trading_days.append(cursor)
        cursor -= timedelta(days=1)
    trading_days.reverse()
    daily = [
        aggregate(datetime.combine(day, time(0), tzinfo=ET), 10 + index / 100, 1_000_000)
        for index, day in enumerate(trading_days)
    ]
    intraday = []
    for day in trading_days + [target]:
        for minute in range(315):
            timestamp = datetime.combine(day, time(4), tzinfo=ET) + timedelta(minutes=minute)
            intraday.append(aggregate(timestamp, 10 + minute / 1000, 5000))
    return daily, intraday


def _reversal_components(**raw_values):
    components = {}
    for metric_id in METRIC_IDS:
        raw = raw_values.get(metric_id, 1.0)
        components[metric_id] = {
            "status": "OBSERVED",
            "score": int(raw_values.get(f"{metric_id}_score", 4)),
            "maximum_score": 4,
            "raw_value": raw,
            "as_of_timestamp": "2026-09-14T09:14:00-04:00",
            "reason_code": "TEST",
            "calculation_version": "test",
        }
    return components


def test_smr_reversal_fixture_scores_above_threshold():
    _, _, reversal = research_scout_alpha._load_contract()
    components = _reversal_components(
        premarket_gap_strength=-12.0,
        price_vs_premarket_vwap={
            "vwap": 9.0, "price_vs_vwap_pct": -3.5
        },
        price_slope_60m=-0.05,
        premarket_trend_consistency=35.0,
        price_slope_15m=0.40,
        price_slope_30m=0.094,
        relative_volume=3.56,
    )

    status, total, score_pct, scored = research_scout_alpha._reversal_score(
        components, reversal
    )

    assert status == "SCORED"
    assert total is not None
    assert score_pct >= 70.0
    assert scored["premarket_gap_strength"]["score"] == 3
    assert scored["price_vs_premarket_vwap"]["score"] == 3
    assert scored["premarket_trend_consistency"]["score"] == 3


def test_cdlx_positive_gap_and_vwap_score_zero_and_stay_below_threshold():
    _, _, reversal = research_scout_alpha._load_contract()
    components = _reversal_components(
        premarket_gap_strength=52.0,
        price_vs_premarket_vwap={
            "vwap": 9.0, "price_vs_vwap_pct": 25.0
        },
        price_slope_60m=0.08,
        premarket_trend_consistency=80.0,
        price_slope_15m=0.40,
        price_slope_30m=0.094,
        relative_volume_score=2,
        volume_slope_15m_score=2,
        volume_slope_30m_score=2,
        volume_slope_60m_score=2,
        price_volume_confirmation_score=2,
        premarket_range_expansion_score=2,
    )

    status, _, score_pct, scored = research_scout_alpha._reversal_score(
        components, reversal
    )

    assert status == "SCORED"
    assert scored["premarket_gap_strength"]["score"] == 0
    assert scored["price_vs_premarket_vwap"]["score"] == 0
    assert score_pct < 70.0


def test_negative_30m_slope_is_not_a_reversal_candidate():
    _, _, reversal = research_scout_alpha._load_contract()
    components = _reversal_components(
        price_slope_15m=0.40,
        price_slope_30m=-0.01,
    )

    status, total, score_pct, _ = research_scout_alpha._reversal_score(
        components, reversal
    )

    assert status == "NOT_A_REVERSAL_CANDIDATE"
    assert total is None
    assert score_pct is None


def test_shadow_reversal_score_is_never_selected():
    _, _, reversal = research_scout_alpha._load_contract()
    candidates = [
        {
            "ticker": "VISIBLE",
            "research_eligible": True,
            "status": "SCORED",
            "qualification_selected": False,
            "research_selected": False,
            "selection_basis": "NOT_SELECTED",
            "reversal_status": "SCORED",
            "reversal_score_pct": 90.0,
            "reason_codes": ["RESEARCH_ALPHA_NOT_SELECTED"],
            "rejection_reasons": ["BELOW_RESEARCH_THRESHOLD"],
        },
        {
            "ticker": "SHADOW",
            "research_eligible": True,
            "status": "SHADOW_SCORED",
            "qualification_selected": False,
            "research_selected": False,
            "selection_basis": "NOT_SELECTED",
            "reversal_status": "SCORED",
            "reversal_score_pct": 100.0,
            "reason_codes": ["SHADOW_SCORED_DIAGNOSTIC_ONLY"],
            "rejection_reasons": ["INSUFFICIENT_PREMARKET_BARS"],
        },
    ]

    selected = research_scout_alpha._select_reversal_exploration(
        candidates, reversal
    )

    assert [item["ticker"] for item in selected] == ["VISIBLE"]
    assert candidates[0]["selection_basis"] == "REVERSAL_EXPLORATION"
    assert candidates[1]["research_selected"] is False
    assert candidates[1]["selection_basis"] == "NOT_SELECTED"


def test_reversal_exploration_selects_top_two_non_primary_candidates():
    _, _, reversal = research_scout_alpha._load_contract()
    candidates = []
    for ticker, score, primary in (
        ("PRIMARY", 100.0, True),
        ("FIRST", 95.0, False),
        ("SECOND", 90.0, False),
        ("THIRD", 85.0, False),
    ):
        candidates.append({
            "ticker": ticker,
            "research_eligible": True,
            "status": "SCORED",
            "qualification_selected": primary,
            "research_selected": primary,
            "selection_basis": (
                "QUALIFYING_THRESHOLD" if primary else "NOT_SELECTED"
            ),
            "reversal_status": "SCORED",
            "reversal_score_pct": score,
            "reason_codes": ["RESEARCH_ALPHA_NOT_SELECTED"],
            "rejection_reasons": ["BELOW_RESEARCH_THRESHOLD"],
        })

    selected = research_scout_alpha._select_reversal_exploration(
        candidates, reversal
    )

    assert [item["ticker"] for item in selected] == ["FIRST", "SECOND"]
    assert candidates[0]["selection_basis"] == "QUALIFYING_THRESHOLD"
    assert candidates[3]["selection_basis"] == "NOT_SELECTED"


def test_real_premarket_bars_never_fill_missing_minutes():
    _, intraday = alpha_inputs()
    target_records = [
        record for index, record in enumerate(intraday[-315:]) if index != 125
    ]
    bars = real_premarket_bars(
        target_records,
        "2026-09-14",
        config={"morning_freeze_time": "09:15:00"},
    )

    assert len(bars) == 314
    assert all(bar["volume"] > 0 for bar in bars)
    assert all(bar["source"] == "MASSIVE" for bar in bars)


def test_snapshot_records_real_bar_counts():
    daily, intraday = alpha_inputs()
    target = date(2026, 9, 14)
    thinned = []
    for record in intraday:
        observed = datetime.fromtimestamp(record["t"] / 1000, tz=ET)
        if (
            observed.date() == target
            and time(8, 15) <= observed.time() < time(8, 45)
        ):
            continue
        thinned.append(record)

    snapshot = build_massive_alpha_snapshot(
        "TEST", "2026-09-14", daily, thinned, exchange="NASDAQ"
    )
    security = snapshot["securities"][0]
    real_count = security["market_data"]["real_bar_count"]
    real_count_60m = security["market_data"]["real_bar_count_60m"]

    assert len(security["premarket_bars"]) == 285
    assert real_count["value"] == 285
    assert real_count_60m["value"] == 30
    assert real_count["as_of_timestamp"] == snapshot["freeze_timestamp"]


def test_alpha_is_100_points_and_never_execution_eligible():
    daily, intraday = alpha_inputs()
    snapshot = build_massive_alpha_snapshot(
        "TEST", "2026-09-14", daily, intraday, exchange="NASDAQ"
    )
    result = run_research_scout_alpha(snapshot, threshold_pct=0)
    candidate = result["candidates"][0]

    assert result["mode"] == "RESEARCH_ONLY"
    assert result["scout_version"] == "research_scout_alpha_v1.3"
    assert len(candidate["component_scores"]) == 25
    assert candidate["maximum_possible_score"] == 100
    assert candidate["research_selected"] is True
    assert candidate["execution_eligible"] is False
    assert "HISTORICAL_SPREAD" in candidate["unavailable_execution_checks"]
    assert candidate["guardrails"]["historical_spread"] == {
        "passed": None,
        "action": "NOT_EVALUATED",
        "observed_value": None,
        "threshold": 0.4,
        "reason_code": "SPREAD_QUOTES_UNAVAILABLE",
    }
    assert candidate["guardrails"]["order_book_depth"]["passed"] is None
    assert "SPREAD_QUOTES_UNAVAILABLE" in candidate["reason_codes"]
    assert "ORDER_BOOK_DEPTH_UNAVAILABLE" in candidate["reason_codes"]


def test_real_bar_minimum_controls_scorability():
    daily, intraday = alpha_inputs()
    target = date(2026, 9, 14)

    def snapshot_with_last_hour_count(count: int):
        kept = []
        retained_last_hour = 0
        for record in intraday:
            observed = datetime.fromtimestamp(record["t"] / 1000, tz=ET)
            in_target_last_hour = (
                observed.date() == target
                and time(8, 15) <= observed.time() < time(9, 15)
            )
            if in_target_last_hour:
                if retained_last_hour >= count:
                    continue
                retained_last_hour += 1
            kept.append(record)
        return build_massive_alpha_snapshot(
            "TEST", "2026-09-14", daily, kept, exchange="NASDAQ"
        )

    scored = run_research_scout_alpha(
        snapshot_with_last_hour_count(45), threshold_pct=0
    )
    shadow = run_research_scout_alpha(
        snapshot_with_last_hour_count(20), threshold_pct=0
    )
    not_scorable = run_research_scout_alpha(
        snapshot_with_last_hour_count(5), threshold_pct=0
    )

    assert scored["candidates"][0]["status"] == "SCORED"
    assert scored["scorable_candidate_count"] == 1
    assert shadow["candidates"][0]["status"] == "SHADOW_SCORED"
    assert shadow["candidates"][0]["shadow"] is True
    assert shadow["candidates"][0]["research_selected"] is False
    assert shadow["shadow_scored_candidate_count"] == 1
    assert shadow["scorable_candidate_count"] == 0
    assert shadow["not_scorable_candidate_count"] == 1
    assert shadow["eligible_universe_count"] == 1
    assert shadow["qualifying_candidate_count"] == 0
    assert shadow["selected_candidate_count"] == 0
    assert not_scorable["candidates"][0]["status"] == "NOT_SCORABLE"
    assert "INSUFFICIENT_PREMARKET_BARS" in (
        not_scorable["candidates"][0]["reason_codes"]
    )
    assert not_scorable["eligible_universe_count"] == 1


def test_alpha_can_restore_missing_market_data_rejection(monkeypatch):
    daily, intraday = alpha_inputs()
    snapshot = build_massive_alpha_snapshot(
        "TEST", "2026-09-14", daily, intraday, exchange="NASDAQ"
    )
    config, registry, reversal = research_scout_alpha._load_contract()
    config = deepcopy(config)
    config["spread"]["missing_quotes_policy"] = "REJECT"
    config["order_book_depth"]["missing_depth_policy"] = "REJECT"
    monkeypatch.setattr(
        research_scout_alpha,
        "_load_contract",
        lambda: (config, registry, reversal),
    )

    candidate = run_research_scout_alpha(
        snapshot, threshold_pct=0
    )["candidates"][0]

    assert candidate["research_selected"] is False
    assert candidate["research_eligible"] is False
    assert candidate["guardrails"]["historical_spread"]["action"] == "REJECT"
    assert candidate["guardrails"]["order_book_depth"]["action"] == "REJECT"
    assert "SPREAD_QUOTES_UNAVAILABLE" in candidate["rejection_reasons"]
    assert "ORDER_BOOK_DEPTH_UNAVAILABLE" in candidate["rejection_reasons"]


def test_pipeline_validation_symbol_can_be_scored_without_becoming_candidate():
    daily, intraday = alpha_inputs()
    snapshot = build_massive_alpha_snapshot(
        "SPY",
        "2026-09-14",
        daily,
        intraday,
        exchange="ARCA",
        eligible=False,
        eligibility_reasons=["DATA_PIPELINE_VALIDATION_SYMBOL"],
    )
    candidate = run_research_scout_alpha(snapshot, threshold_pct=0)["candidates"][0]

    assert candidate["total_score"] >= 0
    assert candidate["research_selected"] is False
    assert "DATA_PIPELINE_VALIDATION_SYMBOL" in candidate["rejection_reasons"]


def test_exploration_selects_top_eligible_candidate_without_lowering_threshold():
    daily, intraday = alpha_inputs()
    snapshot = build_massive_alpha_snapshot(
        "TEST", "2026-09-14", daily, intraday, exchange="NASDAQ"
    )
    result = run_research_scout_alpha(
        snapshot,
        threshold_pct=100,
        exploration_top_k=1,
    )
    candidate = result["candidates"][0]

    assert result["scoring_threshold_pct"] == 100
    assert result["qualifying_candidate_count"] == 0
    assert result["exploration_candidate_count"] == 1
    assert result["selected_candidate_count"] == 1
    assert candidate["qualification_selected"] is False
    assert candidate["research_selected"] is True
    assert candidate["selection_basis"] == "EXPLORATION_TOP_K"
    assert candidate["rank"] == 1
    assert "BELOW_RESEARCH_THRESHOLD" not in candidate["rejection_reasons"]
