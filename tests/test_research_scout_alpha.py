from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from trainer.massive_alpha_snapshot import build_massive_alpha_snapshot, regularize_last_premarket_hour
from trainer.research_scout_alpha import run_research_scout_alpha


ET = ZoneInfo("America/New_York")


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
        for minute in range(180):
            timestamp = datetime.combine(day, time(4), tzinfo=ET) + timedelta(minutes=minute)
            intraday.append(aggregate(timestamp, 10 + minute / 1000, 5000))
    return daily, intraday


def test_regularization_marks_zero_volume_carry_forward():
    _, intraday = alpha_inputs()
    target_records = [
        record for index, record in enumerate(intraday[-180:]) if index != 125
    ]
    bars = regularize_last_premarket_hour(target_records, "2026-09-14")

    assert len(bars) == 60
    assert bars[5]["volume"] == 0
    assert bars[5]["source"] == "MASSIVE_ZERO_VOLUME_CARRY_FORWARD"


def test_snapshot_records_regularization_padding_as_frozen_observation():
    daily, intraday = alpha_inputs()
    target = date(2026, 9, 14)
    thinned = []
    for record in intraday:
        observed = datetime.fromtimestamp(record["t"] / 1000, tz=ET)
        if (
            observed.date() == target
            and time(6, 0) <= observed.time() < time(6, 30)
        ):
            continue
        thinned.append(record)

    snapshot = build_massive_alpha_snapshot(
        "TEST", "2026-09-14", daily, thinned, exchange="NASDAQ"
    )
    security = snapshot["securities"][0]
    padding = security["market_data"]["padded_bar_count"]

    assert len(security["premarket_bars"]) == 60
    assert padding["value"] == 30
    assert padding["as_of_timestamp"] == security["premarket_bars"][-1]["timestamp"]


def test_alpha_is_48_points_and_never_execution_eligible():
    daily, intraday = alpha_inputs()
    snapshot = build_massive_alpha_snapshot(
        "TEST", "2026-09-14", daily, intraday, exchange="NASDAQ"
    )
    result = run_research_scout_alpha(snapshot, threshold_pct=0)
    candidate = result["candidates"][0]

    assert result["mode"] == "RESEARCH_ONLY"
    assert result["scout_version"] == "research_scout_alpha_v1.0"
    assert len(candidate["component_scores"]) == 12
    assert candidate["maximum_possible_score"] == 48
    assert candidate["research_selected"] is True
    assert candidate["execution_eligible"] is False
    assert "HISTORICAL_SPREAD" in candidate["unavailable_execution_checks"]


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
