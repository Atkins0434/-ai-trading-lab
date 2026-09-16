from __future__ import annotations

from copy import deepcopy

import pytest

from trainer.benchmark import (
    RANDOM_BASELINE_DRAWS,
    RANDOM_BASELINE_SEED,
    build_same_universe_benchmark,
)
from trainer.outcome_grader import grade_replay_outcomes
from trainer.postmortem import build_postmortem
from trainer.postmortem_report import generate_postmortem_pdf


EVIDENCE = {
    "universe_mode": "historical_research",
    "universe_manifest_hash": "sha256:" + "a" * 64,
    "universe_coverage": "complete",
    "research_evidence": True,
    "promotion_eligible": True,
}


def _bars(close: float = 10.5, *, high: float = 11.0, low: float = 10.0):
    return [{
        "timestamp": "2026-09-14T13:30:00+00:00",
        "open": 10.0,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000,
    }]


def _component_scores(relative_volume=2.0, gap_pct=2.0):
    return {
        "premarket_gap_strength": {
            "status": "OBSERVED", "score": 1, "raw_value": gap_pct
        },
        "relative_volume": {
            "status": "OBSERVED", "score": 1, "raw_value": relative_volume
        },
        "price_slope_15m": {
            "status": "OBSERVED", "score": 0, "raw_value": -0.1
        },
    }


def _candidate(*, selected=False, total_score=20, guardrail_action="PASS"):
    return {
        "ticker": "MOVE",
        "research_eligible": guardrail_action != "REJECT",
        "research_selected": selected,
        "score_pct": total_score / 48 * 100,
        "total_score": total_score,
        "threshold_points": 30,
        "rejection_reasons": (
            ["LOW_AGGREGATE_LIQUIDITY"]
            if guardrail_action == "REJECT"
            else (["BELOW_RESEARCH_THRESHOLD"] if not selected else [])
        ),
        "component_scores": _component_scores(),
        "guardrails": {
            "aggregate_liquidity": {
                "action": guardrail_action,
                "reason_code": "LOW_AGGREGATE_LIQUIDITY",
            }
        },
    }


def _snapshot(*, eligible=True, bar_count=60):
    return {
        "replay_id": "test-replay",
        "trading_date": "2026-09-14",
        "universe_version": "research_universe_v1.0",
        **EVIDENCE,
        "securities": [{
            "ticker": "MOVE",
            "eligible": eligible,
            "eligibility_reasons": (
                [] if eligible else ["MARKET_CAP_OUT_OF_RANGE"]
            ),
            "premarket_bars": [{} for _ in range(bar_count)],
        }],
    }


def _benchmark_candidate(*, selected=False, universe_eligible=True):
    return {
        "ticker": "MOVE",
        "benchmark_rank": 1,
        "raw_move_pct": 10.0,
        "maximum_capturable_move_pct": 10.0,
        "day_high_timestamp": "2026-09-14T15:00:00+00:00",
        "universe_eligible": universe_eligible,
        "scout_selected": selected,
        "scout_trade_executed": selected,
        "scout_realized_return_pct": 2.0 if selected else 0.0,
        "scout_exit_reason": "TRAILING_STOP" if selected else None,
        "scout_exit_timestamp": (
            "2026-09-14T14:00:00+00:00" if selected else None
        ),
        "fair_simulation_possible": universe_eligible,
        "simulation_exclusion_reason": (
            None if universe_eligible else "NOT_IN_ELIGIBLE_UNIVERSE"
        ),
        "trade_executed": universe_eligible,
        "realized_return_pct": 5.0 if universe_eligible else None,
        "realized_pnl_usd": 25.0 if universe_eligible else None,
        "capture_ratio": 0.5 if universe_eligible else None,
        "exit_reason": "SESSION_END" if universe_eligible else None,
    }


def _benchmark(*, selected=False, universe_eligible=True):
    return {
        "replay_id": "test-replay",
        "trading_date": "2026-09-14",
        "universe_version": "research_universe_v1.0",
        **EVIDENCE,
        "execution_policy_version": "execution_policy_v1.0_hypothetical",
        "benchmark_candidates": [
            _benchmark_candidate(
                selected=selected, universe_eligible=universe_eligible
            )
        ],
        "return_baselines": {
            "eligible_ticker_mean_realized_return_pct": 1.0,
            "eligible_ticker_mean_realized_pnl_usd": 25.0,
            "random_draw_mean_realized_return_pct": 1.0,
            "random_draw_mean_realized_pnl_usd": 25.0,
        },
        "scout_summary": {
            "realized_return_pct": 0.0,
            "realized_pnl_usd": 0.0,
            "max_drawdown_pct": None,
            "win_rate_pct": None,
            "average_capture_ratio": None,
        },
        "comparison": {"result_code": "SCOUT_UNDERPERFORMED"},
    }


def _postmortem(
    *, candidate=None, eligible=True, bar_count=60, universe_eligible=True
):
    candidate = candidate or _candidate()
    scout = {
        "scout_version": "research_scout_alpha_v1.0",
        **EVIDENCE,
        "candidates": [candidate],
    }
    return build_postmortem(
        _snapshot(eligible=eligible, bar_count=bar_count),
        scout,
        _benchmark(
            selected=candidate["research_selected"],
            universe_eligible=universe_eligible,
        ),
    )


def test_not_in_universe_classification_includes_component_scores():
    result = _postmortem(eligible=False, universe_eligible=False)
    miss = result["missed_opportunities"][0]
    assert miss["miss_classification"] == "NOT_IN_UNIVERSE"
    assert miss["component_scores"][0] == {
        "metric_id": "premarket_gap_strength",
        "score": 1,
        "raw_value": 2.0,
    }


@pytest.mark.parametrize(
    ("bar_count", "relative_volume", "gap_pct"),
    [(59, 2.0, 2.0), (60, 0.9, 0.5)],
)
def test_invisible_at_freeze_classification(
    bar_count, relative_volume, gap_pct
):
    candidate = _candidate()
    candidate["component_scores"] = _component_scores(
        relative_volume, gap_pct
    )
    result = _postmortem(candidate=candidate, bar_count=bar_count)
    assert result["missed_opportunities"][0]["miss_classification"] == (
        "INVISIBLE_AT_FREEZE"
    )
    assert result["unreachable_mover_count"] == 1
    assert result["unreachable_pct"] == 100.0


def test_visible_scored_low_classification():
    result = _postmortem(candidate=_candidate(total_score=20))
    assert result["missed_opportunities"][0]["miss_classification"] == (
        "VISIBLE_SCORED_LOW"
    )


def test_visible_guardrail_reject_classification():
    result = _postmortem(
        candidate=_candidate(total_score=35, guardrail_action="REJECT")
    )
    assert result["missed_opportunities"][0]["miss_classification"] == (
        "VISIBLE_GUARDRAIL_REJECT"
    )


def test_picked_execution_loss_is_separate_from_scout_hypotheses():
    result = _postmortem(candidate=_candidate(selected=True, total_score=35))
    miss = result["missed_opportunities"][0]
    assert miss["miss_classification"] == "PICKED_EXECUTION_LOSS"
    assert "EXECUTION_CAPTURE_BELOW_50_PERCENT" in miss["failure_reason_codes"]
    assert "TRAILING_STOP_BEFORE_DAY_HIGH" in miss["failure_reason_codes"]
    assert result["execution_policy_review"][0]["ticker"] == "MOVE"
    assert result["feature_failures"] == []


def test_random_draw_verdict_is_deterministic_and_top_10_is_diagnostic_only(
    tmp_path,
):
    snapshot = {
        "replay_id": "baseline-replay",
        "trading_date": "2026-09-14",
        "universe_version": "research_universe_v1.0",
        **EVIDENCE,
        "securities": [
            {"ticker": ticker, "eligible": True}
            for ticker in ("AAA", "BBB", "CCC")
        ],
    }
    candidates = [
        {
            "ticker": ticker,
            "research_eligible": True,
            "research_selected": ticker == "AAA",
            "component_scores": {},
        }
        for ticker in ("AAA", "BBB", "CCC")
    ]
    scout = {
        "scout_version": "research_scout_alpha_v1.0",
        **EVIDENCE,
        "candidates": candidates,
    }
    outcomes = []
    paths = {
        "AAA": _bars(close=10.5, high=11.0, low=10.0),
        "BBB": _bars(close=10.0, high=10.2, low=10.0),
        "CCC": _bars(close=9.8, high=10.0, low=9.0),
    }
    for ticker, bars in paths.items():
        selected = ticker == "AAA"
        outcomes.append({
            "ticker": ticker,
            "selected": selected,
            "mfe_pct": {"AAA": 10.0, "BBB": 2.0, "CCC": 0.0}[ticker],
            "mae_pct": {"AAA": 0.0, "BBB": 0.0, "CCC": -10.0}[ticker],
            "maximum_capturable_move_pct": {
                "AAA": 10.0, "BBB": 2.0, "CCC": 0.0
            }[ticker],
            "intraday_path": bars,
            "execution_result": (
                {
                    "trade_executed": True,
                    "realized_return_pct": 5.0,
                    "realized_pnl_usd": 25.0,
                    "capture_ratio": 0.5,
                    "maximum_position_drawdown_pct": 0.0,
                    "exit_reason": "SESSION_END",
                    "exit_timestamp": bars[-1]["timestamp"],
                }
                if selected
                else {"trade_executed": False, "realized_return_pct": 0.0}
            ),
        })
    outcome_result = {
        "execution_policy_version": "execution_policy_v1.0_hypothetical",
        **EVIDENCE,
        "outcomes": outcomes,
    }

    first = build_same_universe_benchmark(
        snapshot, scout, outcome_result, 2500.0
    )
    second = build_same_universe_benchmark(
        snapshot, scout, deepcopy(outcome_result), 2500.0
    )

    assert first == second
    assert first["return_baselines"]["random_seed"] == RANDOM_BASELINE_SEED
    assert first["return_baselines"]["random_draw_count"] == RANDOM_BASELINE_DRAWS
    assert first["return_baselines"]["random_draw_size"] == 1
    assert first["comparison"]["top_10_capture_rate_pct"] == pytest.approx(
        100 / 3
    )
    assert first["comparison"]["result_code"] == "SCOUT_OUTPERFORMED"
    assert first["scout_summary"]["candidate_count"] == 1
    assert first["exploration_summary"]["candidate_count"] == 0
    assert first["combined_summary"] == first["scout_summary"]
    assert first["comparison"]["return_difference_pct"] == first[
        "comparison"
    ]["random_draw_mean_return_difference_pct"]
    postmortem = build_postmortem(snapshot, scout, first)
    report = tmp_path / "postmortem.pdf"
    generate_postmortem_pdf(first, postmortem, report)
    assert report.read_bytes().startswith(b"%PDF")


def _benchmark_inputs_for_exploration_top_k(exploration_top_k: int):
    tickers = ("QUAL", "EXP1", "EXP2", "EXP3")
    snapshot = {
        "replay_id": f"exploration-{exploration_top_k}",
        "trading_date": "2026-09-14",
        "universe_version": "research_universe_v1.0",
        **EVIDENCE,
        "securities": [
            {"ticker": ticker, "eligible": True} for ticker in tickers
        ],
    }
    scout = {
        "scout_version": "research_scout_alpha_v1.0",
        **EVIDENCE,
        "candidates": [
            {
                "ticker": ticker,
                "research_eligible": True,
                "research_selected": (
                    ticker == "QUAL"
                    or (
                        ticker.startswith("EXP")
                        and int(ticker[-1]) <= exploration_top_k
                    )
                ),
                "selection_basis": (
                    "QUALIFYING_THRESHOLD"
                    if ticker == "QUAL"
                    else (
                        "EXPLORATION_TOP_K"
                        if int(ticker[-1]) <= exploration_top_k
                        else "NOT_SELECTED"
                    )
                ),
                "component_scores": {},
            }
            for ticker in tickers
        ],
    }
    exploration_pnls = {"EXP1": 10.0, "EXP2": -5.0, "EXP3": 15.0}
    outcomes = []
    for ticker in tickers:
        selected = ticker == "QUAL" or (
            ticker.startswith("EXP")
            and int(ticker[-1]) <= exploration_top_k
        )
        basis = (
            "QUALIFYING_THRESHOLD"
            if ticker == "QUAL"
            else ("EXPLORATION_TOP_K" if selected else "NOT_SELECTED")
        )
        pnl = 25.0 if ticker == "QUAL" else exploration_pnls[ticker]
        bars = _bars(close=10.5, high=11.0, low=10.0)
        outcomes.append({
            "ticker": ticker,
            "selected": selected,
            "selection_basis": basis,
            "mfe_pct": 10.0,
            "mae_pct": 0.0,
            "maximum_capturable_move_pct": 10.0,
            "intraday_path": bars,
            "execution_result": (
                {
                    "trade_executed": True,
                    "realized_return_pct": pnl / 25.0 * 5.0,
                    "realized_pnl_usd": pnl,
                    "capture_ratio": 0.5,
                    "maximum_position_drawdown_pct": 0.0,
                    "exit_reason": "SESSION_END",
                    "exit_timestamp": bars[-1]["timestamp"],
                }
                if selected
                else {"trade_executed": False, "realized_return_pct": 0.0}
            ),
        })
    outcome_result = {
        "execution_policy_version": "execution_policy_v1.0_hypothetical",
        **EVIDENCE,
        "outcomes": outcomes,
    }
    return snapshot, scout, outcome_result


def test_changing_exploration_top_k_does_not_change_scout_realized_pnl():
    without_exploration = build_same_universe_benchmark(
        *_benchmark_inputs_for_exploration_top_k(0),
        2500.0,
    )
    with_three_exploration_names = build_same_universe_benchmark(
        *_benchmark_inputs_for_exploration_top_k(3),
        2500.0,
    )

    assert without_exploration["scout_summary"]["realized_pnl_usd"] == 25.0
    assert with_three_exploration_names["scout_summary"]["realized_pnl_usd"] == 25.0
    assert without_exploration["comparison"]["result_code"] == with_three_exploration_names["comparison"]["result_code"]
    assert without_exploration["return_baselines"]["random_draw_size"] == 1
    assert with_three_exploration_names["return_baselines"]["random_draw_size"] == 1
    assert without_exploration["exploration_summary"]["realized_pnl_usd"] == 0.0
    assert with_three_exploration_names["exploration_summary"]["realized_pnl_usd"] == 20.0
    assert with_three_exploration_names["combined_summary"]["realized_pnl_usd"] == 45.0


def test_qualifying_candidates_consume_execution_slots_before_exploration():
    snapshot = {
        "replay_id": "execution-priority",
        "trading_date": "2026-09-14",
        "scout_version": "research_scout_alpha_v1.0",
        "execution_policy_version": "execution_policy_v1.0_hypothetical",
        **EVIDENCE,
    }
    candidates = [
        {
            "ticker": f"EXP{index}",
            "research_selected": True,
            "selection_basis": "EXPLORATION_TOP_K",
            "rank": index,
            "score_pct": 50.0,
        }
        for index in range(1, 6)
    ]
    candidates.append({
        "ticker": "QUAL",
        "research_selected": True,
        "selection_basis": "QUALIFYING_THRESHOLD",
        "rank": 6,
        "score_pct": 80.0,
    })
    scout = {"candidates": candidates}
    bars = {
        candidate["ticker"]: _bars(close=10.5, high=11.0, low=10.0)
        for candidate in candidates
    }

    result = grade_replay_outcomes(snapshot, scout, bars, 2500.0)
    by_ticker = {item["ticker"]: item for item in result["outcomes"]}

    assert by_ticker["QUAL"]["execution_result"]["trade_executed"] is True
    assert by_ticker["EXP5"]["execution_result"]["trade_executed"] is False
    assert by_ticker["EXP5"]["execution_result"]["exit_reason"] == "ENTRY_REJECTED"
