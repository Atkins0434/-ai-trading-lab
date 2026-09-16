from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from trainer.multi_day_trainer import run_multi_day_trainer


def fake_day_runner(
    client,
    tickers,
    trading_date,
    *,
    cache_root,
    output_dir,
    threshold_pct,
    exploration_top_k,
    dataset_partition,
    universe_mode,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "universe_mode": "historical_research",
        "universe_manifest_hash": "sha256:" + trading_date.replace("-", "").ljust(64, "0"),
        "universe_coverage": "complete",
        "research_evidence": True,
        "promotion_eligible": True,
    }
    benchmark = {
        **evidence,
        "scout_summary": {"realized_return_pct": 1.0, "realized_pnl_usd": 25.0},
        "exploration_summary": {
            "realized_return_pct": -0.2,
            "realized_pnl_usd": -5.0,
        },
        "combined_summary": {
            "realized_return_pct": 0.8,
            "realized_pnl_usd": 20.0,
        },
        "benchmark_summary": {"realized_return_pct": 2.0, "realized_pnl_usd": 50.0},
        "return_baselines": {
            "eligible_single_position_mean_return_pct": 0.4,
            "eligible_ticker_mean_realized_pnl_usd": 10.0,
            "eligible_basket_expected_return_pct": 0.4,
            "eligible_basket_expected_pnl_usd": 10.0,
            "random_draw_mean_realized_return_pct": 2.0,
            "random_draw_mean_realized_pnl_usd": 50.0,
        },
        "benchmark_candidates": [
            {"ticker": "MISS", "benchmark_rank": 1},
            {"ticker": "INVISIBLE", "benchmark_rank": 2},
            {"ticker": "EXEC", "benchmark_rank": 3},
        ],
        "comparison": {"top_10_capture_rate_pct": 40.0},
    }
    postmortem = {
        **evidence,
        "trading_date": trading_date,
        "result": "MISS",
        "unreachable_mover_count": 1,
        "unreachable_pct": 100 / 3,
        "missed_opportunities": [
            {
                "ticker": "MISS",
                "benchmark_rank": 1,
                "miss_classification": "VISIBLE_SCORED_LOW",
                "failure_reason_codes": ["BELOW_RESEARCH_THRESHOLD"],
                "component_scores": [
                    {"metric_id": "relative_volume", "score": 1, "raw_value": float(trading_date[-2:])},
                    {"metric_id": "price_slope_15m", "score": 0, "raw_value": -0.2},
                    {"metric_id": "volume_slope_15m", "score": 1, "raw_value": 0.3},
                    {"metric_id": "volume_slope_30m", "score": 0, "raw_value": -0.4},
                    {"metric_id": "premarket_gap_strength", "score": 2, "raw_value": 1.5},
                ],
            },
            {
                "ticker": "INVISIBLE",
                "benchmark_rank": 2,
                "miss_classification": "INVISIBLE_AT_FREEZE",
                "failure_reason_codes": ["AT_LEAST_30_PADDED_PREMARKET_BARS"],
                "component_scores": [],
            },
            {
                "ticker": "EXEC",
                "benchmark_rank": 3,
                "miss_classification": "PICKED_EXECUTION_LOSS",
                "failure_reason_codes": ["EXECUTION_CAPTURE_BELOW_50_PERCENT"],
                "component_scores": [],
            },
        ],
        "execution_policy_review": [{
            "ticker": "EXEC",
            "benchmark_rank": 3,
            "realized_return_pct": 1.0,
            "maximum_capturable_move_pct": 10.0,
            "exit_reason": "TRAILING_STOP",
            "reason_codes": ["EXECUTION_CAPTURE_BELOW_50_PERCENT"],
        }],
    }
    (output_dir / "benchmark_result.json").write_text(json.dumps(benchmark))
    (output_dir / "postmortem.json").write_text(json.dumps(postmortem))
    (output_dir / "catalyst_shadow_metrics.json").write_text(json.dumps({
        "ticker_metrics": [{
            "ticker": "MISS",
            "components": [{
                "metric": "catalyst_quality",
                "status": "OBSERVED",
                "points": 2,
                "included_in_alpha_score": False,
            }],
        }],
    }))
    return {"status": "COMPLETE", "scored_tickers": ["MISS"], "skipped": {}, **evidence}


def test_multi_day_trainer_persists_evidence_and_never_promotes(tmp_path: Path):
    dates = ["2026-09-10", "2026-09-11", "2026-09-14"]
    state = run_multi_day_trainer(object(), None, dates, cache_root=tmp_path / "cache", output_root=tmp_path / "reports" / "trainer", day_runner=fake_day_runner, universe_mode="historical_research")

    assert state["status"] == "COMPLETE"
    assert state["massive_plan"]["plan"] == "DEVELOPER"
    assert all(
        day["massive_plan"] == state["massive_plan"]
        for day in state["days"]
    )
    assert state["completed_dates"] == dates
    assert state["aggregate_performance"]["days_processed"] == 3
    assert state["aggregate_performance"]["misses"] == 3
    assert state["hypotheses"][0]["independent_occurrence_count"] == 3
    assert len(state["hypotheses"]) == 1
    assert all(
        item["miss_classification"] == "VISIBLE_SCORED_LOW"
        for item in state["hypotheses"][0]["evidence"]
    )
    assert state["hypotheses"][0]["status"] == "COLLECTING_EVIDENCE"
    assert state["controls"]["production_mutation_allowed"] is False
    assert state["controls"]["automatic_promotion_allowed"] is False
    assert state["selection_policy"]["exploration_top_k"] == 0
    assert state["aggregate_performance"]["scout_cumulative_return_pct"] == 3.0301
    assert state["aggregate_performance"]["exploration_cumulative_return_pct"] == pytest.approx(-0.598801)
    assert state["aggregate_performance"]["combined_cumulative_return_pct"] == pytest.approx(2.419251)
    assert state["aggregate_performance"]["scout_total_realized_pnl_usd"] == 75.0
    assert state["aggregate_performance"]["exploration_total_realized_pnl_usd"] == -15.0
    assert state["aggregate_performance"]["combined_total_realized_pnl_usd"] == 60.0
    assert state["aggregate_performance"]["realized_pnl_capture_pct"] == 50.0
    assert state["aggregate_performance"]["scout_max_drawdown_pct"] == 0.0
    assert state["aggregate_performance"]["unreachable_mover_count"] == 3
    assert state["aggregate_performance"]["unreachable_pct"] == pytest.approx(
        100 / 3, abs=1e-6
    )
    assert state["aggregate_performance"]["execution_policy_review_count"] == 3
    assert len(state["execution_policy_review"]) == 3
    assert [
        item["metric_id"] for item in state["candidate_threshold_reviews"]
    ] == ["price_slope_15m", "relative_volume", "volume_slope_15m"]
    relative_review = next(
        item
        for item in state["candidate_threshold_reviews"]
        if item["metric_id"] == "relative_volume"
    )
    assert relative_review["observed_raw_value_min"] == 10.0
    assert relative_review["observed_raw_value_max"] == 14.0
    assert all(item["unreachable_mover_count"] == 1 for item in state["days"])
    assert state["feature_evidence"][0]["independent_occurrence_count"] == 3
    assert state["feature_evidence"][0]["average_points"] == 2.0
    assert state["queue"]["counts"]["COMPLETE"] == 3
    assert (tmp_path / "reports" / "trainer" / "trainer_summary_report.pdf").read_bytes().startswith(b"%PDF")


def test_only_visible_score_and_guardrail_misses_generate_hypotheses(
    tmp_path: Path,
):
    def mixed_miss_runner(*args, **kwargs):
        manifest = fake_day_runner(*args, **kwargs)
        path = kwargs["output_dir"] / "postmortem.json"
        postmortem = json.loads(path.read_text())
        postmortem["missed_opportunities"].append({
            "ticker": "GUARD",
            "benchmark_rank": 4,
            "miss_classification": "VISIBLE_GUARDRAIL_REJECT",
            "failure_reason_codes": ["LOW_AGGREGATE_LIQUIDITY"],
            "component_scores": [],
        })
        postmortem["missed_opportunities"].append({
            "ticker": "UNKNOWN",
            "benchmark_rank": 5,
            "miss_classification": "UNCLASSIFIED",
            "failure_reason_codes": ["NO_KNOWN_REJECTION_PATH"],
            "component_scores": [],
        })
        path.write_text(json.dumps(postmortem))
        benchmark_path = kwargs["output_dir"] / "benchmark_result.json"
        benchmark = json.loads(benchmark_path.read_text())
        benchmark["benchmark_candidates"].append({
            "ticker": "GUARD", "benchmark_rank": 4
        })
        benchmark["benchmark_candidates"].append({
            "ticker": "UNKNOWN", "benchmark_rank": 5
        })
        benchmark_path.write_text(json.dumps(benchmark))
        return manifest

    state = run_multi_day_trainer(
        object(),
        None,
        ["2026-09-14"],
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "trainer",
        day_runner=mixed_miss_runner,
        universe_mode="historical_research",
    )

    classifications = {
        evidence["miss_classification"]
        for hypothesis in state["hypotheses"]
        for evidence in hypothesis["evidence"]
    }
    assert classifications == {
        "VISIBLE_SCORED_LOW",
        "VISIBLE_GUARDRAIL_REJECT",
    }
    assert len(state["hypotheses"]) == 2


def test_multi_day_trainer_resumes_completed_days(tmp_path: Path):
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append(args[2])
        return fake_day_runner(*args, **kwargs)

    common = dict(client=object(), tickers=None, trading_dates=["2026-09-14"], cache_root=tmp_path / "cache", output_root=tmp_path / "trainer", day_runner=recording_runner, universe_mode="historical_research")
    run_multi_day_trainer(**common)
    run_multi_day_trainer(**common)
    assert calls == ["2026-09-14"]


def test_multi_day_trainer_reopens_partial_day_for_ticker_resume(tmp_path: Path):
    calls = []

    def partial_then_complete(*args, **kwargs):
        calls.append(args[2])
        manifest = fake_day_runner(*args, **kwargs)
        manifest["status"] = "PARTIAL" if len(calls) == 1 else "COMPLETE"
        return manifest

    common = dict(
        client=object(),
        tickers=None,
        trading_dates=["2026-09-14"],
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "trainer",
        day_runner=partial_then_complete,
        universe_mode="historical_research",
    )
    first = run_multi_day_trainer(**common)
    second = run_multi_day_trainer(**common)

    assert first["status"] == "PARTIAL"
    assert first["queue"]["counts"]["PENDING"] == 1
    assert second["status"] == "COMPLETE"
    assert second["queue"]["counts"]["COMPLETE"] == 1
    assert calls == ["2026-09-14", "2026-09-14"]


def test_multi_day_trainer_carries_prior_dates_into_next_run(tmp_path: Path):
    root = tmp_path / "trainer"
    run_multi_day_trainer(object(), None, ["2026-09-11"], cache_root=tmp_path / "cache", output_root=root, day_runner=fake_day_runner, universe_mode="historical_research")
    state = run_multi_day_trainer(object(), None, ["2026-09-14"], cache_root=tmp_path / "cache", output_root=root, day_runner=fake_day_runner, universe_mode="historical_research")

    assert state["requested_dates"] == ["2026-09-11", "2026-09-14"]
    assert state["completed_dates"] == ["2026-09-11", "2026-09-14"]
    assert state["hypotheses"][0]["independent_occurrence_count"] == 2


def test_selection_policy_change_invalidates_prior_day_results(tmp_path: Path):
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append((args[2], kwargs["exploration_top_k"]))
        return fake_day_runner(*args, **kwargs)

    root = tmp_path / "trainer"
    common = dict(client=object(), tickers=None, trading_dates=["2026-09-14"], cache_root=tmp_path / "cache", output_root=root, day_runner=recording_runner, universe_mode="historical_research")
    run_multi_day_trainer(**common, exploration_top_k=0)
    run_multi_day_trainer(**common, exploration_top_k=3)

    assert calls == [("2026-09-14", 0), ("2026-09-14", 3)]
    registry = json.loads(
        (root / "LEGACY_RESEARCH_QUARANTINE.json").read_text()
    )
    assert registry["version"] == "legacy_result_quarantine_v1.0"


def test_holdout_dates_are_generated_but_not_executed_without_unlock(tmp_path: Path):
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append(args[2])
        return fake_day_runner(*args, **kwargs)

    dates = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14"]
    state = run_multi_day_trainer(
        object(), None, dates,
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "trainer",
        day_runner=recording_runner,
        max_workers=2,
        universe_mode="historical_research",
    )
    assert state["dataset_policy"]["locked_holdout_dates"] == ["2026-09-14"]
    assert "2026-09-14" not in calls
    assert len(calls) == 4
    # The four runnable sessions split into three development dates and one
    # validation date. Validation measures but never trains the hypothesis.
    assert state["hypotheses"][0]["independent_occurrence_count"] == 3
    assert state["feature_evidence"][0]["independent_occurrence_count"] == 3
    assert state["feature_evidence"][0]["validation_occurrence_count"] == 1
    assert state["feature_evidence"][0]["holdout_occurrence_count"] == 0
    assert all(
        item["partition"] == "DEVELOPMENT"
        for item in state["hypotheses"][0]["evidence"]
    )


def test_bounded_workers_execute_independent_dates_concurrently(tmp_path: Path):
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def concurrent_runner(*args, **kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        try:
            return fake_day_runner(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    state = run_multi_day_trainer(
        object(), None,
        ["2026-09-10", "2026-09-11", "2026-09-14"],
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "trainer",
        day_runner=concurrent_runner,
        max_workers=2,
        universe_mode="historical_research",
    )
    assert maximum_active == 2
    assert state["accelerator"]["max_workers"] == 2


def test_frozen_dataset_partitions_cannot_shift_on_resume(tmp_path: Path):
    root = tmp_path / "trainer"
    first_dates = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14"]
    run_multi_day_trainer(
        object(), None, first_dates,
        cache_root=tmp_path / "cache",
        output_root=root,
        day_runner=fake_day_runner,
        universe_mode="historical_research",
    )
    with pytest.raises(ValueError, match="frozen development"):
        run_multi_day_trainer(
            object(), None, ["2026-09-15"],
            cache_root=tmp_path / "cache",
            output_root=root,
            day_runner=fake_day_runner,
            universe_mode="historical_research",
        )
