from __future__ import annotations

import json
from pathlib import Path
import threading
import time

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
        "benchmark_summary": {"realized_return_pct": 2.0, "realized_pnl_usd": 50.0},
        "comparison": {"top_10_capture_rate_pct": 40.0},
    }
    postmortem = {
        **evidence,
        "trading_date": trading_date,
        "result": "MISS",
        "missed_opportunities": [{"ticker": "MISS", "benchmark_rank": 1, "failure_stage": "BELOW_SELECTION_THRESHOLD", "failure_reason_codes": ["BELOW_RESEARCH_THRESHOLD"]}],
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
    assert state["completed_dates"] == dates
    assert state["aggregate_performance"]["days_processed"] == 3
    assert state["aggregate_performance"]["misses"] == 3
    assert state["hypotheses"][0]["independent_occurrence_count"] == 3
    assert state["hypotheses"][0]["status"] == "COLLECTING_EVIDENCE"
    assert state["controls"]["production_mutation_allowed"] is False
    assert state["controls"]["automatic_promotion_allowed"] is False
    assert state["selection_policy"]["exploration_top_k"] == 0
    assert state["aggregate_performance"]["scout_cumulative_return_pct"] == 3.0301
    assert state["aggregate_performance"]["realized_pnl_capture_pct"] == 50.0
    assert state["aggregate_performance"]["scout_max_drawdown_pct"] == 0.0
    assert state["feature_evidence"][0]["independent_occurrence_count"] == 3
    assert state["feature_evidence"][0]["average_points"] == 2.0
    assert state["queue"]["counts"]["COMPLETE"] == 3
    assert (tmp_path / "reports" / "trainer" / "trainer_summary_report.pdf").read_bytes().startswith(b"%PDF")


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
    import pytest

    with pytest.raises(ValueError, match="frozen development"):
        run_multi_day_trainer(
            object(), None, ["2026-09-15"],
            cache_root=tmp_path / "cache",
            output_root=root,
            day_runner=fake_day_runner,
            universe_mode="historical_research",
        )
