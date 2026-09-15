from __future__ import annotations

import json
from pathlib import Path

from trainer.multi_day_trainer import run_multi_day_trainer


def fake_day_runner(client, tickers, trading_date, *, cache_root, output_dir, threshold_pct):
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = {
        "scout_summary": {"realized_return_pct": 1.0, "realized_pnl_usd": 25.0},
        "benchmark_summary": {"realized_return_pct": 2.0, "realized_pnl_usd": 50.0},
        "comparison": {"top_10_capture_rate_pct": 40.0},
    }
    postmortem = {
        "trading_date": trading_date,
        "result": "MISS",
        "missed_opportunities": [{"ticker": "MISS", "benchmark_rank": 1, "failure_stage": "BELOW_SELECTION_THRESHOLD", "failure_reason_codes": ["BELOW_RESEARCH_THRESHOLD"]}],
    }
    (output_dir / "benchmark_result.json").write_text(json.dumps(benchmark))
    (output_dir / "postmortem.json").write_text(json.dumps(postmortem))
    return {"status": "COMPLETE", "scored_tickers": ["MISS"], "skipped": {}}


def test_multi_day_trainer_persists_evidence_and_never_promotes(tmp_path: Path):
    dates = ["2026-09-10", "2026-09-11", "2026-09-14"]
    state = run_multi_day_trainer(object(), ["MISS"], dates, cache_root=tmp_path / "cache", output_root=tmp_path / "reports" / "trainer", day_runner=fake_day_runner)

    assert state["status"] == "COMPLETE"
    assert state["completed_dates"] == dates
    assert state["aggregate_performance"]["days_processed"] == 3
    assert state["aggregate_performance"]["misses"] == 3
    assert state["hypotheses"][0]["independent_occurrence_count"] == 3
    assert state["hypotheses"][0]["status"] == "COLLECTING_EVIDENCE"
    assert state["controls"]["production_mutation_allowed"] is False
    assert state["controls"]["automatic_promotion_allowed"] is False
    assert (tmp_path / "reports" / "trainer" / "trainer_summary_report.pdf").read_bytes().startswith(b"%PDF")


def test_multi_day_trainer_resumes_completed_days(tmp_path: Path):
    calls = []

    def recording_runner(*args, **kwargs):
        calls.append(args[2])
        return fake_day_runner(*args, **kwargs)

    common = dict(client=object(), tickers=["MISS"], trading_dates=["2026-09-14"], cache_root=tmp_path / "cache", output_root=tmp_path / "trainer", day_runner=recording_runner)
    run_multi_day_trainer(**common)
    run_multi_day_trainer(**common)
    assert calls == ["2026-09-14"]


def test_multi_day_trainer_carries_prior_dates_into_next_run(tmp_path: Path):
    root = tmp_path / "trainer"
    run_multi_day_trainer(object(), ["MISS"], ["2026-09-11"], cache_root=tmp_path / "cache", output_root=root, day_runner=fake_day_runner)
    state = run_multi_day_trainer(object(), ["MISS"], ["2026-09-14"], cache_root=tmp_path / "cache", output_root=root, day_runner=fake_day_runner)

    assert state["requested_dates"] == ["2026-09-11", "2026-09-14"]
    assert state["completed_dates"] == ["2026-09-11", "2026-09-14"]
    assert state["hypotheses"][0]["independent_occurrence_count"] == 2
