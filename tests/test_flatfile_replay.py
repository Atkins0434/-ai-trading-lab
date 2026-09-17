from __future__ import annotations

import json
from pathlib import Path

import pytest

import trainer.flatfile_replay as flatfile_replay
from trainer.flatfile_replay import (
    REPLAY_PHASES,
    _new_day_record,
    _timed_phase,
    run_flatfile_replay,
)


def test_flatfile_replay_resumes_completed_dates_without_reprocessing(tmp_path: Path):
    calls = []

    def day_runner(
        reference_client,
        flatfiles,
        trading_date,
        *,
        output_root,
        **kwargs,
    ):
        calls.append(trading_date)
        marker = output_root / "days" / trading_date / "complete.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"date": trading_date}) + "\n")
        return {
            "trading_date": trading_date,
            "status": "COMPLETE",
            "universe_size": 2,
            "universe_manifest_hash": "sha256:" + "a" * 64,
            "scored_ticker_count": 2,
            "padded_bar_statistics": {
                "premarket_padded_bar_count": 0,
                "regular_padded_bar_count": 0,
            },
            "files": [],
            "reference_cache": {
                "hits": 2,
                "fetches": 1,
                "quarter_reuse_hits": 1,
                "errors": 0,
                "definitive_misses": 0,
                "transient_retries": 0,
                "unresolved_failures": 0,
                "listed_after_lagged_date": 0,
            },
            "artifacts": {
                "marker": str(marker.relative_to(output_root)),
            },
            "error": None,
        }

    dates = ["2018-01-02", "2018-01-03"]
    first = run_flatfile_replay(
        object(),
        object(),
        dates,
        output_root=tmp_path,
        lookback_sessions=1,
        day_runner=day_runner,
    )
    second = run_flatfile_replay(
        object(),
        object(),
        dates,
        output_root=tmp_path,
        lookback_sessions=1,
        day_runner=day_runner,
    )

    assert first["status"] == "COMPLETE"
    assert second["status"] == "COMPLETE"
    assert second["completed_dates"] == dates
    assert second["reference_cache_summary"] == {
        "hits": 4,
        "fetches": 2,
        "quarter_reuse_hits": 2,
        "errors": 0,
        "definitive_misses": 0,
        "transient_retries": 0,
        "unresolved_failures": 0,
        "listed_after_lagged_date": 0,
    }
    assert calls == dates


def test_resume_upgrades_pre_failure_classification_cache_metrics(
    tmp_path: Path,
):
    artifact = tmp_path / "days" / "2018-01-03" / "complete.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    day = _new_day_record(
        "2018-01-03",
        smoke_mode=False,
        max_tickers=None,
    )
    day.update({
        "status": "COMPLETE",
        "current_phase": None,
        "research_evidence": True,
        "artifacts": {"marker": str(artifact.relative_to(tmp_path))},
    })
    day["phase_status"] = {phase: "COMPLETE" for phase in REPLAY_PHASES}
    day["reference_cache"] = {
        "hits": 1,
        "fetches": 2,
        "quarter_reuse_hits": 0,
        "errors": 0,
    }
    initial = run_flatfile_replay(
        object(),
        object(),
        ["2018-01-03"],
        output_root=tmp_path / "seed",
        lookback_sessions=1,
        day_runner=lambda *args, **kwargs: day,
    )
    # Repoint the artifact into the seed output so the completed day resumes.
    seed_artifact = tmp_path / "seed" / "complete.json"
    seed_artifact.write_text("{}\n")
    initial["days"][0]["artifacts"] = {"marker": "complete.json"}
    initial["days"][0]["reference_cache"] = day["reference_cache"]
    (tmp_path / "seed" / "flatfile_replay_manifest.json").write_text(
        json.dumps(initial) + "\n"
    )

    resumed = run_flatfile_replay(
        object(),
        object(),
        ["2018-01-03"],
        output_root=tmp_path / "seed",
        lookback_sessions=1,
        day_runner=lambda *args, **kwargs: pytest.fail(
            "completed date should resume"
        ),
    )

    assert resumed["days"][0]["reference_cache"] == {
        "hits": 1,
        "fetches": 2,
        "quarter_reuse_hits": 0,
        "errors": 0,
        "definitive_misses": 0,
        "transient_retries": 0,
        "unresolved_failures": 0,
        "listed_after_lagged_date": 0,
    }


def test_phase_timing_logs_only_to_stderr(capsys):
    with _timed_phase("2018-01-03", "snapshot"):
        pass

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "trading_date=2018-01-03 phase=snapshot" in captured.err
    assert "elapsed_seconds=" in captured.err


def test_run_manifest_checkpoints_each_phase_while_in_progress(tmp_path: Path):
    observed = []

    def day_runner(
        reference_client,
        flatfiles,
        trading_date,
        *,
        output_root,
        progress_callback,
        max_tickers,
        **kwargs,
    ):
        record = _new_day_record(
            trading_date,
            smoke_mode=False,
            max_tickers=max_tickers,
        )
        for phase in REPLAY_PHASES:
            record["current_phase"] = phase
            record["phase_status"][phase] = "IN_PROGRESS"
            progress_callback(record)
            manifest = json.loads(
                (output_root / "flatfile_replay_manifest.json").read_text()
            )
            observed.append((manifest["status"], manifest["current_phase"]))
            record["phase_status"][phase] = "COMPLETE"
            progress_callback(record)
        record["status"] = "COMPLETE"
        record["current_phase"] = None
        return record

    result = run_flatfile_replay(
        object(),
        object(),
        ["2018-01-03"],
        output_root=tmp_path,
        lookback_sessions=1,
        day_runner=day_runner,
    )

    assert observed == [("IN_PROGRESS", phase) for phase in REPLAY_PHASES]
    assert result["status"] == "COMPLETE"
    assert result["current_phase"] is None


def test_interrupted_run_leaves_readable_in_progress_phase(tmp_path: Path):
    def interrupted_runner(
        reference_client,
        flatfiles,
        trading_date,
        *,
        progress_callback,
        **kwargs,
    ):
        record = _new_day_record(
            trading_date,
            smoke_mode=False,
            max_tickers=None,
        )
        record["current_phase"] = "universe"
        record["phase_status"]["universe"] = "IN_PROGRESS"
        progress_callback(record)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_flatfile_replay(
            object(),
            object(),
            ["2018-01-03"],
            output_root=tmp_path,
            lookback_sessions=1,
            day_runner=interrupted_runner,
        )

    manifest = json.loads(
        (tmp_path / "flatfile_replay_manifest.json").read_text()
    )
    assert manifest["status"] == "IN_PROGRESS"
    assert manifest["current_trading_date"] == "2018-01-03"
    assert manifest["current_phase"] == "universe"
    assert manifest["days"][0]["status"] == "IN_PROGRESS"


def test_smoke_cap_is_recorded_and_never_research_evidence(tmp_path: Path):
    observed_caps = []

    def smoke_runner(
        reference_client,
        flatfiles,
        trading_date,
        *,
        max_tickers,
        **kwargs,
    ):
        observed_caps.append(max_tickers)
        record = _new_day_record(
            trading_date,
            smoke_mode=True,
            max_tickers=max_tickers,
        )
        record["status"] = "COMPLETE"
        record["phase_status"] = {
            phase: "SKIPPED" if phase == "postmortem" else "COMPLETE"
            for phase in REPLAY_PHASES
        }
        return record

    result = run_flatfile_replay(
        object(),
        object(),
        ["2018-01-03"],
        output_root=tmp_path,
        lookback_sessions=1,
        max_tickers=3,
        day_runner=smoke_runner,
    )

    assert observed_caps == [3]
    assert result["smoke_mode"] is True
    assert result["max_tickers"] == 3
    assert result["days"][0]["research_evidence"] is False


def test_smoke_day_grades_benchmark_but_skips_trainer_postmortem(
    tmp_path: Path,
    monkeypatch,
):
    class SnapshotResult:
        snapshot = {"execution_policy_version": "execution_disabled"}
        outcome_bars = {}
        padded_bar_statistics = {"premarket_padded_bar_count": 0}

    monkeypatch.setattr(
        flatfile_replay,
        "_ensure_day_files",
        lambda *args, **kwargs: ([], []),
    )
    monkeypatch.setattr(
        flatfile_replay,
        "build_point_in_time_universe",
        lambda *args, **kwargs: {
            "coverage_status": "complete",
            "coverage_reasons": [],
            "eligible_symbol_count": 1,
            "manifest_hash": "sha256:" + "a" * 64,
            "research_evidence": False,
            "source": {
                "query_parameters": {"ticker_overview_cache": {}}
            },
        },
    )
    monkeypatch.setattr(
        flatfile_replay,
        "build_flatfile_snapshot",
        lambda *args, **kwargs: SnapshotResult(),
    )
    monkeypatch.setattr(
        flatfile_replay,
        "run_research_scout_alpha",
        lambda *args, **kwargs: {"candidates": []},
    )
    monkeypatch.setattr(
        flatfile_replay,
        "grade_replay_outcomes",
        lambda *args, **kwargs: {"outcomes": []},
    )
    monkeypatch.setattr(
        flatfile_replay,
        "build_same_universe_benchmark",
        lambda *args, **kwargs: {"research_evidence": False},
    )
    monkeypatch.setattr(
        flatfile_replay,
        "build_postmortem",
        lambda *args, **kwargs: pytest.fail(
            "Smoke evidence must never enter the Trainer postmortem."
        ),
    )
    monkeypatch.setattr(
        flatfile_replay,
        "generate_scout_pdf_report",
        lambda *args, **kwargs: None,
    )

    result = flatfile_replay.run_flatfile_day(
        object(),
        object(),
        "2018-01-03",
        output_root=tmp_path,
        lookback_sessions=1,
        strategy_capital=2500.0,
        threshold_pct=None,
        exploration_top_k=0,
        max_tickers=3,
    )

    assert result["status"] == "COMPLETE"
    assert result["research_evidence"] is False
    assert result["phase_status"]["benchmark"] == "COMPLETE"
    assert result["phase_status"]["postmortem"] == "SKIPPED"
    assert "benchmark_result" in result["artifacts"]
    assert "postmortem" not in result["artifacts"]


def test_failure_between_phases_preserves_last_phase_and_error(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        flatfile_replay,
        "_ensure_day_files",
        lambda *args, **kwargs: ([], []),
    )
    monkeypatch.setattr(
        flatfile_replay,
        "build_point_in_time_universe",
        lambda *args, **kwargs: {
            # Deliberately omit coverage_status. The universe phase completes,
            # then the between-phase eligibility check raises KeyError.
            "coverage_reasons": [],
            "eligible_symbol_count": 1,
            "manifest_hash": "sha256:" + "a" * 64,
            "research_evidence": True,
            "source": {
                "query_parameters": {"ticker_overview_cache": {}}
            },
        },
    )

    result = run_flatfile_replay(
        object(),
        object(),
        ["2018-01-03"],
        output_root=tmp_path,
        lookback_sessions=1,
    )
    persisted = json.loads(
        (tmp_path / "flatfile_replay_manifest.json").read_text()
    )

    for manifest in (result, persisted):
        assert manifest["status"] == "FAILED"
        assert manifest["current_trading_date"] == "2018-01-03"
        assert manifest["current_phase"] == "universe"
        assert manifest["days"][0]["status"] == "FAILED"
        assert manifest["days"][0]["current_phase"] == "universe"
        assert manifest["days"][0]["phase_status"]["universe"] == "COMPLETE"
        assert "KeyError" in manifest["days"][0]["error"]
