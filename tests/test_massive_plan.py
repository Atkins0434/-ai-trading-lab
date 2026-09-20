from __future__ import annotations

import json
from pathlib import Path

from trainer.rate_control import (
    AdaptiveRateLimiter,
    load_massive_plan,
    load_reference_fetch_workers,
)


ROOT = Path(__file__).resolve().parent.parent
MASSIVE_WORKFLOWS = {
    "massive-smoke.yml": "massive-capability-${{ github.repository }}",
    "massive-universe-smoke.yml": "massive-universe-${{ github.repository }}",
    "research-alpha-batch.yml": "research-alpha-${{ github.repository }}",
    "multi-day-trainer.yml": "multi-day-trainer-${{ github.repository }}",
}


def test_active_massive_plan_matches_developer_contract():
    assert load_massive_plan() == {
        "plan": "DEVELOPER",
        "rest_calls_per_minute": None,
        "history_years": 10,
        "flat_files": True,
    }
    assert load_reference_fetch_workers() == 8


def test_finite_plan_config_preserves_request_pacing(tmp_path: Path):
    path = tmp_path / "massive_plan.json"
    path.write_text(json.dumps({
        "plan": "BASIC_FREE",
        "rest_calls_per_minute": 5,
        "history_years": 2,
        "flat_files": False,
    }))
    now = [0.0]
    slept = []

    def clock():
        return now[0]

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = AdaptiveRateLimiter(
        plan_path=path,
        clock=clock,
        sleep=sleep,
    )
    limiter.wait()
    limiter.wait()

    assert limiter.snapshot()["requests_per_minute"] == 5
    assert slept == [12.0]


def test_reference_worker_config_defaults_and_validates(tmp_path: Path):
    default_path = tmp_path / "default.json"
    default_path.write_text(json.dumps({
        "plan": "BASIC_FREE",
        "rest_calls_per_minute": 5,
        "history_years": 2,
        "flat_files": False,
    }))
    assert load_reference_fetch_workers(default_path) == 8

    configured_path = tmp_path / "configured.json"
    configured_path.write_text(json.dumps({
        "plan": "DEVELOPER",
        "rest_calls_per_minute": None,
        "reference_fetch_workers": 4,
        "history_years": 10,
        "flat_files": True,
    }))
    assert load_reference_fetch_workers(configured_path) == 4


def test_massive_workflows_use_distinct_non_overlapping_groups_without_sleeps():
    for filename, concurrency_group in MASSIVE_WORKFLOWS.items():
        workflow = (ROOT / ".github" / "workflows" / filename).read_text(
            encoding="utf-8"
        )
        assert f"group: {concurrency_group}" in workflow
        assert "cancel-in-progress: false" in workflow
        assert "sleep " not in workflow


def test_flatfile_workflow_persists_versioned_provider_caches():
    workflow = (
        ROOT / ".github" / "workflows" / "flat-file-replay-day.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count("uses: actions/cache@v4") == 2
    assert "jq -er .cache_version config/flatfile_replay.json" in workflow
    assert "path: data/flatfiles/" in workflow
    assert "path: |\n            data/reference_cache/\n            data/news_cache/" in workflow
    assert "restore-keys:" in workflow
    assert "10 GB per repository" in workflow
    assert "max_tickers:" in workflow
    assert "--max-tickers" in workflow
    assert "Run status:" in workflow
    assert "Current state:" in workflow
    assert "Last phase:" in workflow
    assert "Error:" in workflow
    assert "Eligible count:" in workflow
    assert "Definitive-exclusion count:" in workflow
    assert "Definitive exclusions by reason:" in workflow
    assert "Coverage-gap count:" in workflow
    assert "Coverage gaps by reason:" in workflow
    assert "Provider limitations:" in workflow
    assert "provider_identifier_gap_count" in workflow
    assert "definitive_misses=" in workflow
    assert "transient_retries=" in workflow
    assert "unresolved_failures=" in workflow
    assert "listed_after_lagged_date=" in workflow
    assert "python -m trainer.flatfile_inspect" in workflow
    assert "--ticker AAL" in workflow
    assert "python -m trainer.replay_report --output-root" in workflow
    assert 'python -m trainer.output_paths "$OUTPUT_ROOT" "$TRADING_DATE" replay_report' in workflow
    assert '"$OUTPUT_ROOT/replay_summary.pdf"' in workflow
    assert (
        "${{ steps.replay.outputs.output_root }}/\n" in workflow
    )
    assert "benchmark_result.json\n            " not in workflow.split(
        "uses: actions/upload-artifact@v4"
    )[-1]


def test_flatfile_range_workflow_resumes_and_verifies_multi_day_output():
    workflow = (
        ROOT / ".github" / "workflows" / "flat-file-replay-range.yml"
    ).read_text(encoding="utf-8")

    assert "name: Flat-File Replay Range" in workflow
    assert "group: multi-day-trainer-${{ github.repository }}" in workflow
    assert "timeout-minutes: 360" in workflow
    assert "actions/cache/restore@v4" in workflow
    assert "actions/cache/save@v4" in workflow
    assert "replay-output-${{ github.repository }}" in workflow
    assert "--max-wall-seconds 18000" in workflow
    assert "python -m trainer.flatfile_coverage" in workflow
    assert "store.ensure_day(DAY_AGGS_DATASET, trading_date)" in workflow
    assert "store.ensure_day(MINUTE_AGGS_DATASET, trading_date)" in workflow
    assert "python -m trainer.flatfile_range_summary" in workflow
    assert 'if day.get("status") != "COMPLETE"' in workflow
    assert '"PAUSED_WALL_BUDGET"' in workflow
    assert "Each run saves a new flat-file cache entry." in workflow
    assert 'data/reference_cache data/news_cache "$OUTPUT_ROOT"' in workflow
    assert "flat-file-replay-range-${{ steps.replay.outputs.start_date }}" in workflow
