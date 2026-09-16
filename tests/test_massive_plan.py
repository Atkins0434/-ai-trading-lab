from __future__ import annotations

import json
from pathlib import Path

from trainer.rate_control import AdaptiveRateLimiter, load_massive_plan


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


def test_massive_workflows_use_distinct_non_overlapping_groups_without_sleeps():
    for filename, concurrency_group in MASSIVE_WORKFLOWS.items():
        workflow = (ROOT / ".github" / "workflows" / filename).read_text(
            encoding="utf-8"
        )
        assert f"group: {concurrency_group}" in workflow
        assert "cancel-in-progress: false" in workflow
        assert "sleep " not in workflow
