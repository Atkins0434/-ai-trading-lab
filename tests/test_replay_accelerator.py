from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainer.dataset_splits import build_chronological_splits, enforce_partition_access
from trainer.rate_control import AdaptiveRateLimiter, RetryPolicy
from trainer.replay_queue import ReplayQueue
from trainer.trading_calendar import generate_trading_dates


def test_trading_calendar_excludes_weekends_and_exchange_holidays():
    assert generate_trading_dates("2026-07-02", "2026-07-07") == [
        "2026-07-02",
        "2026-07-06",
        "2026-07-07",
    ]


def test_trading_calendar_honors_extra_closures_and_sessions():
    assert generate_trading_dates(
        "2026-09-12",
        "2026-09-15",
        extra_closures=["2026-09-14"],
        extra_sessions=["2026-09-12"],
    ) == ["2026-09-12", "2026-09-15"]


def test_chronological_split_locks_latest_holdout():
    dates = [f"2026-09-{day:02d}" for day in range(1, 11)]
    split = build_chronological_splits(dates)
    assert split[dates[0]] == "DEVELOPMENT"
    assert split[dates[-1]] == "HOLDOUT"
    with pytest.raises(PermissionError, match="Holdout sessions are locked"):
        enforce_partition_access(split, ["HOLDOUT"])
    assert enforce_partition_access(
        split, ["HOLDOUT"], holdout_unlocked=True
    ) == [dates[-2], dates[-1]]


def test_small_probe_remains_development_only():
    split = build_chronological_splits(["2026-09-10", "2026-09-11", "2026-09-14"])
    assert set(split.values()) == {"DEVELOPMENT"}


def test_queue_resumes_interrupted_date_and_ticker_tasks(tmp_path: Path):
    path = tmp_path / "queue.json"
    queue = ReplayQueue(path, max_attempts=2)
    queue.enqueue([
        ("2026-09-10", None, "DAY_REPLAY", "DEVELOPMENT"),
        ("2026-09-10", "JOBY", "NEWS_BACKFILL", "DEVELOPMENT"),
    ])
    task = queue.pending()[0]
    queue.claim(task["task_id"])

    resumed = ReplayQueue(path, max_attempts=2)
    recovered = resumed.pending()[0]
    assert recovered["last_error"] == "RECOVERED_INTERRUPTED_TASK"
    claimed = resumed.claim(recovered["task_id"])
    resumed.fail(claimed["task_id"], "temporary", retryable=True)
    assert resumed.snapshot()["counts"]["FAILED"] == 1


def test_queue_can_reopen_completed_parent_with_pending_children(tmp_path: Path):
    queue = ReplayQueue(tmp_path / "queue.json")
    queue.enqueue([("2026-09-14", None, "DAY_REPLAY", "DEVELOPMENT")])
    task_id = ReplayQueue.task_id("2026-09-14", None, "DAY_REPLAY")
    queue.claim(task_id)
    queue.complete(task_id)

    queue.reopen(task_id, "PARTIAL_DAY_REQUIRES_TICKER_RESUME")

    assert queue.snapshot()["counts"]["PENDING"] == 1
    assert queue.pending()[0]["last_error"] == "PARTIAL_DAY_REQUIRES_TICKER_RESUME"


def test_queue_is_idempotent_and_atomic(tmp_path: Path):
    path = tmp_path / "queue.json"
    queue = ReplayQueue(path)
    item = ("2026-09-10", "JOBY", "PRICE_BACKFILL", "DEVELOPMENT")
    queue.enqueue([item, item])
    assert len(queue.snapshot()["tasks"]) == 1
    json.loads(path.read_text(encoding="utf-8"))


def test_adaptive_limiter_reserves_shared_slots_and_recovers():
    now = [0.0]
    sleeps = []

    def clock():
        return now[0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = AdaptiveRateLimiter(60, clock=clock, sleep=sleep)
    limiter.wait()
    limiter.wait()
    assert sleeps == [1.0]
    limiter.record_throttle(5.0)
    assert limiter.snapshot()["current_interval_seconds"] == 2.0
    limiter.wait()
    assert sleeps[-1] == 5.0
    limiter.record_success()
    assert limiter.snapshot()["current_interval_seconds"] == 1.8


def test_retry_policy_is_bounded():
    policy = RetryPolicy(base_backoff_seconds=2, maximum_backoff_seconds=5)
    assert [policy.delay_for_attempt(value) for value in range(1, 5)] == [2, 4, 5, 5]
