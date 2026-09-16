from __future__ import annotations

from pathlib import Path

from trainer.historical_cache import HistoricalCache
from trainer.historical_news import backfill_news_snapshot


class NewsProvider:
    provider_name = "TEST_NEWS"
    feed_version = "test_news_v1"

    def __init__(self, records):
        self.records = records

    def get_news(self, tickers, start, end):
        return self.records


def record(**overrides):
    value = {
        "id": "story-1",
        "title": "JOBY raises full-year guidance",
        "published_utc": "2026-09-14T09:30:00+00:00",
        "provider_available_timestamp": "2026-09-14T09:31:00+00:00",
        "tickers": ["JOBY"],
        "publisher": {"name": "Test Wire"},
        "insights": [{"ticker": "JOBY", "sentiment": "positive"}],
    }
    value.update(overrides)
    return value


def test_backfill_admits_known_pre_freeze_news_and_extracts_sentiment(tmp_path: Path):
    snapshot, metrics = backfill_news_snapshot(
        NewsProvider([record()]),
        "JOBY",
        "2026-09-14",
        cache=HistoricalCache(tmp_path),
        retrieved_at="2026-09-15T12:00:00+00:00",
    )
    assert snapshot["summary"]["admitted"] == 1
    assert snapshot["events"][0]["event_type"] == "GUIDANCE"
    assert snapshot["events"][0]["sentiment"] == "POSITIVE"
    assert metrics["mode"] == "SHADOW_ONLY"


def test_backfill_excludes_unknown_historical_provider_availability(tmp_path: Path):
    snapshot, metrics = backfill_news_snapshot(
        NewsProvider([record(provider_available_timestamp=None)]),
        "JOBY",
        "2026-09-14",
        cache=HistoricalCache(tmp_path),
        retrieved_at="2026-09-15T12:00:00+00:00",
    )
    assert snapshot["summary"]["admitted"] == 0
    assert snapshot["events"][0]["admission_reason"] == "UNKNOWN_PROVIDER_AVAILABILITY"
    assert metrics["components"][0]["status"] == "MISSING"


def test_backfill_cache_deduplicates_provider_rows(tmp_path: Path):
    duplicate = record()
    _, metrics = backfill_news_snapshot(
        NewsProvider([duplicate, dict(duplicate)]),
        "JOBY",
        "2026-09-14",
        cache=HistoricalCache(tmp_path),
        retrieved_at="2026-09-15T12:00:00+00:00",
    )
    assert metrics["cache_manifest"]["source_record_count"] == 2
    assert metrics["cache_manifest"]["record_count"] == 1
    assert metrics["cache_manifest"]["duplicate_records_removed"] == 1
