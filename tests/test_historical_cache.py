from __future__ import annotations

import json

import pytest

from trainer.historical_cache import CacheError, HistoricalCache


class FakeProvider:
    provider_name = "TEST_DATA"
    feed_version = "test_v1"

    def __init__(self) -> None:
        self.daily_calls = 0
        self.intraday_calls = 0

    def get_daily_prices(self, ticker, start_date, end_date):
        self.daily_calls += 1
        return [{"date": start_date, "close": 10.0}]

    def get_intraday_prices(
        self,
        ticker,
        start_timestamp,
        end_timestamp,
        resample_frequency="1min",
    ):
        self.intraday_calls += 1
        return [{"date": start_timestamp, "close": 10.0}]

    def get_news(self, tickers, start_date, end_date):
        return []


def test_cache_fetches_once_and_verifies_manifest(tmp_path):
    provider = FakeProvider()
    cache = HistoricalCache(tmp_path)

    first = cache.get_daily_prices(
        provider,
        "SPY",
        "2017-12-01",
        "2018-01-02",
        retrieved_at="2026-09-15T15:00:00+00:00",
    )
    second = cache.get_daily_prices(
        provider,
        "SPY",
        "2017-12-01",
        "2018-01-02",
    )

    assert first == second
    assert provider.daily_calls == 1
    assert first["manifest"]["record_count"] == 1
    assert len(first["manifest"]["content_sha256"]) == 64
    assert first["manifest"]["request"]["purpose"] == "SELECTION_BASELINE"


def test_cache_detects_content_tampering(tmp_path):
    provider = FakeProvider()
    cache = HistoricalCache(tmp_path)
    cache.get_daily_prices(
        provider,
        "SPY",
        "2017-12-01",
        "2018-01-02",
    )

    path = next(tmp_path.rglob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["close"] = 999.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CacheError, match="checksum"):
        cache.get_daily_prices(
            provider,
            "SPY",
            "2017-12-01",
            "2018-01-02",
        )


def test_intraday_cache_separates_selection_and_outcome(tmp_path):
    provider = FakeProvider()
    cache = HistoricalCache(tmp_path)
    kwargs = {
        "provider": provider,
        "ticker": "SPY",
        "start_timestamp": "2018-01-02T09:30:00-05:00",
        "end_timestamp": "2018-01-02T16:00:00-05:00",
    }

    selection = cache.get_intraday_prices(
        **kwargs,
        purpose="SCOUT_SELECTION",
    )
    outcome = cache.get_intraday_prices(
        **kwargs,
        purpose="OUTCOME_GRADING",
    )

    assert selection["manifest"]["request"]["purpose"] == "SCOUT_SELECTION"
    assert outcome["manifest"]["request"]["purpose"] == "OUTCOME_GRADING"
    assert provider.intraday_calls == 2


def test_unsafe_cache_segment_is_rejected(tmp_path):
    with pytest.raises(CacheError, match="Unsafe ticker"):
        HistoricalCache(tmp_path).get_daily_prices(
            FakeProvider(),
            "../SPY",
            "2017-12-01",
            "2018-01-02",
        )


def test_cache_deduplicates_exact_provider_records(tmp_path):
    provider = FakeProvider()
    provider.get_news = lambda tickers, start, end: [
        {"id": "same", "published_utc": start},
        {"published_utc": start, "id": "same"},
    ]
    result = HistoricalCache(tmp_path).get_news(
        provider,
        "SPY",
        "2026-09-14T00:00:00Z",
        "2026-09-14T07:00:00-04:00",
    )
    assert len(result["records"]) == 1
    assert result["manifest"]["source_record_count"] == 2
    assert result["manifest"]["duplicate_records_removed"] == 1
