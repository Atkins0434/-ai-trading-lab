from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from trainer.catalyst_events import (
    CatalystError,
    build_catalyst_snapshot,
    calculate_shadow_catalyst_metrics,
    normalize_catalyst_event,
)
from trainer.historical_cache import HistoricalCache
from trainer.providers.base import MarketDataProvider


__all__ = [
    "CatalystError",
    "backfill_news_snapshot",
    "normalize_historical_news_record",
]


ET = ZoneInfo("America/New_York")


def _classify_event(record: dict[str, Any]) -> str:
    text = " ".join(
        str(value)
        for value in (
            record.get("title", ""),
            record.get("description", ""),
            " ".join(record.get("keywords", []) or []),
        )
    ).lower()
    rules = (
        ("OFFERING_DILUTION", ("offering", "dilution", "shelf", "warrant")),
        ("GUIDANCE", ("guidance", "outlook", "forecast")),
        ("EARNINGS", ("earnings", "quarterly results", "revenue")),
        ("FDA_REGULATORY", ("fda", "clinical trial", "phase 2", "phase 3")),
        ("MERGER_ACQUISITION", ("merger", "acquisition", "acquire")),
        ("CONTRACT_AWARD", ("contract award", "awarded contract")),
        ("ANALYST_ACTION", ("price target", "upgrade", "downgrade", "analyst")),
        ("SEC_FILING", ("8-k", "10-q", "10-k", "sec filing")),
        ("PRODUCT", ("product launch", "launches", "unveils")),
        ("LEGAL_REGULATORY", ("lawsuit", "investigation", "regulator")),
    )
    for event_type, needles in rules:
        if any(needle in text for needle in needles):
            return event_type
    return "OTHER"


def _sentiment_for_ticker(record: dict[str, Any], ticker: str) -> str:
    for insight in record.get("insights", []) or []:
        if str(insight.get("ticker", "")).upper() != ticker.upper():
            continue
        sentiment = str(insight.get("sentiment", "UNKNOWN")).upper()
        if sentiment in {"POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"}:
            return sentiment
    return "UNKNOWN"


def normalize_historical_news_record(
    record: dict[str, Any],
    *,
    ticker: str,
    provider_name: str,
    feed_version: str,
    retrieved_at: str,
) -> dict[str, Any]:
    publisher = record.get("publisher") or {}
    published = record.get("published_utc") or record.get("published_timestamp")
    availability = (
        record.get("provider_available_timestamp")
        or record.get("available_utc")
        or record.get("ingested_utc")
    )
    explicit_tickers = {
        str(value).upper() for value in (record.get("tickers") or [])
    }
    raw = {
        "ticker": ticker,
        "source_name": publisher.get("name") or provider_name,
        "source_reference": record.get("id") or record.get("article_url"),
        "headline": record.get("title") or "Untitled provider event",
        "event_type": _classify_event(record),
        "published_timestamp": published,
        "provider_available_timestamp": availability,
        "source_tier": "OTHER",
        "verification_status": "SINGLE_SOURCE",
        "sentiment": _sentiment_for_ticker(record, ticker),
        "relevance_confidence": 1.0 if ticker.upper() in explicit_tickers else 0.5,
        "duplicate_group_id": record.get("duplicate_group_id"),
    }
    return normalize_catalyst_event(
        raw,
        source_provider=provider_name,
        feed_version=feed_version,
        ingested_timestamp=retrieved_at,
    )


def backfill_news_snapshot(
    provider: MarketDataProvider,
    ticker: str,
    trading_date: str,
    *,
    cache: HistoricalCache,
    retrieved_at: str,
    lookback_days: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    day = date.fromisoformat(trading_date)
    freeze = datetime.combine(day, time(7), tzinfo=ET)
    start = freeze - timedelta(days=lookback_days)
    envelope = cache.get_news(
        provider,
        ticker,
        start.isoformat(),
        freeze.isoformat(),
        retrieved_at=retrieved_at,
    )
    actual_retrieved_at = envelope["manifest"]["retrieved_at"]
    events = [
        normalize_historical_news_record(
            record,
            ticker=ticker,
            provider_name=provider.provider_name,
            feed_version=provider.feed_version,
            retrieved_at=actual_retrieved_at,
        )
        for record in envelope["records"]
    ]
    snapshot = build_catalyst_snapshot(
        events,
        replay_id=f"{trading_date}-0700-news-backfill",
        trading_date=trading_date,
        freeze_timestamp=freeze.isoformat(),
    )
    metrics = calculate_shadow_catalyst_metrics(snapshot, ticker)
    metrics["cache_manifest"] = {
        "record_count": envelope["manifest"]["record_count"],
        "source_record_count": envelope["manifest"].get(
            "source_record_count", envelope["manifest"]["record_count"]
        ),
        "duplicate_records_removed": envelope["manifest"].get(
            "duplicate_records_removed", 0
        ),
        "content_sha256": envelope["manifest"]["content_sha256"],
    }
    return snapshot, metrics
