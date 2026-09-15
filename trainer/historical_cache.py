from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Callable

from trainer.providers.base import MarketDataProvider, ProviderError


class CacheError(Exception):
    """Raised when cached historical data is missing or corrupted."""


@dataclass(frozen=True)
class CacheRequest:
    dataset: str
    ticker: str
    start: str
    end: str
    purpose: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def content_sha256(records: list[dict[str, Any]]) -> str:
    """Hash provider content independently from retrieval time."""
    return sha256(_canonical_bytes(records)).hexdigest()


def _safe_segment(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.:+-]+", value):
        raise CacheError(f"Unsafe {label}: {value}")
    return value


class HistoricalCache:
    """Immutable-on-read JSON cache with a verifiable provenance manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(
        self,
        provider: MarketDataProvider,
        request: CacheRequest,
    ) -> Path:
        provider_name = _safe_segment(provider.provider_name.lower(), "provider")
        ticker = _safe_segment(request.ticker.upper(), "ticker")
        dataset = _safe_segment(request.dataset, "dataset")
        purpose = _safe_segment(request.purpose.lower(), "purpose")
        start = _safe_segment(request.start, "start")
        end = _safe_segment(request.end, "end")
        return (
            self.root
            / provider_name
            / ticker
            / dataset
            / purpose
            / f"{start}__{end}.json"
        )

    def load(
        self,
        provider: MarketDataProvider,
        request: CacheRequest,
    ) -> dict[str, Any]:
        path = self.path_for(provider, request)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CacheError(f"Cache entry not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise CacheError(f"Cache entry is invalid JSON: {path}") from exc

        if not isinstance(envelope, dict):
            raise CacheError("Cache envelope must be an object.")
        manifest = envelope.get("manifest")
        records = envelope.get("records")
        if not isinstance(manifest, dict) or not isinstance(records, list):
            raise CacheError("Cache envelope requires manifest and records.")

        expected_request = asdict(request)
        expected_manifest = {
            "provider": provider.provider_name,
            "feed_version": provider.feed_version,
            "request": expected_request,
        }
        for field, expected in expected_manifest.items():
            if manifest.get(field) != expected:
                raise CacheError(f"Cache manifest mismatch for {field}.")

        digest = content_sha256(records)
        if manifest.get("content_sha256") != digest:
            raise CacheError("Cache content checksum mismatch.")
        if manifest.get("record_count") != len(records):
            raise CacheError("Cache record count mismatch.")
        if not all(isinstance(record, dict) for record in records):
            raise CacheError("Cache records must all be objects.")
        return envelope

    def fetch(
        self,
        provider: MarketDataProvider,
        request: CacheRequest,
        fetcher: Callable[[], list[dict[str, Any]]],
        *,
        force_refresh: bool = False,
        retrieved_at: str | None = None,
    ) -> dict[str, Any]:
        path = self.path_for(provider, request)
        if path.exists() and not force_refresh:
            return self.load(provider, request)

        try:
            records = fetcher()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Historical provider fetch failed for {request.dataset}: {exc}"
            ) from exc
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ProviderError("Provider fetch must return a list of objects.")

        retrieval_timestamp = retrieved_at or datetime.now(timezone.utc).isoformat()
        try:
            parsed_retrieval = datetime.fromisoformat(
                retrieval_timestamp.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise CacheError("retrieved_at must be ISO-8601.") from exc
        if parsed_retrieval.tzinfo is None:
            raise CacheError("retrieved_at must include a timezone.")

        envelope = {
            "manifest": {
                "cache_format_version": "historical_cache_v1.0",
                "provider": provider.provider_name,
                "feed_version": provider.feed_version,
                "request": asdict(request),
                "retrieved_at": retrieval_timestamp,
                "record_count": len(records),
                "content_sha256": content_sha256(records),
            },
            "records": records,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_bytes(_canonical_bytes(envelope) + b"\n")
        temporary_path.replace(path)
        return self.load(provider, request)

    def get_daily_prices(
        self,
        provider: MarketDataProvider,
        ticker: str,
        start_date: str,
        end_date: str,
        *,
        force_refresh: bool = False,
        retrieved_at: str | None = None,
    ) -> dict[str, Any]:
        request = CacheRequest(
            dataset="daily_prices",
            ticker=ticker,
            start=start_date,
            end=end_date,
            purpose="SELECTION_BASELINE",
        )
        return self.fetch(
            provider,
            request,
            lambda: provider.get_daily_prices(ticker, start_date, end_date),
            force_refresh=force_refresh,
            retrieved_at=retrieved_at,
        )

    def get_intraday_prices(
        self,
        provider: MarketDataProvider,
        ticker: str,
        start_timestamp: str,
        end_timestamp: str,
        *,
        purpose: str,
        force_refresh: bool = False,
        retrieved_at: str | None = None,
    ) -> dict[str, Any]:
        request = CacheRequest(
            dataset="intraday_1min",
            ticker=ticker,
            start=start_timestamp,
            end=end_timestamp,
            purpose=purpose,
        )
        return self.fetch(
            provider,
            request,
            lambda: provider.get_intraday_prices(
                ticker,
                start_timestamp,
                end_timestamp,
                "1min",
            ),
            force_refresh=force_refresh,
            retrieved_at=retrieved_at,
        )
