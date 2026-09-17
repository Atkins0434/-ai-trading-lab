from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Callable
from urllib.parse import quote
from uuid import uuid4

import requests

from trainer.providers.base import (
    DEFINITIVE,
    TRANSIENT,
    ClassifiedProviderError,
    ProviderError,
)


CACHE_VERSION = "massive_ticker_overview_cache_v1.1"
LEGACY_CACHE_VERSION = "massive_ticker_overview_cache_v1.0"
MAX_TRANSIENT_RETRIES = 4


class ReferenceCacheError(ProviderError):
    """Raised when cached point-in-time reference data is invalid."""


@dataclass(frozen=True)
class OverviewCacheResult:
    overview: dict[str, Any] | None
    cache_hit: bool
    quarter_reuse: bool
    requested_lagged_date: str
    provider_query_date: str
    definitive_miss: bool = False
    transient_retries: int = 0
    http_status: int | None = None
    exception_type: str | None = None


@dataclass(frozen=True)
class OverviewCacheBatchItem:
    ticker: str
    result: OverviewCacheResult | None
    error: ProviderError | None


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quarter(value: date) -> tuple[int, int]:
    return value.year, ((value.month - 1) // 3) + 1


class TickerOverviewCache:
    """Deterministic disk cache for lagged Massive Ticker Overview rows."""

    def __init__(
        self,
        root: Path = Path("data/reference_cache"),
        *,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.root = Path(root)
        self._sleep = sleep
        self._jitter = jitter

    def _ticker_directory(self, ticker: str) -> Path:
        normalized = ticker.strip().upper()
        if not normalized:
            raise ReferenceCacheError("Ticker Overview cache requires a ticker.")
        return self.root / quote(normalized, safe="")

    def _path(self, ticker: str, lagged_date: str) -> Path:
        date.fromisoformat(lagged_date)
        return self._ticker_directory(ticker) / f"{lagged_date}.json"

    def _read(
        self,
        path: Path,
        *,
        ticker: str,
        provider_name: str,
    ) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReferenceCacheError(
                f"Invalid Ticker Overview cache entry {path}: {exc}"
            ) from exc
        legacy_required = {
            "version",
            "provider",
            "ticker",
            "lagged_date",
            "retrieved_at",
            "overview",
            "overview_sha256",
        }
        current_required = legacy_required | {"entry_type", "failure"}
        if not isinstance(payload, dict):
            raise ReferenceCacheError(
                f"Ticker Overview cache entry has an invalid shape: {path}"
            )
        payload_keys = set(payload)
        if payload_keys not in (legacy_required, current_required):
            raise ReferenceCacheError(
                f"Ticker Overview cache entry has an invalid shape: {path}"
            )
        is_legacy = payload["version"] == LEGACY_CACHE_VERSION
        if (
            payload["version"] not in {CACHE_VERSION, LEGACY_CACHE_VERSION}
            or (is_legacy and payload_keys != legacy_required)
            or (not is_legacy and payload_keys != current_required)
            or payload["provider"] != provider_name
            or payload["ticker"] != ticker.upper()
            or path.stem != payload["lagged_date"]
            or not isinstance(payload["overview"], dict)
            or payload["overview_sha256"]
            != _canonical_digest(payload["overview"])
        ):
            raise ReferenceCacheError(
                f"Ticker Overview cache integrity check failed: {path}"
            )
        if is_legacy:
            payload = {
                **payload,
                "entry_type": "OVERVIEW",
                "failure": None,
            }
        entry_type = payload["entry_type"]
        failure = payload["failure"]
        if entry_type == "OVERVIEW":
            if failure is not None:
                raise ReferenceCacheError(
                    f"Positive cache entry contains failure metadata: {path}"
                )
        elif entry_type == "DEFINITIVE_MISS":
            if (
                payload["overview"] != {}
                or not isinstance(failure, dict)
                or set(failure)
                != {"classification", "http_status", "exception_type"}
                or failure["classification"] != DEFINITIVE
                or not isinstance(failure["exception_type"], str)
                or (
                    failure["http_status"] is not None
                    and not isinstance(failure["http_status"], int)
                )
            ):
                raise ReferenceCacheError(
                    f"Negative cache entry has invalid failure metadata: {path}"
                )
        else:
            raise ReferenceCacheError(
                f"Ticker Overview cache entry type is invalid: {path}"
            )
        try:
            date.fromisoformat(payload["lagged_date"])
            retrieved_at = datetime.fromisoformat(
                payload["retrieved_at"].replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as exc:
            raise ReferenceCacheError(
                f"Ticker Overview cache timestamps are invalid: {path}"
            ) from exc
        if retrieved_at.tzinfo is None:
            raise ReferenceCacheError(
                f"Ticker Overview cache retrieval time lacks timezone: {path}"
            )
        return payload

    def _write(
        self,
        path: Path,
        *,
        ticker: str,
        lagged_date: str,
        provider_name: str,
        overview: dict[str, Any],
        failure: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "version": CACHE_VERSION,
            "provider": provider_name,
            "ticker": ticker.upper(),
            "lagged_date": lagged_date,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "entry_type": (
                "DEFINITIVE_MISS" if failure is not None else "OVERVIEW"
            ),
            "overview": overview,
            "overview_sha256": _canonical_digest(overview),
            "failure": failure,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _reusable_path(
        self,
        ticker: str,
        lagged_date: str,
        provider_name: str,
    ) -> Path | None:
        requested = date.fromisoformat(lagged_date)
        exact = self._path(ticker, lagged_date)
        if exact.is_file():
            return exact
        directory = exact.parent
        if not directory.is_dir():
            return None
        candidates: list[tuple[date, Path]] = []
        for path in directory.glob("*.json"):
            try:
                cached_date = date.fromisoformat(path.stem)
            except ValueError:
                continue
            # Reusing a later snapshot could shorten the configured lag and
            # leak information. The closest earlier date in the same quarter
            # is deterministic and at least as conservative as the request.
            if (
                _quarter(cached_date) == _quarter(requested)
                and cached_date <= requested
            ):
                payload = self._read(
                    path,
                    ticker=ticker,
                    provider_name=provider_name,
                )
                # A definitive miss is knowable only for its exact query date;
                # never carry that absence forward through quarter reuse.
                if payload["entry_type"] == "OVERVIEW":
                    candidates.append((cached_date, path))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def get(
        self,
        reference_client: Any,
        ticker: str,
        lagged_date: str,
    ) -> OverviewCacheResult:
        """Return a cached quarter proxy or fetch the exact lagged date."""
        date.fromisoformat(lagged_date)
        provider_name = str(reference_client.provider_name)
        cached = self._cached_result(
            ticker=ticker,
            lagged_date=lagged_date,
            provider_name=provider_name,
        )
        return cached or self._fetch_result(
            reference_client, ticker, lagged_date
        )

    @staticmethod
    def _result_from_payload(
        payload: dict[str, Any],
        *,
        lagged_date: str,
    ) -> OverviewCacheResult:
        provider_query_date = payload["lagged_date"]
        failure = payload["failure"]
        definitive_miss = payload["entry_type"] == "DEFINITIVE_MISS"
        return OverviewCacheResult(
            overview=None if definitive_miss else payload["overview"],
            cache_hit=True,
            quarter_reuse=provider_query_date != lagged_date,
            requested_lagged_date=lagged_date,
            provider_query_date=provider_query_date,
            definitive_miss=definitive_miss,
            http_status=(failure or {}).get("http_status"),
            exception_type=(failure or {}).get("exception_type"),
        )

    def _cached_result(
        self,
        *,
        ticker: str,
        lagged_date: str,
        provider_name: str,
    ) -> OverviewCacheResult | None:
        reusable = self._reusable_path(
            ticker, lagged_date, provider_name
        )
        if reusable is None:
            return None
        payload = self._read(
            reusable,
            ticker=ticker,
            provider_name=provider_name,
        )
        return self._result_from_payload(
            payload,
            lagged_date=lagged_date,
        )

    def _retry_delay(
        self,
        retry_number: int,
        retry_after_seconds: float | None,
    ) -> float:
        base = float(2 ** (retry_number - 1))
        jittered = base + self._jitter(0.0, base * 0.25)
        return max(jittered, retry_after_seconds or 0.0)

    @staticmethod
    def _classified_error(
        error: BaseException,
        *,
        retry_count: int,
        retries_exhausted: bool,
    ) -> ClassifiedProviderError:
        if isinstance(error, ClassifiedProviderError):
            return ClassifiedProviderError(
                str(error),
                classification=error.classification,
                http_status=error.http_status,
                exception_type=error.exception_type,
                retry_count=retry_count + error.retry_count,
                retry_after_seconds=error.retry_after_seconds,
                retries_exhausted=(
                    retries_exhausted or error.retries_exhausted
                ),
            )
        return ClassifiedProviderError(
            str(error),
            classification=TRANSIENT,
            exception_type=type(error).__name__,
            retry_count=retry_count,
            retries_exhausted=retries_exhausted,
        )

    def _fetch_result(
        self,
        reference_client: Any,
        ticker: str,
        lagged_date: str,
    ) -> OverviewCacheResult:
        provider_name = str(reference_client.provider_name)
        cache_retries = 0
        while True:
            try:
                overview = reference_client.get_ticker_overview(
                    ticker, lagged_date
                )
            except (
                ClassifiedProviderError,
                requests.ConnectionError,
                requests.Timeout,
            ) as exc:
                classified = self._classified_error(
                    exc,
                    retry_count=cache_retries,
                    retries_exhausted=False,
                )
                if classified.classification == DEFINITIVE:
                    failure = {
                        "classification": DEFINITIVE,
                        "http_status": classified.http_status,
                        "exception_type": classified.exception_type,
                    }
                    self._write(
                        self._path(ticker, lagged_date),
                        ticker=ticker,
                        lagged_date=lagged_date,
                        provider_name=provider_name,
                        overview={},
                        failure=failure,
                    )
                    return OverviewCacheResult(
                        overview=None,
                        cache_hit=False,
                        quarter_reuse=False,
                        requested_lagged_date=lagged_date,
                        provider_query_date=lagged_date,
                        definitive_miss=True,
                        transient_retries=classified.retry_count,
                        http_status=classified.http_status,
                        exception_type=classified.exception_type,
                    )
                if (
                    classified.retries_exhausted
                    or classified.retry_count >= MAX_TRANSIENT_RETRIES
                ):
                    raise self._classified_error(
                        classified,
                        retry_count=0,
                        retries_exhausted=True,
                    ) from exc
                cache_retries += 1
                self._sleep(
                    self._retry_delay(
                        cache_retries,
                        classified.retry_after_seconds,
                    )
                )
                continue

            if not isinstance(overview, dict) or not overview:
                failure = {
                    "classification": DEFINITIVE,
                    "http_status": 200,
                    "exception_type": "EMPTY_RESULTS",
                }
                self._write(
                    self._path(ticker, lagged_date),
                    ticker=ticker,
                    lagged_date=lagged_date,
                    provider_name=provider_name,
                    overview={},
                    failure=failure,
                )
                return OverviewCacheResult(
                    overview=None,
                    cache_hit=False,
                    quarter_reuse=False,
                    requested_lagged_date=lagged_date,
                    provider_query_date=lagged_date,
                    definitive_miss=True,
                    transient_retries=cache_retries,
                    http_status=200,
                    exception_type="EMPTY_RESULTS",
                )

            provider_retries = int(
                getattr(overview, "transient_retries", 0)
            )
            normalized_overview = dict(overview)
            self._write(
                self._path(ticker, lagged_date),
                ticker=ticker,
                lagged_date=lagged_date,
                provider_name=provider_name,
                overview=normalized_overview,
            )
            return OverviewCacheResult(
                overview=normalized_overview,
                cache_hit=False,
                quarter_reuse=False,
                requested_lagged_date=lagged_date,
                provider_query_date=lagged_date,
                transient_retries=cache_retries + provider_retries,
                http_status=getattr(overview, "http_status", 200),
            )

    def get_many(
        self,
        reference_client: Any,
        tickers: list[str],
        lagged_date: str,
        *,
        max_workers: int,
        on_progress: Callable[[OverviewCacheBatchItem], None] | None = None,
    ) -> list[OverviewCacheBatchItem]:
        """Resolve cache hits serially and fetch misses with bounded workers."""
        date.fromisoformat(lagged_date)
        if max_workers < 1:
            raise ValueError("max_workers must be at least one.")
        provider_name = str(reference_client.provider_name)
        outcomes: list[OverviewCacheBatchItem | None] = [None] * len(tickers)
        pending: list[tuple[int, str]] = []

        def complete(index: int, item: OverviewCacheBatchItem) -> None:
            outcomes[index] = item
            if on_progress is not None:
                on_progress(item)

        for index, raw_ticker in enumerate(tickers):
            ticker = str(raw_ticker).strip().upper()
            cached = self._cached_result(
                ticker=ticker,
                lagged_date=lagged_date,
                provider_name=provider_name,
            )
            if cached is None:
                pending.append((index, ticker))
            else:
                complete(
                    index,
                    OverviewCacheBatchItem(ticker, cached, None),
                )

        if max_workers == 1:
            for index, ticker in pending:
                try:
                    result = self._fetch_result(
                        reference_client, ticker, lagged_date
                    )
                except ClassifiedProviderError as exc:
                    item = OverviewCacheBatchItem(ticker, None, exc)
                else:
                    item = OverviewCacheBatchItem(ticker, result, None)
                complete(index, item)
        elif pending:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: dict[Future[OverviewCacheResult], tuple[int, str]] = {
                    executor.submit(
                        self._fetch_result,
                        reference_client,
                        ticker,
                        lagged_date,
                    ): (index, ticker)
                    for index, ticker in pending
                }
                for future in as_completed(futures):
                    index, ticker = futures[future]
                    try:
                        result = future.result()
                    except ClassifiedProviderError as exc:
                        item = OverviewCacheBatchItem(ticker, None, exc)
                    else:
                        item = OverviewCacheBatchItem(ticker, result, None)
                    complete(index, item)

        if any(item is None for item in outcomes):
            raise ReferenceCacheError(
                "Ticker Overview batch did not resolve every requested ticker."
            )
        return [item for item in outcomes if item is not None]
