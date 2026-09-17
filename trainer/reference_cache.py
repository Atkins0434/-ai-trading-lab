from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from uuid import uuid4

from trainer.providers.base import ProviderError


CACHE_VERSION = "massive_ticker_overview_cache_v1.0"


class ReferenceCacheError(ProviderError):
    """Raised when cached point-in-time reference data is invalid."""


@dataclass(frozen=True)
class OverviewCacheResult:
    overview: dict[str, Any]
    cache_hit: bool
    quarter_reuse: bool
    requested_lagged_date: str
    provider_query_date: str


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

    def __init__(self, root: Path = Path("data/reference_cache")) -> None:
        self.root = Path(root)

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
        required = {
            "version",
            "provider",
            "ticker",
            "lagged_date",
            "retrieved_at",
            "overview",
            "overview_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ReferenceCacheError(
                f"Ticker Overview cache entry has an invalid shape: {path}"
            )
        if (
            payload["version"] != CACHE_VERSION
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
    ) -> None:
        payload = {
            "version": CACHE_VERSION,
            "provider": provider_name,
            "ticker": ticker.upper(),
            "lagged_date": lagged_date,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "overview": overview,
            "overview_sha256": _canonical_digest(overview),
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

    def _reusable_path(self, ticker: str, lagged_date: str) -> Path | None:
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
        requested = date.fromisoformat(lagged_date)
        provider_name = str(reference_client.provider_name)
        reusable = self._reusable_path(ticker, lagged_date)
        if reusable is not None:
            payload = self._read(
                reusable,
                ticker=ticker,
                provider_name=provider_name,
            )
            provider_query_date = payload["lagged_date"]
            return OverviewCacheResult(
                overview=payload["overview"],
                cache_hit=True,
                quarter_reuse=provider_query_date != lagged_date,
                requested_lagged_date=lagged_date,
                provider_query_date=provider_query_date,
            )

        overview = reference_client.get_ticker_overview(ticker, lagged_date)
        if not isinstance(overview, dict):
            raise ReferenceCacheError(
                f"Ticker Overview for {ticker} must be an object."
            )
        self._write(
            self._path(ticker, lagged_date),
            ticker=ticker,
            lagged_date=lagged_date,
            provider_name=provider_name,
            overview=overview,
        )
        return OverviewCacheResult(
            overview=overview,
            cache_hit=False,
            quarter_reuse=False,
            requested_lagged_date=lagged_date,
            provider_query_date=requested.isoformat(),
        )

    def _cached_result(
        self,
        *,
        ticker: str,
        lagged_date: str,
        provider_name: str,
    ) -> OverviewCacheResult | None:
        reusable = self._reusable_path(ticker, lagged_date)
        if reusable is None:
            return None
        payload = self._read(
            reusable,
            ticker=ticker,
            provider_name=provider_name,
        )
        provider_query_date = payload["lagged_date"]
        return OverviewCacheResult(
            overview=payload["overview"],
            cache_hit=True,
            quarter_reuse=provider_query_date != lagged_date,
            requested_lagged_date=lagged_date,
            provider_query_date=provider_query_date,
        )

    def _fetch_result(
        self,
        reference_client: Any,
        ticker: str,
        lagged_date: str,
    ) -> OverviewCacheResult:
        overview = reference_client.get_ticker_overview(ticker, lagged_date)
        if not isinstance(overview, dict):
            raise ReferenceCacheError(
                f"Ticker Overview for {ticker} must be an object."
            )
        self._write(
            self._path(ticker, lagged_date),
            ticker=ticker,
            lagged_date=lagged_date,
            provider_name=str(reference_client.provider_name),
            overview=overview,
        )
        return OverviewCacheResult(
            overview=overview,
            cache_hit=False,
            quarter_reuse=False,
            requested_lagged_date=lagged_date,
            provider_query_date=lagged_date,
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
            try:
                cached = self._cached_result(
                    ticker=ticker,
                    lagged_date=lagged_date,
                    provider_name=provider_name,
                )
            except ProviderError as exc:
                complete(
                    index,
                    OverviewCacheBatchItem(ticker, None, exc),
                )
            else:
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
                except ProviderError as exc:
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
                    except ProviderError as exc:
                        item = OverviewCacheBatchItem(ticker, None, exc)
                    else:
                        item = OverviewCacheBatchItem(ticker, result, None)
                    complete(index, item)

        if any(item is None for item in outcomes):
            raise ReferenceCacheError(
                "Ticker Overview batch did not resolve every requested ticker."
            )
        return [item for item in outcomes if item is not None]
