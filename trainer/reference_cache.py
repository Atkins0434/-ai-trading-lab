from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

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
