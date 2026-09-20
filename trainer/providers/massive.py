from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import os
import random
import threading
import time
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from trainer.providers.base import (
    DEFINITIVE,
    TRANSIENT,
    ClassifiedProviderError,
    ProviderError,
)
from trainer.rate_control import (
    AdaptiveRateLimiter,
    RETRYABLE_STATUS_CODES,
    RetryPolicy,
    load_massive_plan,
)


@dataclass(frozen=True)
class _RequestMetadata:
    transient_retries: int
    http_status: int | None


class TickerOverviewResult(dict[str, Any]):
    """Dictionary-compatible overview carrying operational retry metadata."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        transient_retries: int,
        http_status: int | None,
    ) -> None:
        super().__init__(payload)
        self.transient_retries = transient_retries
        self.http_status = http_status


class MassiveClient:
    """Small REST adapter for Massive stock aggregate and news data."""

    provider_name = "MASSIVE"
    feed_version = "massive_rest_v2_aggs"
    base_url = "https://api.massive.com"

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        timeout_seconds: float = 30.0,
        before_request: Callable[[], None] | None = None,
        rate_limiter: AdaptiveRateLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if not api_key.strip():
            raise ProviderError("Massive API key is required.")
        self._api_key = api_key
        self._session = session
        self._thread_local = threading.local()
        self._timeout_seconds = timeout_seconds
        self._before_request = before_request or (lambda: None)
        self._rate_limiter = rate_limiter
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._jitter = jitter
        self.massive_plan = load_massive_plan()

    @classmethod
    def from_environment(
        cls,
        variable_name: str = "MASSIVE_API_KEY",
        *,
        before_request: Callable[[], None] | None = None,
        rate_limiter: AdaptiveRateLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> "MassiveClient":
        api_key = os.getenv(variable_name, "")
        if not api_key:
            raise ProviderError(
                f"Missing required environment variable: {variable_name}"
            )
        return cls(
            api_key,
            before_request=before_request,
            rate_limiter=rate_limiter,
            retry_policy=retry_policy,
        )

    def _request_session(self) -> requests.Session:
        if self._session is not None:
            return self._session
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    @staticmethod
    def _retry_after_seconds(response: Any) -> float | None:
        raw = getattr(response, "headers", {}).get("Retry-After")
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(raw))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(
                    0.0,
                    (retry_at - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                return None

    def _retry_delay(
        self,
        attempt: int,
        retry_after_seconds: float | None,
    ) -> float:
        base = self._retry_policy.delay_for_attempt(attempt)
        jittered = base + self._jitter(0.0, base * 0.25)
        return max(jittered, retry_after_seconds or 0.0)

    def _get_page_with_metadata(
        self,
        path_or_url: str,
        params: dict[str, Any] | None = None,
        *,
        results_type: str = "array",
        allow_missing_results: bool = False,
    ) -> tuple[dict[str, Any], _RequestMetadata]:
        if path_or_url.startswith("http"):
            parsed = urlparse(path_or_url)
            if parsed.scheme != "https" or parsed.netloc != "api.massive.com":
                raise ProviderError("Massive pagination URL changed origin.")
            url = path_or_url
        else:
            url = f"{self.base_url}{path_or_url}"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        transient_retries = 0
        max_attempts = min(self._retry_policy.max_attempts, 5)
        for attempt in range(1, max_attempts + 1):
            self._before_request()
            if self._rate_limiter is not None:
                self._rate_limiter.wait()
            try:
                response = self._request_session().get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt < max_attempts:
                    transient_retries += 1
                    self._sleep(self._retry_delay(attempt, None))
                    continue
                raise ClassifiedProviderError(
                    f"Massive request failed after transient retries: {exc}",
                    classification=TRANSIENT,
                    exception_type=type(exc).__name__,
                    retry_count=transient_retries,
                    retries_exhausted=True,
                ) from exc
            except requests.RequestException as exc:
                raise ProviderError(
                    f"Massive request failed: {exc}"
                ) from exc

            status_code = int(getattr(response, "status_code", 200))
            retry_after_seconds = self._retry_after_seconds(response)
            if status_code == 404:
                raise ClassifiedProviderError(
                    "Massive returned HTTP 404.",
                    classification=DEFINITIVE,
                    http_status=404,
                    exception_type="HTTPError",
                    retry_count=transient_retries,
                )
            is_transient_status = (
                status_code == 429
                or 500 <= status_code <= 599
            )
            if is_transient_status:
                if self._rate_limiter is not None:
                    self._rate_limiter.record_throttle(retry_after_seconds)
                if attempt < max_attempts:
                    transient_retries += 1
                    self._sleep(
                        self._retry_delay(attempt, retry_after_seconds)
                    )
                    continue
                raise ClassifiedProviderError(
                    f"Massive returned transient HTTP {status_code} after retries.",
                    classification=TRANSIENT,
                    http_status=status_code,
                    exception_type="HTTPError",
                    retry_count=transient_retries,
                    retry_after_seconds=retry_after_seconds,
                    retries_exhausted=True,
                )
            try:
                response.raise_for_status()
            except requests.RequestException as exc:
                raise ProviderError(
                    f"Massive request failed with HTTP {status_code}: {exc}"
                ) from exc
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderError(
                    "Massive response was not valid JSON."
                ) from exc
            if self._rate_limiter is not None:
                self._rate_limiter.record_success()
            break
        else:  # pragma: no cover - loop exits through success or exception.
            raise ProviderError("Massive request loop ended unexpectedly.")

        if not isinstance(payload, dict):
            raise ProviderError("Massive response must be a JSON object.")
        missing_allowed_result = (
            allow_missing_results
            and status_code == 200
            and payload.get("results") in (None, {}, [])
        )
        if (
            payload.get("status") not in {"OK", "DELAYED"}
            and not missing_allowed_result
        ):
            message = payload.get("error") or payload.get("message") or "unknown"
            raise ProviderError(f"Massive returned an error: {message}")
        results = payload.get("results", [] if results_type == "array" else None)
        if results_type == "array":
            if not isinstance(results, list) or not all(
                isinstance(item, dict) for item in results
            ):
                raise ProviderError("Massive results must be an array of objects.")
        elif results_type == "object":
            if allow_missing_results and results in (None, {}, []):
                pass
            elif not isinstance(results, dict):
                raise ProviderError("Massive results must be an object.")
        else:
            raise ProviderError(f"Unsupported Massive results type: {results_type}")
        return payload, _RequestMetadata(
            transient_retries=transient_retries,
            http_status=status_code,
        )

    def _get_page(
        self,
        path_or_url: str,
        params: dict[str, Any] | None = None,
        *,
        results_type: str = "array",
    ) -> dict[str, Any]:
        # Keep the established request behavior for aggregates, news, and
        # ticker-list pagination. The classified retry path above is limited
        # to Ticker Overview, whose per-ticker failures can be admitted or
        # rejected independently by the universe builder.
        if path_or_url.startswith("http"):
            parsed = urlparse(path_or_url)
            if parsed.scheme != "https" or parsed.netloc != "api.massive.com":
                raise ProviderError("Massive pagination URL changed origin.")
            url = path_or_url
        else:
            url = f"{self.base_url}{path_or_url}"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        payload = None
        last_error: Exception | None = None
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            self._before_request()
            if self._rate_limiter is not None:
                self._rate_limiter.wait()
            try:
                response = self._request_session().get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
                status_code = int(getattr(response, "status_code", 200))
                if status_code in RETRYABLE_STATUS_CODES:
                    retry_after = getattr(response, "headers", {}).get(
                        "Retry-After"
                    )
                    retry_after_seconds = None
                    if retry_after is not None:
                        try:
                            retry_after_seconds = float(retry_after)
                        except (TypeError, ValueError):
                            retry_after_seconds = None
                    if self._rate_limiter is not None:
                        self._rate_limiter.record_throttle(
                            retry_after_seconds
                        )
                    if attempt < self._retry_policy.max_attempts:
                        self._sleep(
                            self._retry_policy.delay_for_attempt(attempt)
                        )
                        continue
                response.raise_for_status()
                payload = response.json()
                if self._rate_limiter is not None:
                    self._rate_limiter.record_success()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt >= self._retry_policy.max_attempts:
                    break
                self._sleep(self._retry_policy.delay_for_attempt(attempt))
        if payload is None:
            raise ProviderError(
                f"Massive request failed: {last_error}"
            ) from last_error

        if not isinstance(payload, dict):
            raise ProviderError("Massive response must be a JSON object.")
        if payload.get("status") not in {"OK", "DELAYED"}:
            message = payload.get("error") or payload.get("message") or "unknown"
            raise ProviderError(f"Massive returned an error: {message}")
        results = payload.get(
            "results", [] if results_type == "array" else None
        )
        if results_type == "array":
            if not isinstance(results, list) or not all(
                isinstance(item, dict) for item in results
            ):
                raise ProviderError(
                    "Massive results must be an array of objects."
                )
        elif results_type == "object":
            if not isinstance(results, dict):
                raise ProviderError("Massive results must be an object.")
        else:
            raise ProviderError(
                f"Unsupported Massive results type: {results_type}"
            )
        return payload

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        payload = self._get_page(path, params)
        while True:
            records.extend(payload.get("results", []))
            next_url = payload.get("next_url")
            if not next_url:
                break
            if not isinstance(next_url, str):
                raise ProviderError("Massive next_url must be a string.")
            payload = self._get_page(next_url)
        return records

    @staticmethod
    def _aggregate_boundary(value: str) -> str:
        """Convert ISO timestamps to the millisecond path form."""
        if "T" not in value:
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ProviderError(
                    f"Invalid aggregate boundary: {value}"
                ) from exc
            return value
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProviderError(f"Invalid aggregate boundary: {value}") from exc
        if parsed.tzinfo is None:
            raise ProviderError("Aggregate timestamps require a timezone.")
        return str(int(parsed.timestamp() * 1000))

    def _get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        start: str,
        end: str,
    ) -> list[dict[str, Any]]:
        start_boundary = self._aggregate_boundary(start)
        end_boundary = self._aggregate_boundary(end)
        return self._get(
            f"/v2/aggs/ticker/{ticker.upper()}/range/"
            f"{multiplier}/{timespan}/{start_boundary}/{end_boundary}",
            {"adjusted": "true", "sort": "asc", "limit": 50000},
        )

    def get_daily_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        return self._get_aggregates(
            ticker, 1, "day", start_date, end_date
        )

    def get_intraday_prices(
        self,
        ticker: str,
        start_timestamp: str,
        end_timestamp: str,
        resample_frequency: str = "1min",
    ) -> list[dict[str, Any]]:
        if resample_frequency != "1min":
            raise ProviderError(
                "Massive adapter currently supports only 1min aggregates."
            )
        return self._get_aggregates(
            ticker,
            1,
            "minute",
            start_timestamp,
            end_timestamp,
        )

    def get_news(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        if len(tickers) != 1:
            raise ProviderError(
                "Massive news requests currently require exactly one ticker."
            )
        return self._get(
            "/v2/reference/news",
            {
                "ticker": tickers[0].upper(),
                "published_utc.gte": start_date,
                "published_utc.lte": end_date,
                "sort": "published_utc",
                "order": "asc",
                "limit": 1000,
            },
        )

    def get_news_response(self, ticker: str, start: str, end: str) -> dict[str, Any]:
        """Preserve raw paginated news responses for the immutable replay cache."""
        pages = []
        payload = self._get_page("/v2/reference/news", {
            "ticker": ticker.upper(), "published_utc.gte": start,
            "published_utc.lte": end, "sort": "published_utc",
            "order": "asc", "limit": 1000,
        })
        while True:
            pages.append(payload)
            next_url = payload.get("next_url")
            if not next_url:
                return {"pages": pages}
            if not isinstance(next_url, str):
                raise ProviderError("Massive next_url must be a string.")
            payload = self._get_page(next_url)

    def get_tickers(
        self,
        as_of_date: str,
        *,
        active: bool = True,
        security_type: str | None = "CS",
    ) -> list[dict[str, Any]]:
        """Return ticker rows for a provider-documented point-in-time date."""
        date.fromisoformat(as_of_date)
        params = {
            "market": "stocks",
            "active": "true" if active else "false",
            "date": as_of_date,
            "order": "asc",
            "sort": "ticker",
            "limit": 1000,
        }
        if security_type is not None:
            params["type"] = security_type
        return self._get(
            "/v3/reference/tickers",
            params,
        )

    def get_ticker_overview(
        self, ticker: str, as_of_date: str
    ) -> dict[str, Any]:
        """Return one point-in-time ticker details object."""
        date.fromisoformat(as_of_date)
        payload, metadata = self._get_page_with_metadata(
            f"/v3/reference/tickers/{ticker.upper()}",
            {"date": as_of_date},
            results_type="object",
            allow_missing_results=True,
        )
        results = payload.get("results")
        if results in (None, {}, []):
            raise ClassifiedProviderError(
                f"Massive returned no Ticker Overview for {ticker.upper()}.",
                classification=DEFINITIVE,
                http_status=200,
                exception_type="EMPTY_RESULTS",
                retry_count=metadata.transient_retries,
            )
        if not isinstance(results, dict):
            raise ProviderError("Massive Ticker Overview must be an object.")
        return TickerOverviewResult(
            results,
            transient_retries=metadata.transient_retries,
            http_status=metadata.http_status,
        )

    def historical_universe_capabilities(self) -> dict[str, Any]:
        """Declare only capabilities Massive can prove for this adapter.

        The All Tickers endpoint is date-aware and exposes FIGIs, but the
        current adapter cannot prove availability-time semantics for historical
        market cap/classification fields or reconstruct every ticker-identity
        interval (the ticker-events API is experimental). Research therefore
        fails closed until those gaps are supplied by a security-master source.
        """
        return {
            "point_in_time_listings": True,
            "delisted_securities": True,
            "stable_security_ids": True,
            "ticker_history": False,
            "point_in_time_exchange": True,
            "point_in_time_security_type": True,
            "historical_trading_status": True,
            "point_in_time_market_cap": False,
            "provider_response_complete": True,
        }

    def get_historical_universe(
        self, trading_date: str, information_cutoff: str
    ) -> dict[str, Any]:
        """Normalize date-scoped ticker rows without inventing missing facts.

        This is callable for capability experiments, but the resolver will not
        admit its output as evidence while required capabilities above remain
        false.
        """
        records = []
        for raw in self.get_tickers(trading_date, active=True):
            records.append({
                "ticker": raw.get("ticker"),
                "share_class_figi": raw.get("share_class_figi"),
                "composite_figi": raw.get("composite_figi"),
                "primary_exchange": raw.get("primary_exchange"),
                "type": raw.get("type"),
                "locale": raw.get("locale"),
                "market": raw.get("market"),
                "list_date": raw.get("list_date"),
                "delisted_utc": raw.get("delisted_utc"),
                "historical_active": raw.get("active"),
            })
        return {
            "records": records,
            "query_parameters": {
                "endpoint": "/v3/reference/tickers",
                "date": trading_date,
                "active": True,
                "market": "stocks",
                "information_cutoff": information_cutoff,
            },
        }


def parse_massive_timestamp(record: dict[str, Any]) -> str:
    """Convert a Massive aggregate millisecond timestamp to ISO-8601 UTC."""
    raw = record.get("t")
    if not isinstance(raw, (int, float)) or raw < 0:
        raise ProviderError("Massive aggregate is missing timestamp `t`.")
    return datetime.fromtimestamp(
        raw / 1000,
        tz=timezone.utc,
    ).isoformat()


def canonical_price_bar(
    record: dict[str, Any],
    *,
    include_source: bool = False,
) -> dict[str, Any]:
    """Convert one Massive aggregate to the provider-neutral bar shape."""
    fields = {
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
    }
    missing = [name for name, source in fields.items() if record.get(source) is None]
    if missing:
        raise ProviderError(f"Massive aggregate is missing fields: {missing}")
    bar = {
        "timestamp": parse_massive_timestamp(record),
        **{name: record[source] for name, source in fields.items()},
    }
    if include_source:
        bar["source"] = MassiveClient.provider_name
    return bar
