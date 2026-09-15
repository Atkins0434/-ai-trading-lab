from __future__ import annotations

from datetime import date, datetime, timezone
import os
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from trainer.providers.base import ProviderError


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
    ) -> None:
        if not api_key.strip():
            raise ProviderError("Massive API key is required.")
        self._api_key = api_key
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._before_request = before_request or (lambda: None)

    @classmethod
    def from_environment(
        cls,
        variable_name: str = "MASSIVE_API_KEY",
        *,
        before_request: Callable[[], None] | None = None,
    ) -> "MassiveClient":
        api_key = os.getenv(variable_name, "")
        if not api_key:
            raise ProviderError(
                f"Missing required environment variable: {variable_name}"
            )
        return cls(api_key, before_request=before_request)

    def _get_page(
        self,
        path_or_url: str,
        params: dict[str, Any] | None = None,
        *,
        results_type: str = "array",
    ) -> dict[str, Any]:
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
        self._before_request()
        try:
            response = self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ProviderError(f"Massive request failed: {exc}") from exc

        if not isinstance(payload, dict):
            raise ProviderError("Massive response must be a JSON object.")
        if payload.get("status") not in {"OK", "DELAYED"}:
            message = payload.get("error") or payload.get("message") or "unknown"
            raise ProviderError(f"Massive returned an error: {message}")
        results = payload.get("results", [] if results_type == "array" else None)
        if results_type == "array":
            if not isinstance(results, list) or not all(
                isinstance(item, dict) for item in results
            ):
                raise ProviderError("Massive results must be an array of objects.")
        elif results_type == "object":
            if not isinstance(results, dict):
                raise ProviderError("Massive results must be an object.")
        else:
            raise ProviderError(f"Unsupported Massive results type: {results_type}")
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

    def get_tickers(self, as_of_date: str) -> list[dict[str, Any]]:
        """Return active US common stocks as they existed on a date."""
        date.fromisoformat(as_of_date)
        return self._get(
            "/v3/reference/tickers",
            {
                "market": "stocks",
                "type": "CS",
                "active": "true",
                "date": as_of_date,
                "order": "asc",
                "sort": "ticker",
                "limit": 1000,
            },
        )

    def get_ticker_overview(
        self, ticker: str, as_of_date: str
    ) -> dict[str, Any]:
        """Return one point-in-time ticker details object."""
        date.fromisoformat(as_of_date)
        payload = self._get_page(
            f"/v3/reference/tickers/{ticker.upper()}",
            {"date": as_of_date},
            results_type="object",
        )
        return payload["results"]


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
