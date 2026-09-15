from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import requests

from trainer.providers.base import ProviderError


class TiingoClient:
    """Small, testable Tiingo transport adapter.

    It retrieves raw provider records. Snapshot construction and feature
    calculation remain separate so Scout is not coupled to one vendor.
    """

    provider_name = "TIINGO"
    feed_version = "tiingo_rest_v1"
    base_url = "https://api.tiingo.com"

    def __init__(
        self,
        api_token: str,
        session: requests.Session | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_token.strip():
            raise ProviderError("Tiingo API token is required.")
        self._api_token = api_token
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(
        cls,
        variable_name: str = "TIINGO_API_TOKEN",
    ) -> "TiingoClient":
        token = os.getenv(variable_name, "")
        if not token:
            raise ProviderError(
                f"Missing required environment variable: {variable_name}"
            )
        return cls(token)

    def _get(
        self,
        path: str,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        headers = {
            "Authorization": f"Token {self._api_token}",
            "Accept": "application/json",
        }
        try:
            response = self._session.get(
                f"{self.base_url}{path}",
                params=params,
                headers=headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ProviderError(f"Tiingo request failed: {exc}") from exc

        if not isinstance(payload, list):
            raise ProviderError("Tiingo response must be a JSON array.")
        if not all(isinstance(item, dict) for item in payload):
            raise ProviderError("Tiingo response contains a non-object item.")
        return payload

    def get_daily_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        return self._get(
            f"/tiingo/daily/{ticker}/prices",
            {
                "startDate": start_date,
                "endDate": end_date,
                "format": "json",
            },
        )

    def get_intraday_prices(
        self,
        ticker: str,
        start_timestamp: str,
        end_timestamp: str,
        resample_frequency: str = "1min",
    ) -> list[dict[str, Any]]:
        return self._get(
            f"/iex/{ticker}/prices",
            {
                "startDate": start_timestamp,
                "endDate": end_timestamp,
                "resampleFreq": resample_frequency,
                "format": "json",
            },
        )

    def get_news(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        return self._get(
            "/tiingo/news",
            {
                "tickers": ",".join(tickers),
                "startDate": start_date,
                "endDate": end_date,
            },
        )


def parse_tiingo_timestamp(record: dict[str, Any]) -> str:
    """Extract and validate Tiingo's timestamp field."""
    raw = record.get("date") or record.get("publishedDate")
    if not isinstance(raw, str):
        raise ProviderError("Tiingo record is missing a timestamp.")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProviderError(f"Invalid Tiingo timestamp: {raw}") from exc
    if parsed.tzinfo is None:
        raise ProviderError("Tiingo timestamp must include a timezone.")
    return raw


def market_observation(
    value: float | int | None,
    record: dict[str, Any],
    provider_field: str,
) -> dict[str, Any]:
    """Map a provider value into the point-in-time snapshot contract."""
    return {
        "value": value,
        "as_of_timestamp": parse_tiingo_timestamp(record),
        "source": TiingoClient.provider_name,
        "provider_field": provider_field,
    }
