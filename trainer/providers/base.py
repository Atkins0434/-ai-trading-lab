from __future__ import annotations

from typing import Any, Protocol


class ProviderError(Exception):
    """Raised when provider data cannot be safely admitted to a replay."""


DEFINITIVE = "DEFINITIVE"
TRANSIENT = "TRANSIENT"


class ClassifiedProviderError(ProviderError):
    """Provider failure with evidence-safe retry and exclusion semantics."""

    def __init__(
        self,
        message: str,
        *,
        classification: str,
        http_status: int | None = None,
        exception_type: str | None = None,
        retry_count: int = 0,
        retry_after_seconds: float | None = None,
        retries_exhausted: bool = False,
    ) -> None:
        if classification not in {DEFINITIVE, TRANSIENT}:
            raise ValueError(
                "classification must be DEFINITIVE or TRANSIENT."
            )
        super().__init__(message)
        self.classification = classification
        self.http_status = http_status
        self.exception_type = exception_type or type(self).__name__
        self.retry_count = retry_count
        self.retry_after_seconds = retry_after_seconds
        self.retries_exhausted = retries_exhausted


class MarketDataProvider(Protocol):
    """Provider-neutral boundary used by historical snapshot builders."""

    provider_name: str
    feed_version: str

    def get_daily_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        ...

    def get_intraday_prices(
        self,
        ticker: str,
        start_timestamp: str,
        end_timestamp: str,
        resample_frequency: str = "1min",
    ) -> list[dict[str, Any]]:
        ...

    def get_news(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        ...


class CatalystDataProvider(Protocol):
    """Provider-neutral boundary for point-in-time catalyst evidence.

    Implementations must preserve provider availability separately from the
    publisher timestamp. A publication timestamp may not be substituted when
    the provider cannot prove when an event entered its historical feed.
    """

    provider_name: str
    feed_version: str

    def get_catalyst_events(
        self,
        tickers: list[str],
        start_timestamp: str,
        end_timestamp: str,
    ) -> list[dict[str, Any]]:
        ...


class HistoricalUniverseProvider(Protocol):
    """Point-in-time security-master boundary for research universes.

    A provider must describe its capabilities independently of the returned
    rows.  Callers fail closed when a required capability is unavailable;
    they never replace a missing historical universe with current symbols.
    """

    provider_name: str
    feed_version: str

    def historical_universe_capabilities(self) -> dict[str, Any]:
        ...

    def get_historical_universe(
        self,
        trading_date: str,
        information_cutoff: str,
    ) -> dict[str, Any]:
        ...
