from __future__ import annotations

from typing import Any, Protocol


class ProviderError(Exception):
    """Raised when provider data cannot be safely admitted to a replay."""


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

    def historical_universe_capabilities(self) -> dict[str, bool]:
        ...

    def get_historical_universe(
        self,
        trading_date: str,
        information_cutoff: str,
    ) -> dict[str, Any]:
        ...
