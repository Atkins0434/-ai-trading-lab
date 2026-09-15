"""Historical data-provider boundaries."""

from trainer.providers.base import (
    CatalystDataProvider,
    MarketDataProvider,
    ProviderError,
)
from trainer.providers.tiingo import TiingoClient

__all__ = [
    "CatalystDataProvider",
    "MarketDataProvider",
    "ProviderError",
    "TiingoClient",
]
