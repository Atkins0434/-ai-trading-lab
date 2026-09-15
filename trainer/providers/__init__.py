"""Historical market-data provider boundaries."""

from trainer.providers.base import MarketDataProvider, ProviderError
from trainer.providers.tiingo import TiingoClient

__all__ = ["MarketDataProvider", "ProviderError", "TiingoClient"]
