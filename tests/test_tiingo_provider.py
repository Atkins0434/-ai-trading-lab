import pytest

from trainer.providers.base import ProviderError
from trainer.providers.tiingo import (
    TiingoClient,
    canonical_price_bar,
    market_observation,
)
from trainer.snapshot_builder import (
    admit_records_before_freeze,
    latest_market_observation,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.payload)


def test_tiingo_client_uses_token_header_and_intraday_endpoint():
    session = FakeSession([])
    client = TiingoClient("secret-token", session=session)

    assert client.get_intraday_prices(
        "AAPL",
        "2018-01-02T04:00:00-05:00",
        "2018-01-02T07:00:00-05:00",
    ) == []

    url, options = session.calls[0]
    assert url.endswith("/iex/AAPL/prices")
    assert options["headers"]["Authorization"] == "Token secret-token"
    assert options["params"]["resampleFreq"] == "1min"
    assert "secret-token" not in options["params"]


def test_future_tiingo_rows_are_not_admitted():
    records = [
        {"date": "2018-01-02T06:59:00-05:00", "close": 10.0},
        {"date": "2018-01-02T07:01:00-05:00", "close": 99.0},
    ]

    admitted = admit_records_before_freeze(
        records,
        "2018-01-02T07:00:00-05:00",
    )

    assert admitted == records[:1]


def test_latest_observation_keeps_provider_provenance():
    records = [
        {"date": "2018-01-02T06:58:00-05:00", "close": 9.5},
        {"date": "2018-01-02T06:59:00-05:00", "close": 10.0},
    ]

    result = latest_market_observation(
        records,
        "close",
        "2018-01-02T07:00:00-05:00",
    )

    assert result == market_observation(10.0, records[1], "close")
    assert result["source"] == "TIINGO"


def test_tiingo_token_is_required():
    with pytest.raises(ProviderError, match="token is required"):
        TiingoClient("")


def test_tiingo_price_row_maps_to_provider_neutral_bar():
    record = {
        "date": "2018-01-02T14:30:00Z",
        "open": 10.0,
        "high": 10.2,
        "low": 9.9,
        "close": 10.1,
        "volume": 500,
    }

    bar = canonical_price_bar(record, include_source=True)

    assert bar == {
        "timestamp": record["date"],
        "open": 10.0,
        "high": 10.2,
        "low": 9.9,
        "close": 10.1,
        "volume": 500,
        "source": "TIINGO",
    }


def test_tiingo_price_row_requires_complete_ohlcv():
    with pytest.raises(ProviderError, match="missing fields"):
        canonical_price_bar(
            {"date": "2018-01-02T14:30:00Z", "close": 10.0}
        )
