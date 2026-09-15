from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trainer.massive_capability_report import build_report
from trainer.providers.base import ProviderError
from trainer.providers.massive import (
    MassiveClient,
    canonical_price_bar,
    parse_massive_timestamp,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.payloads.pop(0))


def aggregate(timestamp: datetime, price: float = 10.0) -> dict:
    return {
        "t": int(timestamp.timestamp() * 1000),
        "o": price,
        "h": price + 0.1,
        "l": price - 0.1,
        "c": price + 0.05,
        "v": 1000,
    }


def test_massive_uses_bearer_header_and_aggregate_endpoint():
    session = FakeSession([{"status": "OK", "results": []}])
    client = MassiveClient("secret-key", session=session)

    assert client.get_intraday_prices(
        "spy",
        "2026-09-14",
        "2026-09-15",
    ) == []

    url, options = session.calls[0]
    assert url.endswith(
        "/v2/aggs/ticker/SPY/range/1/minute/2026-09-14/2026-09-15"
    )
    assert options["headers"]["Authorization"] == "Bearer secret-key"
    assert "secret-key" not in url
    assert "secret-key" not in options["params"]


def test_massive_follows_only_same_origin_pagination():
    session = FakeSession(
        [
            {
                "status": "OK",
                "results": [{"id": 1}],
                "next_url": "https://api.massive.com/next?cursor=abc",
            },
            {"status": "OK", "results": [{"id": 2}]},
        ]
    )
    client = MassiveClient("secret-key", session=session)

    assert client.get_news(["SPY"], "2026-09-14", "2026-09-15") == [
        {"id": 1},
        {"id": 2},
    ]
    assert len(session.calls) == 2


def test_massive_rejects_cross_origin_pagination():
    session = FakeSession(
        [
            {
                "status": "OK",
                "results": [],
                "next_url": "https://example.com/steal",
            }
        ]
    )
    with pytest.raises(ProviderError, match="changed origin"):
        MassiveClient("secret-key", session=session).get_news(
            ["SPY"], "2026-09-14", "2026-09-15"
        )


def test_massive_aggregate_maps_to_canonical_bar():
    observed = datetime(2026, 9, 14, 8, 0, tzinfo=timezone.utc)
    record = aggregate(observed)

    assert parse_massive_timestamp(record) == observed.isoformat()
    assert canonical_price_bar(record, include_source=True) == {
        "timestamp": observed.isoformat(),
        "open": 10.0,
        "high": 10.1,
        "low": 9.9,
        "close": 10.05,
        "volume": 1000,
        "source": "MASSIVE",
    }


class FakeCapabilityClient:
    provider_name = "MASSIVE"
    feed_version = "massive_rest_v2_aggs"

    def get_intraday_prices(self, ticker, start, end, frequency="1min"):
        start_time = datetime(2026, 9, 14, 8, 0, tzinfo=timezone.utc)
        correct_day = [
            aggregate(start_time.replace(minute=index), 10 + index / 100)
            for index in range(60)
        ]
        wrong_day = aggregate(
            datetime(2026, 9, 15, 8, 0, tzinfo=timezone.utc)
        )
        return correct_day + [wrong_day]

    def get_daily_prices(self, ticker, start, end):
        return [
            aggregate(datetime(2026, 9, 14, 4, 0, tzinfo=timezone.utc))
        ]


def test_capability_report_requires_sixty_premarket_bars():
    report = build_report(
        FakeCapabilityClient(),
        "SPY",
        "2026-09-14",
    )

    assert report["selection_contract_passed"] is True
    assert (
        report["intraday"]["premarket_0400_to_0700_et"]["record_count"]
        == 60
    )


def test_massive_api_key_is_required():
    with pytest.raises(ProviderError, match="API key is required"):
        MassiveClient("")
