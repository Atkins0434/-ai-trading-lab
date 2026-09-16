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
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"status {self.status_code}")
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
    assert report["plan_under_test"] == "DEVELOPER"
    assert report["massive_plan"] == {
        "plan": "DEVELOPER",
        "rest_calls_per_minute": None,
        "history_years": 10,
        "flat_files": True,
    }
    assert (
        report["intraday"]["premarket_0400_to_0700_et"]["record_count"]
        == 60
    )


def test_massive_api_key_is_required():
    with pytest.raises(ProviderError, match="API key is required"):
        MassiveClient("")


def test_massive_reference_endpoints_support_arrays_and_objects():
    session = FakeSession([
        {"status": "OK", "results": [{"ticker": "A", "type": "CS"}]},
        {"status": "OK", "results": {"ticker": "A", "market_cap": 1_000_000_000}},
    ])
    client = MassiveClient("secret-key", session=session)

    assert client.get_tickers("2026-09-14")[0]["ticker"] == "A"
    assert client.get_ticker_overview("a", "2026-09-14")["market_cap"] == 1_000_000_000
    assert session.calls[0][1]["params"]["type"] == "CS"
    assert session.calls[1][0].endswith("/v3/reference/tickers/A")


def test_massive_reference_ticker_query_can_include_types_for_exclusion_audit():
    session = FakeSession([
        {"status": "OK", "results": [{"ticker": "A.WS", "type": "WARRANT"}]},
    ])
    client = MassiveClient("secret-key", session=session)

    assert client.get_tickers(
        "2026-09-14", security_type=None
    )[0]["type"] == "WARRANT"
    assert "type" not in session.calls[0][1]["params"]


def test_massive_retries_throttled_request_without_leaking_key():
    from trainer.rate_control import RetryPolicy

    sleeps = []
    session = FakeSession([
        FakeResponse({"status": "ERROR"}, status_code=429, headers={"Retry-After": "0"}),
        FakeResponse({"status": "OK", "results": []}),
    ])
    # FakeSession normally wraps payloads; use a tiny session that returns responses.
    session.payloads = list(session.payloads)

    class ResponseSession:
        def __init__(self, responses):
            self.responses = responses
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return self.responses.pop(0)

    response_session = ResponseSession(session.payloads)
    client = MassiveClient(
        "secret-key",
        session=response_session,
        retry_policy=RetryPolicy(max_attempts=2, base_backoff_seconds=0),
        sleep=sleeps.append,
    )
    assert client.get_daily_prices("SPY", "2026-09-14", "2026-09-14") == []
    assert len(response_session.calls) == 2
    assert sleeps == [0]
    assert "secret-key" not in response_session.calls[0][0]
