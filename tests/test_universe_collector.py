from pathlib import Path

from trainer.universe_collector import (
    RequestRateLimiter,
    collect_ticker_overviews,
    discover_common_stocks,
)


class FakeClient:
    def __init__(self):
        self.calls = []

    def get_tickers(self, as_of_date):
        return [
            {"ticker": "GOOD", "active": True, "type": "CS", "primary_exchange": "XNAS"},
            {"ticker": "ETF", "active": True, "type": "ETF", "primary_exchange": "XNYS"},
            {"ticker": "OTC", "active": True, "type": "CS", "primary_exchange": "PINX"},
        ]

    def get_ticker_overview(self, ticker, as_of_date):
        self.calls.append(ticker)
        values = {
            "BIG": {"ticker": ticker, "type": "CS", "primary_exchange": "XNYS", "market_cap": 20_000_000_000},
            "GOOD": {"ticker": ticker, "type": "CS", "primary_exchange": "XNAS", "market_cap": 2_000_000_000},
        }
        return values[ticker]


def test_discovery_keeps_only_primary_exchange_common_stocks():
    assert discover_common_stocks(FakeClient(), "2026-09-14") == ["GOOD"]


def test_collection_resumes_from_atomic_checkpoint(tmp_path: Path):
    path = tmp_path / "universe.json"
    client = FakeClient()
    partial = collect_ticker_overviews(
        client, ["GOOD", "BIG"], "2026-09-14", path, max_new=1
    )
    assert partial["status"] == "PARTIAL"
    assert partial["massive_plan"]["plan"] == "DEVELOPER"
    assert partial["estimated_remaining_minutes"] == 0.0
    assert partial["completed_tickers"] == ["BIG"]

    complete = collect_ticker_overviews(
        client, ["GOOD", "BIG"], "2026-09-14", path, max_new=1
    )
    assert complete["status"] == "COMPLETE"
    assert complete["eligible_tickers"] == ["GOOD"]
    assert complete["eligible_securities"] == [{
        "ticker": "GOOD",
        "primary_exchange": "XNAS",
        "market_cap_usd": 2_000_000_000,
        "as_of_date": "2026-09-14",
    }]
    assert complete["rejected"] == {"BIG": "MARKET_CAP_OUT_OF_RANGE"}
    assert client.calls == ["BIG", "GOOD"]


def test_finite_plan_rate_limiter_waits_between_calls():
    now = [100.0]
    slept = []

    def clock():
        return now[0]

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = RequestRateLimiter(5, clock=clock, sleep=sleep)
    limiter.wait()
    now[0] += 2
    limiter.wait()

    assert slept == [10.1]


def test_developer_plan_rate_limiter_does_not_sleep():
    slept = []
    limiter = RequestRateLimiter(sleep=slept.append)

    limiter.wait()
    limiter.wait()

    assert limiter.requests_per_minute is None
    assert slept == []
