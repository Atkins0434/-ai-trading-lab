from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trainer.flatfile_snapshot import build_flatfile_snapshot
from trainer.providers.massive_flatfiles import (
    DAY_AGGS_DATASET,
    MINUTE_AGGS_DATASET,
    PADDED_SOURCE,
    SOURCE,
)
from trainer.replay_engine import (
    validate_freeze_timestamp,
    validate_point_in_time_inputs,
)
from trainer.universe_builder import build_point_in_time_universe


ET = ZoneInfo("America/New_York")


def bar(
    ticker: str,
    timestamp: datetime,
    *,
    trading_date: str,
    session: str,
    close: float = 10.0,
    volume: float = 1_000_000,
) -> dict:
    return {
        "ticker": ticker,
        "timestamp": timestamp.isoformat(),
        "trading_date": trading_date,
        "open": close,
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "volume": volume,
        "transactions": 10,
        "session": session,
        "source": SOURCE,
    }


class FakeFlatFiles:
    provider_name = "MASSIVE"
    feed_version = "massive_sip_flatfiles_v1"

    def __init__(self):
        self.records = {
            (DAY_AGGS_DATASET, "2018-01-02"): [
                bar(
                    "OLD",
                    datetime(2018, 1, 2, 0, 0, tzinfo=ET),
                    trading_date="2018-01-02",
                    session="DAILY",
                ),
                bar(
                    "FUTURE",
                    datetime(2018, 1, 2, 0, 0, tzinfo=ET),
                    trading_date="2018-01-02",
                    session="DAILY",
                ),
                *[
                    bar(
                        ticker,
                        datetime(2018, 1, 2, 0, 0, tzinfo=ET),
                        trading_date="2018-01-02",
                        session="DAILY",
                    )
                    for ticker in ("WARRANT", "UNIT", "SPAC.U", "SHELL")
                ],
            ],
            (MINUTE_AGGS_DATASET, "2018-01-02"): [
                bar(
                    "OLD",
                    datetime(2018, 1, 2, 6, 0, tzinfo=ET),
                    trading_date="2018-01-02",
                    session="PREMARKET",
                    volume=1_000,
                )
            ],
            (MINUTE_AGGS_DATASET, "2018-01-03"): [
                bar(
                    "OLD",
                    datetime(2018, 1, 3, 6, 0, tzinfo=ET),
                    trading_date="2018-01-03",
                    session="PREMARKET",
                    close=10.2,
                    volume=2_000,
                ),
                bar(
                    "OLD",
                    datetime(2018, 1, 3, 9, 30, tzinfo=ET),
                    trading_date="2018-01-03",
                    session="REGULAR",
                    close=10.3,
                    volume=5_000,
                ),
            ],
        }

    def iter_bars(self, dataset, trading_date, tickers=None):
        allowed = {value.upper() for value in tickers} if tickers else None
        for item in self.records.get((dataset, trading_date), []):
            if allowed is None or item["ticker"] in allowed:
                yield item


class FakeReferenceClient:
    provider_name = "MASSIVE"

    def __init__(self):
        self.calls = []

    def get_tickers(self, as_of_date, *, active=True, security_type="CS"):
        self.calls.append((as_of_date, active, security_type))
        return [
            {
                "ticker": "OLD",
                "active": False,
                "type": "CS",
                "primary_exchange": "XNAS",
                "locale": "us",
                "market": "stocks",
                "share_class_figi": "FIGI-OLD",
                "delisted_utc": "2019-06-01",
            },
            {
                "ticker": "FUTURE",
                "active": True,
                "type": "CS",
                "primary_exchange": "XNAS",
                "locale": "us",
                "market": "stocks",
                "share_class_figi": "FIGI-FUTURE",
            },
            *[
                {
                    "ticker": ticker,
                    "active": True,
                    "type": security_type,
                    "primary_exchange": "XNAS",
                    "locale": "us",
                    "market": "stocks",
                    "share_class_figi": f"FIGI-{ticker}",
                }
                for ticker, security_type in (
                    ("WARRANT", "WARRANT"),
                    ("UNIT", "UNIT"),
                    ("SPAC.U", "CS"),
                    ("SHELL", "CS"),
                )
            ],
        ]

    def get_ticker_overview(self, ticker, as_of_date):
        result = {
            "ticker": ticker,
            "name": f"{ticker} OPERATING COMPANY",
            "list_date": "2010-01-01" if ticker == "OLD" else "2019-01-01",
            "weighted_shares_outstanding": 100_000_000,
            "shares_outstanding_available_at": "2018-01-02T17:00:00-05:00",
            "is_shell": False,
        }
        if ticker not in {"FUTURE"}:
            result["list_date"] = "2010-01-01"
        if ticker == "SHELL":
            result["is_shell"] = True
        return result


def build_manifest(tmp_path: Path):
    return build_point_in_time_universe(
        FakeReferenceClient(),
        FakeFlatFiles(),
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        retrieved_at="2018-01-03T12:00:00+00:00",
    )


def test_universe_uses_date_scoped_status_and_listing_boundary(tmp_path: Path):
    client = FakeReferenceClient()
    path = tmp_path / "daily_universe_manifest.json"
    manifest = build_point_in_time_universe(
        client,
        FakeFlatFiles(),
        "2018-01-03",
        path,
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    by_ticker = {item["ticker"]: item for item in manifest["securities"]}

    assert client.calls == [("2018-01-03", True, None)]
    assert by_ticker["OLD"]["inclusion"] is True
    assert by_ticker["OLD"]["delisting_date"] is None
    assert by_ticker["OLD"]["market_cap_usd"] == 1_000_000_000
    assert by_ticker["FUTURE"]["inclusion"] is False
    assert "NOT_YET_LISTED" in by_ticker["FUTURE"]["reason_codes"]
    assert "SECURITY_TYPE_WARRANT" in by_ticker["WARRANT"]["reason_codes"]
    assert "SECURITY_TYPE_UNIT" in by_ticker["UNIT"]["reason_codes"]
    assert "SPAC_SUFFIX_SECURITY" in by_ticker["SPAC.U"]["reason_codes"]
    assert "SHELL_COMPANY" in by_ticker["SHELL"]["reason_codes"]
    assert all(
        item["reference_data_as_of_timestamp"]
        == "2018-01-03T00:00:00-05:00"
        for item in manifest["securities"]
    )
    assert manifest["coverage_status"] == "complete"
    assert path.exists()


def test_flatfile_snapshot_preserves_sources_and_passes_freeze_checks(tmp_path: Path):
    flatfiles = FakeFlatFiles()
    manifest = build_point_in_time_universe(
        FakeReferenceClient(),
        flatfiles,
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        retrieved_at="2018-01-03T12:00:00+00:00",
    )

    result = build_flatfile_snapshot(
        "2018-01-03",
        manifest,
        flatfiles,
        lookback_sessions=1,
    )
    security = result.snapshot["securities"][0]

    validate_freeze_timestamp(result.snapshot)
    validate_point_in_time_inputs(result.snapshot)
    assert security["premarket_bars"][0]["source"] == SOURCE
    assert security["premarket_bars"][1]["source"] == PADDED_SOURCE
    assert len(security["premarket_bars"]) == 60
    assert result.padded_bar_statistics["premarket_padded_bar_count"] == 59
    assert result.outcome_bars["OLD"][0]["source"] == SOURCE


def test_universe_resume_reuses_the_immutable_manifest(tmp_path: Path):
    path = tmp_path / "daily_universe_manifest.json"
    first = build_point_in_time_universe(
        FakeReferenceClient(),
        FakeFlatFiles(),
        "2018-01-03",
        path,
        retrieved_at="2018-01-03T12:00:00+00:00",
    )

    class FailingReferenceClient:
        provider_name = "MASSIVE"

        def get_tickers(self, *args, **kwargs):
            raise AssertionError("resume must not query newer reference state")

    resumed = build_point_in_time_universe(
        FailingReferenceClient(),
        FakeFlatFiles(),
        "2018-01-03",
        path,
    )

    assert resumed["manifest_hash"] == first["manifest_hash"]


def test_unproven_shares_outstanding_availability_fails_closed(tmp_path: Path):
    class UnprovenReferenceClient(FakeReferenceClient):
        def get_ticker_overview(self, ticker, as_of_date):
            result = super().get_ticker_overview(ticker, as_of_date)
            result.pop("shares_outstanding_available_at", None)
            return result

    manifest = build_point_in_time_universe(
        UnprovenReferenceClient(),
        FakeFlatFiles(),
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    old = next(
        item for item in manifest["securities"] if item["ticker"] == "OLD"
    )

    assert old["inclusion"] is False
    assert "SHARES_OUTSTANDING_AVAILABILITY_UNPROVEN" in old["reason_codes"]
    assert manifest["coverage_status"] == "incomplete"
    assert manifest["research_evidence"] is False
