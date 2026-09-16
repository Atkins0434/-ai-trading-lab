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
        self.overview_calls = []

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
        self.overview_calls.append((ticker, as_of_date))
        result = {
            "ticker": ticker,
            "name": f"{ticker} OPERATING COMPANY",
            "list_date": "2010-01-01" if ticker == "OLD" else "2019-01-01",
            "weighted_shares_outstanding": 100_000_000,
            "period_of_report_date": "2017-12-01",
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
        reference_cache_root=tmp_path / "reference_cache",
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
        reference_cache_root=tmp_path / "reference_cache",
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    by_ticker = {item["ticker"]: item for item in manifest["securities"]}

    assert client.calls == [("2018-01-03", True, None)]
    assert all(
        lagged_date == "2017-09-05"
        for _, lagged_date in client.overview_calls
    )
    assert by_ticker["OLD"]["inclusion"] is True
    assert by_ticker["OLD"]["delisting_date"] is None
    assert by_ticker["OLD"]["market_cap_usd"] == 1_000_000_000
    assert by_ticker["OLD"]["shares_outstanding_lag_days"] == 120
    assert by_ticker["OLD"]["shares_outstanding_lagged_date"] == "2017-09-05"
    assert by_ticker["OLD"]["shares_outstanding_provider_period_date"] == (
        "2017-12-01"
    )
    assert by_ticker["FUTURE"]["inclusion"] is False
    assert "NOT_YET_LISTED" in by_ticker["FUTURE"]["reason_codes"]
    assert "SECURITY_TYPE_WARRANT" in by_ticker["WARRANT"]["reason_codes"]
    assert "SECURITY_TYPE_UNIT" in by_ticker["UNIT"]["reason_codes"]
    assert "SPAC_SUFFIX_SECURITY" in by_ticker["SPAC.U"]["reason_codes"]
    assert "SHELL_COMPANY" in by_ticker["SHELL"]["reason_codes"]
    assert "SHELL_STATUS_UNPROVEN" not in by_ticker["OLD"]["reason_codes"]
    assert all(
        item["reference_data_as_of_timestamp"]
        == "2018-01-03T00:00:00-05:00"
        for item in manifest["securities"]
    )
    assert manifest["coverage_status"] == "complete"
    assert manifest["capabilities"]["point_in_time_market_cap"] == (
        "LAGGED_PROXY"
    )
    assert path.exists()


def test_flatfile_snapshot_preserves_sources_and_passes_freeze_checks(tmp_path: Path):
    flatfiles = FakeFlatFiles()
    manifest = build_point_in_time_universe(
        FakeReferenceClient(),
        flatfiles,
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        reference_cache_root=tmp_path / "reference_cache",
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
        reference_cache_root=tmp_path / "reference_cache",
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
        reference_cache_root=tmp_path / "reference_cache",
    )

    assert resumed["manifest_hash"] == first["manifest_hash"]


def test_lagged_count_is_admitted_despite_later_provider_period_date(
    tmp_path: Path,
):
    client = FakeReferenceClient()
    manifest = build_point_in_time_universe(
        client,
        FakeFlatFiles(),
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        reference_cache_root=tmp_path / "reference_cache",
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    old = next(
        item for item in manifest["securities"] if item["ticker"] == "OLD"
    )

    assert client.overview_calls[0][1] == "2017-09-05"
    assert old["shares_outstanding_provider_period_date"] == "2017-12-01"
    assert old["shares_outstanding"] == 100_000_000
    assert old["inclusion"] is True
    assert manifest["coverage_status"] == "complete"
    assert manifest["research_evidence"] is True


def test_quarter_bucket_cache_reuses_first_lagged_fetch(tmp_path: Path):
    client = FakeReferenceClient()
    flatfiles = FakeFlatFiles()
    flatfiles.records[(DAY_AGGS_DATASET, "2018-01-09")] = [
        bar(
            ticker,
            datetime(2018, 1, 9, 0, 0, tzinfo=ET),
            trading_date="2018-01-09",
            session="DAILY",
        )
        for ticker in ("OLD", "FUTURE", "WARRANT", "UNIT", "SPAC.U", "SHELL")
    ]
    cache_root = tmp_path / "reference_cache"

    first = build_point_in_time_universe(
        client,
        flatfiles,
        "2018-01-03",
        tmp_path / "first" / "daily_universe_manifest.json",
        reference_cache_root=cache_root,
    )
    calls_after_first = list(client.overview_calls)
    second = build_point_in_time_universe(
        client,
        flatfiles,
        "2018-01-10",
        tmp_path / "second" / "daily_universe_manifest.json",
        reference_cache_root=cache_root,
    )

    assert len(calls_after_first) == 6
    assert client.overview_calls == calls_after_first
    assert first["source"]["query_parameters"]["ticker_overview_cache"] == {
        "hits": 0,
        "fetches": 6,
        "quarter_reuse_hits": 0,
        "errors": 0,
    }
    assert second["source"]["query_parameters"]["ticker_overview_cache"] == {
        "hits": 6,
        "fetches": 0,
        "quarter_reuse_hits": 6,
        "errors": 0,
    }
    old = next(
        item for item in second["securities"] if item["ticker"] == "OLD"
    )
    assert old["shares_outstanding_lagged_date"] == "2017-09-12"
    assert old["shares_outstanding_provider_query_date"] == "2017-09-05"


def test_cache_telemetry_does_not_change_universe_hash(tmp_path: Path):
    client = FakeReferenceClient()
    cache_root = tmp_path / "reference_cache"
    first = build_point_in_time_universe(
        client,
        FakeFlatFiles(),
        "2018-01-03",
        tmp_path / "first" / "daily_universe_manifest.json",
        reference_cache_root=cache_root,
    )
    second = build_point_in_time_universe(
        client,
        FakeFlatFiles(),
        "2018-01-03",
        tmp_path / "second" / "daily_universe_manifest.json",
        reference_cache_root=cache_root,
    )

    assert first["manifest_hash"] == second["manifest_hash"]
    assert first["source"]["query_parameters"]["ticker_overview_cache"][
        "fetches"
    ] == 6
    assert second["source"]["query_parameters"]["ticker_overview_cache"][
        "hits"
    ] == 6


def test_shell_flag_is_optional_but_true_shell_is_excluded(tmp_path: Path):
    manifest = build_manifest(tmp_path)
    by_ticker = {item["ticker"]: item for item in manifest["securities"]}

    assert by_ticker["OLD"]["inclusion"] is True
    assert by_ticker["SHELL"]["inclusion"] is False
    assert "SHELL_COMPANY" in by_ticker["SHELL"]["reason_codes"]
