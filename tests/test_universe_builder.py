from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from trainer.flatfile_snapshot import build_flatfile_snapshot
from trainer.providers.massive_flatfiles import (
    DAY_AGGS_DATASET,
    MINUTE_AGGS_DATASET,
    PADDED_SOURCE,
    SOURCE,
)
from trainer.providers.base import (
    DEFINITIVE,
    TRANSIENT,
    ClassifiedProviderError,
)
from trainer.replay_engine import (
    validate_freeze_timestamp,
    validate_point_in_time_inputs,
)
import trainer.universe_builder as universe_builder
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
        "definitive_misses": 0,
        "transient_retries": 0,
        "unresolved_failures": 0,
        "listed_after_lagged_date": 0,
    }
    assert second["source"]["query_parameters"]["ticker_overview_cache"] == {
        "hits": 6,
        "fetches": 0,
        "quarter_reuse_hits": 6,
        "errors": 0,
        "definitive_misses": 0,
        "transient_retries": 0,
        "unresolved_failures": 0,
        "listed_after_lagged_date": 0,
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


def test_serial_and_parallel_overview_fetches_build_identical_manifests(
    tmp_path: Path,
):
    retrieved_at = "2018-01-03T12:00:00+00:00"
    serial = build_point_in_time_universe(
        FakeReferenceClient(),
        FakeFlatFiles(),
        "2018-01-03",
        tmp_path / "serial" / "daily_universe_manifest.json",
        reference_cache_root=tmp_path / "serial-cache",
        reference_fetch_workers=1,
        retrieved_at=retrieved_at,
    )
    parallel = build_point_in_time_universe(
        FakeReferenceClient(),
        FakeFlatFiles(),
        "2018-01-03",
        tmp_path / "parallel" / "daily_universe_manifest.json",
        reference_cache_root=tmp_path / "parallel-cache",
        reference_fetch_workers=4,
        retrieved_at=retrieved_at,
    )

    assert parallel == serial
    assert parallel["manifest_hash"] == serial["manifest_hash"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_finite_rest_plan_forces_serial_reference_fetches(
    tmp_path: Path,
    monkeypatch,
):
    observed_workers = []
    original = universe_builder.TickerOverviewCache.get_many

    def capture_workers(self, *args, max_workers, **kwargs):
        observed_workers.append(max_workers)
        return original(
            self,
            *args,
            max_workers=max_workers,
            **kwargs,
        )

    monkeypatch.setattr(
        universe_builder,
        "load_massive_plan",
        lambda: {
            "plan": "BASIC_FREE",
            "rest_calls_per_minute": 5,
            "history_years": 2,
            "flat_files": False,
        },
    )
    monkeypatch.setattr(
        universe_builder.TickerOverviewCache,
        "get_many",
        capture_workers,
    )

    build_point_in_time_universe(
        FakeReferenceClient(),
        FakeFlatFiles(),
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        reference_cache_root=tmp_path / "reference-cache",
        reference_fetch_workers=8,
    )

    assert observed_workers == [1]


def test_smoke_cap_is_sorted_before_fetch_and_disables_evidence(
    tmp_path: Path,
):
    client = FakeReferenceClient()
    manifest = build_point_in_time_universe(
        client,
        FakeFlatFiles(),
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        reference_cache_root=tmp_path / "reference-cache",
        reference_fetch_workers=1,
        max_tickers=2,
        retrieved_at="2018-01-03T12:00:00+00:00",
    )

    assert [ticker for ticker, _ in client.overview_calls] == [
        "FUTURE",
        "OLD",
    ]
    assert len(manifest["securities"]) == 2
    assert manifest["source"]["query_parameters"]["smoke_mode"] is True
    assert manifest["source"]["query_parameters"]["max_tickers"] == 2
    assert manifest["coverage_status"] == "complete"
    assert manifest["research_evidence"] is False
    assert manifest["promotion_eligible"] is False


def _failure_fixture_flatfiles(tickers: tuple[str, ...]) -> FakeFlatFiles:
    flatfiles = FakeFlatFiles()
    flatfiles.records[(DAY_AGGS_DATASET, "2018-01-02")] = [
        bar(
            ticker,
            datetime(2018, 1, 2, 0, 0, tzinfo=ET),
            trading_date="2018-01-02",
            session="DAILY",
        )
        for ticker in tickers
    ]
    return flatfiles


def _ticker_row(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "active": True,
        "type": "CS",
        "primary_exchange": "XNAS",
        "locale": "us",
        "market": "stocks",
        "share_class_figi": f"FIGI-{ticker}",
        "list_date": "2010-01-01",
    }


def _overview(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "name": f"{ticker} OPERATING COMPANY",
        "list_date": "2010-01-01",
        "weighted_shares_outstanding": 100_000_000,
        "period_of_report_date": "2017-06-30",
    }


def _disable_reference_retry_sleeps(monkeypatch):
    original = universe_builder.TickerOverviewCache
    monkeypatch.setattr(
        universe_builder,
        "TickerOverviewCache",
        lambda root: original(
            root,
            sleep=lambda _: None,
            jitter=lambda low, high: 0.0,
        ),
    )


def test_definitive_overview_misses_and_recovered_503_keep_day_complete(
    tmp_path: Path,
    monkeypatch,
):
    tickers = ("MISSING1", "MISSING2", "RECOVERED")

    class MixedFailureClient:
        provider_name = "MASSIVE"

        def __init__(self):
            self.calls = []

        def get_tickers(self, *args, **kwargs):
            rows = [_ticker_row(ticker) for ticker in tickers]
            for row in rows:
                if row["ticker"] in {"MISSING1", "MISSING2"}:
                    row.pop("list_date")
            return rows

        def get_ticker_overview(self, ticker, as_of_date):
            self.calls.append((ticker, as_of_date))
            if ticker in {"MISSING1", "MISSING2"}:
                raise ClassifiedProviderError(
                    f"{ticker} not found",
                    classification=DEFINITIVE,
                    http_status=404,
                    exception_type="HTTPError",
                )
            if sum(call[0] == ticker for call in self.calls) == 1:
                raise ClassifiedProviderError(
                    "temporary service outage",
                    classification=TRANSIENT,
                    http_status=503,
                    exception_type="HTTPError",
                )
            return _overview(ticker)

    _disable_reference_retry_sleeps(monkeypatch)
    client = MixedFailureClient()
    cache_root = tmp_path / "reference-cache"
    manifest = build_point_in_time_universe(
        client,
        _failure_fixture_flatfiles(tickers),
        "2018-01-03",
        tmp_path / "first" / "daily_universe_manifest.json",
        reference_cache_root=cache_root,
        reference_fetch_workers=1,
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    telemetry = manifest["source"]["query_parameters"][
        "ticker_overview_cache"
    ]
    by_ticker = {item["ticker"]: item for item in manifest["securities"]}

    assert manifest["coverage_status"] == "complete"
    assert manifest["capabilities"]["provider_response_complete"] is True
    assert telemetry["definitive_misses"] == 2
    assert telemetry["transient_retries"] == 1
    assert telemetry["unresolved_failures"] == 0
    assert telemetry["errors"] == 0
    assert by_ticker["RECOVERED"]["inclusion"] is True
    for ticker in ("MISSING1", "MISSING2"):
        assert by_ticker[ticker]["inclusion"] is False
        assert by_ticker[ticker]["reason_codes"] == [
            "OVERVIEW_UNAVAILABLE_AT_LAGGED_DATE"
        ]
        assert by_ticker[ticker]["deciding_definitive_reason"] == (
            "OVERVIEW_UNAVAILABLE_AT_LAGGED_DATE"
        )
        assert by_ticker[ticker]["overview_failure"] == {
            "classification": "DEFINITIVE",
            "http_status": 404,
            "exception_type": "HTTPError",
        }

    calls_after_first = list(client.calls)
    resumed_from_negative_cache = build_point_in_time_universe(
        client,
        _failure_fixture_flatfiles(tickers),
        "2018-01-03",
        tmp_path / "second" / "daily_universe_manifest.json",
        reference_cache_root=cache_root,
        reference_fetch_workers=1,
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    assert client.calls == calls_after_first
    assert resumed_from_negative_cache["source"]["query_parameters"][
        "ticker_overview_cache"
    ]["hits"] == 3


def test_persistent_503_marks_day_incomplete_and_is_not_cached(
    tmp_path: Path,
    monkeypatch,
):
    ticker = "FAILED"

    class PersistentFailureClient:
        provider_name = "MASSIVE"

        def __init__(self):
            self.calls = 0

        def get_tickers(self, *args, **kwargs):
            return [_ticker_row(ticker)]

        def get_ticker_overview(self, ticker, as_of_date):
            self.calls += 1
            raise ClassifiedProviderError(
                "temporary service outage",
                classification=TRANSIENT,
                http_status=503,
                exception_type="HTTPError",
            )

    _disable_reference_retry_sleeps(monkeypatch)
    client = PersistentFailureClient()
    cache_root = tmp_path / "reference-cache"
    manifest = build_point_in_time_universe(
        client,
        _failure_fixture_flatfiles((ticker,)),
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        reference_cache_root=cache_root,
        reference_fetch_workers=1,
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    telemetry = manifest["source"]["query_parameters"][
        "ticker_overview_cache"
    ]
    failed = manifest["securities"][0]

    assert client.calls == 5
    assert manifest["coverage_status"] == "incomplete"
    assert manifest["capabilities"]["provider_response_complete"] is False
    assert manifest["research_evidence"] is False
    assert telemetry["definitive_misses"] == 0
    assert telemetry["transient_retries"] == 4
    assert telemetry["unresolved_failures"] == 1
    assert telemetry["errors"] == 1
    assert failed["inclusion"] is False
    assert failed["reason_codes"] == ["OVERVIEW_FETCH_FAILED"]
    assert failed["deciding_definitive_reason"] is None
    assert failed["overview_failure"] == {
        "classification": "TRANSIENT",
        "http_status": 503,
        "exception_type": "HTTPError",
    }
    assert not list(cache_root.rglob("*.json"))


def test_missing_prior_session_trade_is_definitive_exclusion(tmp_path: Path):
    ticker = "NOHISTORY"

    class NoPriorTradeClient:
        provider_name = "MASSIVE"

        def get_tickers(self, *args, **kwargs):
            return [_ticker_row(ticker)]

        def get_ticker_overview(self, ticker, as_of_date):
            return _overview(ticker)

    manifest = build_point_in_time_universe(
        NoPriorTradeClient(),
        _failure_fixture_flatfiles(()),
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        reference_cache_root=tmp_path / "reference-cache",
        reference_fetch_workers=1,
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    security = manifest["securities"][0]

    assert security["reason_codes"] == ["NO_PRIOR_SESSION_TRADE"]
    assert security["deciding_definitive_reason"] == (
        "NO_PRIOR_SESSION_TRADE"
    )
    assert manifest["coverage_status"] == "complete"
    assert manifest["coverage_reasons"] == []


def test_successful_overview_with_only_missing_listing_date_is_gap(
    tmp_path: Path,
):
    ticker = "NOLISTDATE"

    class MissingListingDateClient:
        provider_name = "MASSIVE"

        def get_tickers(self, *args, **kwargs):
            row = _ticker_row(ticker)
            row.pop("list_date")
            return [row]

        def get_ticker_overview(self, ticker, as_of_date):
            overview = _overview(ticker)
            overview.pop("list_date")
            return overview

    manifest = build_point_in_time_universe(
        MissingListingDateClient(),
        _failure_fixture_flatfiles((ticker,)),
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        reference_cache_root=tmp_path / "reference-cache",
        reference_fetch_workers=1,
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    security = manifest["securities"][0]

    assert security["reason_codes"] == ["LISTING_DATE_MISSING"]
    assert security["deciding_definitive_reason"] is None
    assert manifest["coverage_status"] == "incomplete"
    assert manifest["coverage_reasons"] == [
        f"SECURITY_METADATA_INCOMPLETE:{security['stable_security_id']}"
    ]


def test_listing_after_lagged_date_skips_overview_fetch(tmp_path: Path):
    ticker = "NEWLY"

    class NewlyListedClient:
        provider_name = "MASSIVE"

        def __init__(self):
            self.overview_calls = []

        def get_tickers(self, *args, **kwargs):
            return [{
                **_ticker_row(ticker),
                "list_date": "2017-12-01",
            }]

        def get_ticker_overview(self, ticker, as_of_date):
            self.overview_calls.append((ticker, as_of_date))
            raise AssertionError("post-lag listing must not fetch an overview")

    client = NewlyListedClient()
    manifest = build_point_in_time_universe(
        client,
        _failure_fixture_flatfiles((ticker,)),
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        reference_cache_root=tmp_path / "reference-cache",
        reference_fetch_workers=1,
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    skipped = manifest["securities"][0]
    telemetry = manifest["source"]["query_parameters"][
        "ticker_overview_cache"
    ]

    assert client.overview_calls == []
    assert manifest["coverage_status"] == "complete"
    assert manifest["capabilities"]["provider_response_complete"] is True
    assert telemetry["listed_after_lagged_date"] == 1
    assert telemetry["fetches"] == 0
    assert skipped["inclusion"] is False
    assert skipped["reason_codes"] == ["LISTED_AFTER_LAGGED_DATE"]
    assert skipped["overview_failure"] is None


def test_prior_close_above_configured_cap_is_excluded(tmp_path: Path):
    ticker = "PRICEY"

    class PriceCapClient:
        provider_name = "MASSIVE"

        def get_tickers(self, *args, **kwargs):
            return [_ticker_row(ticker)]

        def get_ticker_overview(self, ticker, as_of_date):
            return {
                **_overview(ticker),
                "weighted_shares_outstanding": 1_000_000,
            }

    flatfiles = _failure_fixture_flatfiles((ticker,))
    flatfiles.records[(DAY_AGGS_DATASET, "2018-01-02")][0] = bar(
        ticker,
        datetime(2018, 1, 2, 0, 0, tzinfo=ET),
        trading_date="2018-01-02",
        session="DAILY",
        close=501.0,
    )
    manifest = build_point_in_time_universe(
        PriceCapClient(),
        flatfiles,
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        reference_cache_root=tmp_path / "reference-cache",
        reference_fetch_workers=1,
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    security = manifest["securities"][0]

    assert security["prior_close"] == 501.0
    assert security["market_cap_usd"] == 501_000_000
    assert security["inclusion"] is False
    assert security["reason_codes"] == ["SHARE_PRICE_ABOVE_CAP"]


@pytest.mark.parametrize("issuer_field", ["cik", "composite_figi"])
def test_one_class_per_issuer_keeps_highest_prior_dollar_volume(
    tmp_path: Path,
    issuer_field: str,
):
    tickers = ("CLASSA", "CLASSB")
    issuer_value = "0001234567" if issuer_field == "cik" else "BBG000ISSUER"

    class MultipleClassClient:
        provider_name = "MASSIVE"

        def get_tickers(self, *args, **kwargs):
            return [
                {**_ticker_row(ticker), issuer_field: issuer_value}
                for ticker in tickers
            ]

        def get_ticker_overview(self, ticker, as_of_date):
            return _overview(ticker)

    flatfiles = _failure_fixture_flatfiles(tickers)
    flatfiles.records[(DAY_AGGS_DATASET, "2018-01-02")] = [
        bar(
            "CLASSA",
            datetime(2018, 1, 2, 0, 0, tzinfo=ET),
            trading_date="2018-01-02",
            session="DAILY",
            close=10.0,
            volume=1_000_000,
        ),
        bar(
            "CLASSB",
            datetime(2018, 1, 2, 0, 0, tzinfo=ET),
            trading_date="2018-01-02",
            session="DAILY",
            close=10.0,
            volume=2_000_000,
        ),
    ]
    manifest = build_point_in_time_universe(
        MultipleClassClient(),
        flatfiles,
        "2018-01-03",
        tmp_path / "daily_universe_manifest.json",
        reference_cache_root=tmp_path / "reference-cache",
        reference_fetch_workers=1,
        retrieved_at="2018-01-03T12:00:00+00:00",
    )
    by_ticker = {item["ticker"]: item for item in manifest["securities"]}

    assert by_ticker["CLASSB"]["inclusion"] is True
    assert by_ticker["CLASSB"]["prior_session_dollar_volume"] == 20_000_000
    assert by_ticker["CLASSA"]["inclusion"] is False
    assert by_ticker["CLASSA"]["reason_codes"] == [
        "DUPLICATE_SHARE_CLASS"
    ]
    assert by_ticker["CLASSA"][
        "duplicate_share_class_kept_ticker"
    ] == "CLASSB"
