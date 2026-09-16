from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainer.benchmark import BenchmarkError, build_same_universe_benchmark
from trainer.evidence_eligibility import (
    EvidenceEligibilityError,
    require_research_evidence,
)
from trainer.multi_day_trainer import run_multi_day_trainer
from trainer.promotion_gate import validate_promotion_evidence
from trainer.providers.base import ProviderError
from trainer.replay_queue import ReplayQueue
from trainer.universe_manifest import (
    REQUIRED_CAPABILITIES,
    build_fixture_manifest,
    build_research_manifest,
    load_or_resolve_manifest,
)


DAY = "2024-06-10"
HASH = "sha256:" + "a" * 64


def capabilities(**overrides):
    result = {name: True for name in REQUIRED_CAPABILITIES}
    result["provider_response_complete"] = True
    result.update(overrides)
    return result


def security(**overrides):
    result = {
        "stable_security_id": "FIGI-ONE",
        "ticker": "OLD",
        "primary_exchange": "XNAS",
        "type": "CS",
        "locale": "us",
        "market": "stocks",
        "list_date": "2015-01-01",
        "historical_active": True,
        "current_active": True,
        "metadata_available_at": "2024-06-09T20:00:00-04:00",
        "market_cap_usd": 1_000_000_000,
        "market_cap_available_at": "2024-06-09T20:00:00-04:00",
    }
    result.update(overrides)
    return result


def manifest(records, trading_date=DAY, **kwargs):
    return build_research_manifest(
        records,
        trading_date,
        replay_id=f"replay-{trading_date}",
        provider="SYNTHETIC_SECURITY_MASTER",
        feed_version="synthetic_v1",
        query_parameters={"date": trading_date, "active": "all"},
        capabilities=capabilities(),
        retrieved_at="2026-01-01T00:00:00+00:00",
        **kwargs,
    )


def by_id(payload, stable_id):
    return next(
        item for item in payload["securities"]
        if item["stable_security_id"] == stable_id
    )


def evidence(mode="historical_research", coverage="complete", enabled=True):
    return {
        "universe_mode": mode,
        "universe_manifest_hash": HASH,
        "universe_coverage": coverage,
        "research_evidence": enabled,
        "promotion_eligible": enabled,
    }


def test_01_security_listed_after_replay_date_is_excluded():
    result = manifest([security(list_date="2024-06-11")])
    assert result["massive_plan"]["plan"] == "DEVELOPER"
    assert by_id(result, "FIGI-ONE")["inclusion"] is False
    assert "NOT_YET_LISTED" in by_id(result, "FIGI-ONE")["reason_codes"]


def test_02_later_delisted_security_is_included_before_delisting():
    result = manifest([security(
        current_active=False,
        delisting_effective_date="2024-07-01",
        delisting_known_at="2024-07-01T00:00:00-04:00",
    )])
    item = by_id(result, "FIGI-ONE")
    assert item["inclusion"] is True
    assert item["delisting_date"] is None  # future event is not leaked backward


def test_03_security_is_excluded_at_effective_delisting_boundary():
    result = manifest([security(
        historical_active=False,
        delisting_effective_date=DAY,
        delisting_known_at="2024-06-10T06:00:00-04:00",
    )])
    assert "DELISTED_EFFECTIVE" in by_id(result, "FIGI-ONE")["reason_codes"]


def test_04_current_inactive_flag_does_not_exclude_historically_active_security():
    result = manifest([security(current_active=False, historical_active=True)])
    assert by_id(result, "FIGI-ONE")["inclusion"] is True


def test_05_current_active_flag_does_not_include_before_listing():
    result = manifest([security(current_active=True, list_date="2025-01-01")])
    assert by_id(result, "FIGI-ONE")["inclusion"] is False


def test_06_ticker_change_preserves_identity_without_duplicate_company():
    old = security(
        ticker="OLD",
        ticker_valid_through="2024-05-31",
        historical_active=False,
    )
    new = security(
        ticker="NEW",
        ticker_valid_from="2024-06-01",
    )
    result = manifest([old, new])
    included = [item for item in result["securities"] if item["inclusion"]]
    assert [(item["stable_security_id"], item["ticker"]) for item in included] == [
        ("FIGI-ONE", "NEW")
    ]


def test_07_reused_ticker_is_distinguished_by_stable_identity():
    prior = security(
        stable_security_id="FIGI-OLD-COMPANY",
        ticker="REUSE",
        historical_active=False,
        delisting_effective_date="2020-01-01",
        delisting_known_at="2020-01-01T00:00:00-05:00",
    )
    current = security(
        stable_security_id="FIGI-NEW-COMPANY",
        ticker="REUSE",
        list_date="2023-01-01",
    )
    result = manifest([prior, current])
    assert {item["stable_security_id"] for item in result["securities"]} == {
        "FIGI-OLD-COMPANY", "FIGI-NEW-COMPANY"
    }
    assert by_id(result, "FIGI-NEW-COMPANY")["inclusion"] is True


def test_08_excluded_security_types_have_deterministic_reason_codes():
    expected = {
        "ETF": "SECURITY_TYPE_ETF",
        "ETN": "SECURITY_TYPE_ETN",
        "PFD": "SECURITY_TYPE_PREFERRED_SHARE",
        "WARRANT": "SECURITY_TYPE_WARRANT",
        "RIGHT": "SECURITY_TYPE_RIGHT",
        "UNIT": "SECURITY_TYPE_UNIT",
        "FUND": "SECURITY_TYPE_FUND",
    }
    records = [
        security(stable_security_id=f"ID-{kind}", ticker=kind, type=kind)
        for kind in expected
    ]
    result = manifest(records)
    for kind, reason in expected.items():
        assert by_id(result, f"ID-{kind}")["reason_codes"] == [reason]


def test_09_information_after_cutoff_is_never_admitted():
    result = manifest([security(
        market_cap_available_at="2024-06-10T07:00:01-04:00"
    )])
    assert result["coverage_status"] == "incomplete"
    assert "MARKET_CAP_POINT_IN_TIME_UNPROVEN" in by_id(
        result, "FIGI-ONE"
    )["reason_codes"]


class SyntheticProvider:
    provider_name = "SYNTHETIC_SECURITY_MASTER"
    feed_version = "synthetic_v1"

    def __init__(self, records=None, error=False):
        self.records = records or [security()]
        self.error = error
        self.calls = 0

    def historical_universe_capabilities(self):
        return capabilities()

    def get_historical_universe(self, trading_date, cutoff):
        self.calls += 1
        if self.error:
            raise ProviderError("synthetic outage")
        return {
            "records": self.records,
            "query_parameters": {"date": trading_date, "cutoff": cutoff},
        }


def test_10_resume_reuses_manifest_hash_and_deduplicates_jobs(tmp_path: Path):
    provider = SyntheticProvider()
    path = tmp_path / "daily_universe_manifest.json"
    first = load_or_resolve_manifest(
        path, provider=provider, trading_date=DAY, replay_id="resume-test",
        universe_mode="historical_research",
    )
    second = load_or_resolve_manifest(
        path, provider=provider, trading_date=DAY, replay_id="resume-test",
        universe_mode="historical_research",
    )
    queue = ReplayQueue(tmp_path / "queue.json")
    item = (DAY, "OLD", "TICKER_REPLAY", "DEVELOPMENT")
    queue.enqueue([item])
    queue.enqueue([item])
    assert first["manifest_hash"] == second["manifest_hash"]
    assert provider.calls == 1
    assert len(queue.snapshot()["tasks"]) == 1


def test_11_scout_and_benchmark_require_identical_manifest_hashes():
    shared = evidence()
    snapshot = {"replay_id": "r", "trading_date": DAY, "universe_version": "u", **shared}
    scout = {"scout_version": "s", "candidates": [], **shared}
    outcome = {"execution_policy_version": "p", "outcomes": [], **shared}
    result = build_same_universe_benchmark(snapshot, scout, outcome, 2500.0)
    assert result["universe_manifest_hash"] == HASH
    outcome["universe_manifest_hash"] = "sha256:" + "b" * 64
    with pytest.raises(BenchmarkError, match="different universe manifests"):
        build_same_universe_benchmark(snapshot, scout, outcome, 2500.0)


def test_12_trainer_rejects_ci_fixture_artifact():
    with pytest.raises(EvidenceEligibilityError, match="NOT_HISTORICAL_RESEARCH"):
        require_research_evidence(
            evidence(mode="ci_fixture", coverage="fixture", enabled=False),
            consumer="Trainer",
        )


def test_13_trainer_rejects_incomplete_historical_coverage():
    with pytest.raises(EvidenceEligibilityError, match="COVERAGE_INCOMPLETE"):
        require_research_evidence(
            evidence(coverage="incomplete", enabled=False), consumer="Trainer"
        )


def test_14_promotion_rejects_noneligible_evidence():
    artifact = evidence()
    artifact["promotion_eligible"] = False
    with pytest.raises(EvidenceEligibilityError, match="PROMOTION_ELIGIBLE_FALSE"):
        validate_promotion_evidence([artifact])


def test_15_provider_failure_does_not_fall_back_to_static_symbols(tmp_path: Path):
    provider = SyntheticProvider(error=True)
    result = load_or_resolve_manifest(
        tmp_path / "manifest.json",
        provider=provider,
        trading_date=DAY,
        replay_id="provider-failure",
        universe_mode="historical_research",
    )
    assert result["coverage_status"] == "incomplete"
    assert result["eligible_symbol_count"] == 0
    assert result["research_evidence"] is False
    assert "PROVIDER_RESPONSE_INCOMPLETE" in result["coverage_reasons"]


def test_16_cumulative_report_quarantines_fixture_days(tmp_path: Path):
    def fixture_runner(*args, **kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "status": "COMPLETE",
            "scored_tickers": ["SMOKE"],
            "skipped": {},
            **evidence(mode="ci_fixture", coverage="fixture", enabled=False),
        }

    state = run_multi_day_trainer(
        object(), None, [DAY], cache_root=tmp_path / "cache",
        output_root=tmp_path / "trainer", day_runner=fixture_runner,
        universe_mode="historical_research",
    )
    assert state["aggregate_performance"]["days_processed"] == 0
    assert state["quarantined_days"] == [DAY]
    assert state["research_evidence"] is False
    registry = json.loads(
        (tmp_path / "trainer" / "LEGACY_RESEARCH_QUARANTINE.json").read_text()
    )
    assert any(
        entry["artifact"] == "trainer_run_state.json"
        for entry in registry["entries"]
    )


def test_17_deterministic_ci_fixture_remains_available_for_engineering():
    first = build_fixture_manifest(
        ["AAPL", "MSFT"], DAY, replay_id="fixture", provider="SYNTHETIC",
        feed_version="v1", retrieved_at="2026-01-01T00:00:00+00:00",
    )
    second = build_fixture_manifest(
        ["MSFT", "AAPL"], DAY, replay_id="fixture", provider="SYNTHETIC",
        feed_version="v1", retrieved_at="2027-01-01T00:00:00+00:00",
    )
    assert first["manifest_hash"] == second["manifest_hash"]
    assert first["universe_mode"] == "ci_fixture"
    assert first["research_evidence"] is False
    assert first["promotion_eligible"] is False
