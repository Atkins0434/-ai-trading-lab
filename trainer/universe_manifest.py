from __future__ import annotations

from datetime import date, datetime, time, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trainer.providers.base import ProviderError
from trainer.rate_control import load_massive_plan
from trainer.validate_contracts import validate_contract


ET = ZoneInfo("America/New_York")
CI_FIXTURE = "ci_fixture"
HISTORICAL_RESEARCH = "historical_research"
RULESET_NAME = "us_listed_common_equity"
RULESET_VERSION = "1.0"
PRIMARY_EXCHANGES = {"XNAS", "XNYS", "XASE"}
INCLUDED_TYPES = {"CS", "COMMON_STOCK", "COMMON_EQUITY"}
TYPE_REASONS = {
    "ETF": "SECURITY_TYPE_ETF",
    "ETN": "SECURITY_TYPE_ETN",
    "PFD": "SECURITY_TYPE_PREFERRED_SHARE",
    "PREFERRED": "SECURITY_TYPE_PREFERRED_SHARE",
    "WARRANT": "SECURITY_TYPE_WARRANT",
    "WARRANTS": "SECURITY_TYPE_WARRANT",
    "RIGHT": "SECURITY_TYPE_RIGHT",
    "RIGHTS": "SECURITY_TYPE_RIGHT",
    "UNIT": "SECURITY_TYPE_UNIT",
    "UNITS": "SECURITY_TYPE_UNIT",
    "FUND": "SECURITY_TYPE_FUND",
    "MUTUAL_FUND": "SECURITY_TYPE_FUND",
}
REQUIRED_CAPABILITIES = {
    "point_in_time_listings",
    "delisted_securities",
    "stable_security_ids",
    "ticker_history",
    "point_in_time_exchange",
    "point_in_time_security_type",
    "historical_trading_status",
    "point_in_time_market_cap",
}
MIN_MARKET_CAP = 300_000_000
MAX_MARKET_CAP = 15_000_000_000


class UniverseManifestError(Exception):
    """Raised when a daily universe cannot be proved or reproduced."""


def information_cutoff(trading_date: str) -> str:
    day = date.fromisoformat(trading_date)
    return datetime.combine(day, time(7), tzinfo=ET).isoformat()


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise UniverseManifestError("Universe timestamps require a timezone.")
    return parsed


def _canonical_hash_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(manifest))
    payload.pop("manifest_hash", None)
    payload["source"].pop("retrieved_at", None)
    payload["securities"] = sorted(
        payload["securities"],
        key=lambda item: (
            item["stable_security_id"],
            item["ticker"],
            item["inclusion"],
        ),
    )
    return payload


def calculate_manifest_hash(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_hash_payload(manifest),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def verify_manifest(manifest: dict[str, Any]) -> None:
    validate_contract("daily_universe_manifest", manifest)
    expected = calculate_manifest_hash(manifest)
    if manifest["manifest_hash"] != expected:
        raise UniverseManifestError(
            "Daily universe manifest hash does not match its canonical content."
        )


def evidence_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "massive_plan": manifest["massive_plan"],
        "universe_mode": manifest["universe_mode"],
        "universe_manifest_hash": manifest["manifest_hash"],
        "universe_coverage": manifest["coverage_status"],
        "research_evidence": manifest["research_evidence"],
        "promotion_eligible": manifest["promotion_eligible"],
    }


def _known_by_cutoff(record: dict[str, Any], field: str, cutoff: datetime) -> bool:
    available = record.get(f"{field}_available_at")
    if available is None:
        return False
    return _parse_timestamp(available) <= cutoff


def _stable_id(record: dict[str, Any]) -> str | None:
    for field in ("stable_security_id", "share_class_figi", "composite_figi"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _date_value(record: dict[str, Any], *fields: str) -> date | None:
    for field in fields:
        value = record.get(field)
        if value:
            return date.fromisoformat(str(value)[:10])
    return None


def _evaluate_record(
    record: dict[str, Any], trading_date: str, cutoff: datetime
) -> tuple[dict[str, Any], list[str]]:
    day = date.fromisoformat(trading_date)
    ticker = str(record.get("ticker") or "").upper()
    stable_id = _stable_id(record)
    venue = str(record.get("primary_exchange") or record.get("listing_venue") or "")
    security_type = str(record.get("type") or record.get("security_type") or "UNKNOWN").upper()
    listing = _date_value(record, "list_date", "listing_date")
    delisting_effective = _date_value(record, "delisting_effective_date")
    last_trading = _date_value(record, "last_trading_date", "delisted_utc")
    ticker_from = _date_value(record, "ticker_valid_from")
    ticker_through = _date_value(record, "ticker_valid_through")
    reasons: list[str] = []

    metadata_available_at = _parse_timestamp(record.get("metadata_available_at"))
    if metadata_available_at is None:
        reasons.append("METADATA_AVAILABILITY_MISSING")
    elif metadata_available_at > cutoff:
        reasons.append("FUTURE_METADATA_AFTER_CUTOFF")

    if not stable_id:
        reasons.append("STABLE_SECURITY_ID_MISSING")
        stable_id = f"MISSING:{ticker or 'UNKNOWN'}"
    if not ticker:
        reasons.append("TICKER_MISSING")
    if listing is None:
        reasons.append("LISTING_DATE_MISSING")
    elif listing > day:
        reasons.append("NOT_YET_LISTED")
    if ticker_from and ticker_from > day:
        reasons.append("TICKER_NOT_YET_VALID")
    if ticker_through and day > ticker_through:
        reasons.append("TICKER_NO_LONGER_VALID")
    delisting_known_at = _parse_timestamp(record.get("delisting_known_at"))
    delisting_is_known = (
        delisting_known_at is not None and delisting_known_at <= cutoff
    )
    if delisting_is_known and delisting_effective and day >= delisting_effective:
        reasons.append("DELISTED_EFFECTIVE")
    elif delisting_is_known and last_trading and day > last_trading:
        reasons.append("AFTER_LAST_TRADING_DATE")

    historical_active = record.get("historical_active")
    if historical_active is not True:
        reasons.append(
            "HISTORICAL_TRADING_STATUS_MISSING"
            if historical_active is None
            else "NOT_TRADABLE_ON_DATE"
        )
    if str(record.get("locale", "us")).lower() != "us":
        reasons.append("NON_US_LISTING")
    if str(record.get("market", "stocks")).lower() != "stocks":
        reasons.append("NON_EQUITY_MARKET")
    if venue not in PRIMARY_EXCHANGES:
        reasons.append("INELIGIBLE_LISTING_VENUE")
    if security_type not in INCLUDED_TYPES:
        reasons.append(TYPE_REASONS.get(security_type, "SECURITY_TYPE_NOT_COMMON_EQUITY"))
    if record.get("is_test_issue") is True:
        reasons.append("TEST_ISSUE")
    if record.get("is_otc") is True:
        reasons.append("OTC_SECURITY")
    if record.get("operating_status") in {"PLACEHOLDER", "NON_OPERATING"}:
        reasons.append("NON_OPERATING_PLACEHOLDER")

    market_cap = record.get("market_cap_usd", record.get("market_cap"))
    if not reasons:
        if not _known_by_cutoff(record, "market_cap", cutoff):
            reasons.append("MARKET_CAP_POINT_IN_TIME_UNPROVEN")
        elif not isinstance(market_cap, (int, float)):
            reasons.append("MARKET_CAP_MISSING")
        elif not MIN_MARKET_CAP <= float(market_cap) <= MAX_MARKET_CAP:
            reasons.append("MARKET_CAP_OUT_OF_RANGE")

    # Current status is deliberately recorded nowhere and never evaluated.
    # Only historical_active for the provider's as-of query is admissible.
    reasons = sorted(set(reasons))
    item = {
        "stable_security_id": stable_id,
        "ticker": ticker,
        "listing_venue": venue,
        "security_type": security_type,
        "listing_date": listing.isoformat() if listing else None,
        "delisting_date": (
            delisting_effective.isoformat()
            if delisting_is_known and delisting_effective
            else last_trading.isoformat() if delisting_is_known and last_trading else None
        ),
        "inclusion": not reasons,
        "reason_codes": reasons or ["ELIGIBLE"],
    }
    return item, reasons


def _base_manifest(
    *,
    replay_id: str,
    trading_date: str,
    universe_mode: str,
    provider: str,
    feed_version: str,
    query_parameters: dict[str, Any],
    retrieved_at: str,
) -> dict[str, Any]:
    return {
        "version": "daily_universe_manifest_v1.0",
        "replay_id": replay_id,
        "trading_date": trading_date,
        "information_cutoff": information_cutoff(trading_date),
        "timezone": "America/New_York",
        "universe_mode": universe_mode,
        "massive_plan": load_massive_plan(),
        "ruleset": {"name": RULESET_NAME, "version": RULESET_VERSION},
        "source": {
            "provider": provider,
            "feed_version": feed_version,
            "query_parameters": query_parameters,
            "retrieved_at": retrieved_at,
        },
        "coverage_status": "incomplete",
        "coverage_reasons": [],
        "securities": [],
        "eligible_symbol_count": 0,
        "excluded_symbol_count": 0,
        "manifest_hash": "",
        "research_evidence": False,
        "promotion_eligible": False,
    }


def build_fixture_manifest(
    tickers: list[str],
    trading_date: str,
    *,
    replay_id: str,
    provider: str,
    feed_version: str,
    eligible_securities: list[dict[str, Any]] | None = None,
    rejected: dict[str, str] | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    requested = sorted({ticker.upper() for ticker in tickers})
    eligible = {
        item["ticker"].upper(): item for item in (eligible_securities or [])
    }
    rejected = rejected or {}
    manifest = _base_manifest(
        replay_id=replay_id,
        trading_date=trading_date,
        universe_mode=CI_FIXTURE,
        provider=provider,
        feed_version=feed_version,
        query_parameters={"hardcoded_symbols": requested},
        retrieved_at=retrieved_at or datetime.now(timezone.utc).isoformat(),
    )
    manifest["coverage_status"] = "fixture"
    manifest["coverage_reasons"] = ["STATIC_SYMBOL_FIXTURE_NOT_RESEARCH_UNIVERSE"]
    manifest["securities"] = [
        {
            "stable_security_id": f"CI_FIXTURE:{ticker}",
            "ticker": ticker,
            "listing_venue": str(eligible.get(ticker, {}).get("primary_exchange", "UNKNOWN")),
            "security_type": "COMMON_STOCK_FIXTURE",
            "listing_date": None,
            "delisting_date": None,
            "inclusion": ticker in eligible if eligible_securities is not None else ticker not in rejected,
            "reason_codes": (
                ["CI_FIXTURE_ELIGIBLE"]
                if (ticker in eligible if eligible_securities is not None else ticker not in rejected)
                else [rejected.get(ticker, "CI_FIXTURE_EXCLUDED")]
            ),
        }
        for ticker in requested
    ]
    manifest["eligible_symbol_count"] = sum(item["inclusion"] for item in manifest["securities"])
    manifest["excluded_symbol_count"] = len(manifest["securities"]) - manifest["eligible_symbol_count"]
    manifest["manifest_hash"] = calculate_manifest_hash(manifest)
    verify_manifest(manifest)
    return manifest


def build_research_manifest(
    records: list[dict[str, Any]],
    trading_date: str,
    *,
    replay_id: str,
    provider: str,
    feed_version: str,
    query_parameters: dict[str, Any],
    capabilities: dict[str, bool],
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    manifest = _base_manifest(
        replay_id=replay_id,
        trading_date=trading_date,
        universe_mode=HISTORICAL_RESEARCH,
        provider=provider,
        feed_version=feed_version,
        query_parameters=query_parameters,
        retrieved_at=retrieved_at or datetime.now(timezone.utc).isoformat(),
    )
    missing_capabilities = sorted(
        capability
        for capability in REQUIRED_CAPABILITIES
        if capabilities.get(capability) is not True
    )
    cutoff = _parse_timestamp(manifest["information_cutoff"])
    assert cutoff is not None
    evaluated = [_evaluate_record(record, trading_date, cutoff) for record in records]

    # A stable identity may have several ticker-history rows; at most one row
    # may be eligible for a date. This prevents ticker changes from duplicating
    # a company while preserving distinct companies that reused the same text.
    included_by_id: dict[str, dict[str, Any]] = {}
    securities: list[dict[str, Any]] = []
    coverage_reasons = [f"PROVIDER_CAPABILITY_MISSING:{item}" for item in missing_capabilities]
    for item, reasons in evaluated:
        if item["inclusion"]:
            prior = included_by_id.get(item["stable_security_id"])
            if prior is not None:
                item["inclusion"] = False
                item["reason_codes"] = ["DUPLICATE_STABLE_SECURITY_ID_FOR_DATE"]
                coverage_reasons.append("AMBIGUOUS_TICKER_HISTORY")
            else:
                included_by_id[item["stable_security_id"]] = item
        if any(
            reason.endswith("MISSING")
            or reason.endswith("UNPROVEN")
            or reason == "FUTURE_METADATA_AFTER_CUTOFF"
            for reason in reasons
        ):
            coverage_reasons.append(
                f"SECURITY_METADATA_INCOMPLETE:{item['stable_security_id']}"
            )
        securities.append(item)

    manifest["securities"] = sorted(
        securities, key=lambda item: (item["stable_security_id"], item["ticker"])
    )
    manifest["eligible_symbol_count"] = sum(item["inclusion"] for item in securities)
    manifest["excluded_symbol_count"] = len(securities) - manifest["eligible_symbol_count"]
    if not records:
        coverage_reasons.append("EMPTY_PROVIDER_UNIVERSE")
    manifest["coverage_reasons"] = sorted(set(coverage_reasons))
    manifest["coverage_status"] = "complete" if not coverage_reasons else "incomplete"
    manifest["research_evidence"] = manifest["coverage_status"] == "complete"
    manifest["promotion_eligible"] = manifest["research_evidence"]
    manifest["manifest_hash"] = calculate_manifest_hash(manifest)
    verify_manifest(manifest)
    return manifest


def load_or_resolve_manifest(
    path: Path,
    *,
    provider: Any,
    trading_date: str,
    replay_id: str,
    universe_mode: str,
    fixture_tickers: list[str] | None = None,
    fixture_universe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if universe_mode not in {CI_FIXTURE, HISTORICAL_RESEARCH}:
        raise UniverseManifestError(f"Unknown universe mode: {universe_mode}")
    if universe_mode == CI_FIXTURE and not fixture_tickers:
        raise UniverseManifestError("ci_fixture mode requires explicit fixture symbols.")
    if universe_mode == HISTORICAL_RESEARCH and fixture_tickers:
        raise UniverseManifestError(
            "Hardcoded symbols are forbidden in historical_research mode."
        )
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("massive_plan") != load_massive_plan():
            raise UniverseManifestError(
                "Existing universe manifest uses a different Massive plan."
            )
        verify_manifest(manifest)
        if (
            manifest["replay_id"] != replay_id
            or manifest["trading_date"] != trading_date
            or manifest["universe_mode"] != universe_mode
        ):
            raise UniverseManifestError(
                "Existing universe manifest does not match this replay identity."
            )
        if manifest["ruleset"] != {
            "name": RULESET_NAME,
            "version": RULESET_VERSION,
        }:
            raise UniverseManifestError("Existing manifest uses a different ruleset.")
        if (
            manifest["source"]["provider"] != provider.provider_name
            or manifest["source"]["feed_version"] != provider.feed_version
        ):
            raise UniverseManifestError(
                "Existing manifest source does not match the configured provider."
            )
        if universe_mode == CI_FIXTURE:
            expected = sorted({ticker.upper() for ticker in fixture_tickers or []})
            actual = manifest["source"]["query_parameters"].get("hardcoded_symbols")
            if actual != expected:
                raise UniverseManifestError(
                    "Existing fixture manifest does not match requested symbols."
                )
        return manifest

    if universe_mode == CI_FIXTURE:
        universe = fixture_universe or {}
        manifest = build_fixture_manifest(
            fixture_tickers or [],
            trading_date,
            replay_id=replay_id,
            provider=provider.provider_name,
            feed_version=provider.feed_version,
            eligible_securities=universe.get("eligible_securities"),
            rejected=universe.get("rejected"),
        )
    else:
        capabilities = provider.historical_universe_capabilities()
        missing = [name for name in REQUIRED_CAPABILITIES if not capabilities.get(name)]
        if missing:
            payload = {
                "records": [],
                "query_parameters": {
                    "date": trading_date,
                    "cutoff": information_cutoff(trading_date),
                    "request_skipped": "provider_capability_gate",
                },
            }
        else:
            try:
                payload = provider.get_historical_universe(
                    trading_date, information_cutoff(trading_date)
                )
            except ProviderError as exc:
                payload = {
                    "records": [],
                    "query_parameters": {
                        "date": trading_date,
                        "cutoff": information_cutoff(trading_date),
                        "provider_error": str(exc),
                    },
                }
                capabilities = {**capabilities, "provider_response_complete": False}
        manifest = build_research_manifest(
            payload.get("records", []),
            trading_date,
            replay_id=replay_id,
            provider=provider.provider_name,
            feed_version=provider.feed_version,
            query_parameters=payload.get("query_parameters", {}),
            capabilities=capabilities,
        )
        if capabilities.get("provider_response_complete") is False:
            manifest["coverage_reasons"] = sorted(
                set(manifest["coverage_reasons"] + ["PROVIDER_RESPONSE_INCOMPLETE"])
            )
            manifest["coverage_status"] = "incomplete"
            manifest["research_evidence"] = False
            manifest["promotion_eligible"] = False
            manifest["manifest_hash"] = calculate_manifest_hash(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return manifest
