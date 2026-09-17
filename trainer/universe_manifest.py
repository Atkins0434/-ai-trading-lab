from __future__ import annotations

from datetime import date, datetime, time, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trainer.providers.base import ProviderError
from trainer.rate_control import load_massive_plan
from trainer.replay_engine import configured_freeze_datetime
from trainer.validate_contracts import validate_contract


ET = ZoneInfo("America/New_York")
LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_CONFIG_PATH = ROOT / "config" / "universe.json"
CI_FIXTURE = "ci_fixture"
HISTORICAL_RESEARCH = "historical_research"
RULESET_NAME = "us_listed_common_equity"
RULESET_VERSION = "1.0"
_UNIVERSE_CONFIG = json.loads(
    UNIVERSE_CONFIG_PATH.read_text(encoding="utf-8")
)
PRIMARY_EXCHANGES = set(_UNIVERSE_CONFIG["allowed_exchanges"])
INCLUDED_TYPES = set(_UNIVERSE_CONFIG["included_security_types"])
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
# This registry is the single authority for deciding whether an exclusion is
# proven from point-in-time evidence or exposes a coverage gap. New reason
# codes must be added here deliberately; unknown codes fail manifest creation.
DEFINITIVE = "DEFINITIVE"
GAP = "GAP"
EXCLUSION_REASON_CLASSIFICATION = {
    "SECURITY_TYPE_ETF": DEFINITIVE,
    "SECURITY_TYPE_ETN": DEFINITIVE,
    "SECURITY_TYPE_PREFERRED_SHARE": DEFINITIVE,
    "SECURITY_TYPE_WARRANT": DEFINITIVE,
    "SECURITY_TYPE_RIGHT": DEFINITIVE,
    "SECURITY_TYPE_UNIT": DEFINITIVE,
    "SECURITY_TYPE_FUND": DEFINITIVE,
    "SECURITY_TYPE_NOT_COMMON_EQUITY": DEFINITIVE,
    "INELIGIBLE_LISTING_VENUE": DEFINITIVE,
    "NON_US_LISTING": DEFINITIVE,
    "NON_EQUITY_MARKET": DEFINITIVE,
    "MARKET_CAP_OUT_OF_RANGE": DEFINITIVE,
    "SPAC_SECURITY": DEFINITIVE,
    "SPAC_SUFFIX_SECURITY": DEFINITIVE,
    "SHELL_COMPANY": DEFINITIVE,
    "TEST_ISSUE": DEFINITIVE,
    "OTC_SECURITY": DEFINITIVE,
    "NON_OPERATING_PLACEHOLDER": DEFINITIVE,
    "SHARE_PRICE_ABOVE_CAP": DEFINITIVE,
    "DUPLICATE_SHARE_CLASS": DEFINITIVE,
    "DUPLICATE_STABLE_SECURITY_ID_FOR_DATE": DEFINITIVE,
    "OVERVIEW_UNAVAILABLE_AT_LAGGED_DATE": DEFINITIVE,
    "OVERVIEW_NO_SHARE_COUNT": DEFINITIVE,
    "LISTED_AFTER_LAGGED_DATE": DEFINITIVE,
    "NO_PRIOR_SESSION_TRADE": DEFINITIVE,
    "NOT_YET_LISTED": DEFINITIVE,
    "TICKER_NOT_YET_VALID": DEFINITIVE,
    "TICKER_NO_LONGER_VALID": DEFINITIVE,
    "DELISTED_EFFECTIVE": DEFINITIVE,
    "AFTER_LAST_TRADING_DATE": DEFINITIVE,
    "NOT_TRADABLE_ON_DATE": DEFINITIVE,
    "OVERVIEW_FETCH_FAILED": GAP,
    "FUTURE_METADATA_AFTER_CUTOFF": GAP,
    "LISTING_DATE_MISSING": GAP,
    "STABLE_SECURITY_ID_MISSING": DEFINITIVE,
    "MARKET_CAP_POINT_IN_TIME_UNPROVEN": GAP,
    "METADATA_AVAILABILITY_MISSING": GAP,
    "TICKER_MISSING": GAP,
    "HISTORICAL_TRADING_STATUS_MISSING": GAP,
    "MARKET_CAP_MISSING": GAP,
    "SHARES_PERIOD_AFTER_REPLAY_DATE": GAP,
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
LAGGED_MARKET_CAP_PROXY = "LAGGED_PROXY"
MIN_MARKET_CAP = float(_UNIVERSE_CONFIG["market_cap_usd"]["minimum"])
MAX_MARKET_CAP = float(_UNIVERSE_CONFIG["market_cap_usd"]["maximum"])


class UniverseManifestError(Exception):
    """Raised when a daily universe cannot be proved or reproduced."""


def _capability_available(name: str, value: Any) -> bool:
    if name == "point_in_time_market_cap":
        return value is True or value == LAGGED_MARKET_CAP_PROXY
    return value is True


def information_cutoff(
    trading_date: str, freeze_config: dict[str, Any] | None = None
) -> str:
    return configured_freeze_datetime(trading_date, freeze_config).isoformat()


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
    payload["source"]["query_parameters"].pop(
        "ticker_overview_cache", None
    )
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


def persist_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Validate and atomically persist an immutable universe manifest."""
    verify_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _classify_reasons(reasons: list[str]) -> tuple[list[str], list[str]]:
    definitive: list[str] = []
    gaps: list[str] = []
    for reason in reasons:
        classification = EXCLUSION_REASON_CLASSIFICATION.get(reason)
        if classification is None:
            raise UniverseManifestError(
                f"Unknown universe exclusion reason code: {reason}"
            )
        (definitive if classification == DEFINITIVE else gaps).append(reason)
    return definitive, gaps


def _deciding_definitive_reason(reasons: list[str]) -> str | None:
    reason_set = set(reasons)
    return next(
        (
            reason
            for reason, classification in EXCLUSION_REASON_CLASSIFICATION.items()
            if classification == DEFINITIVE and reason in reason_set
        ),
        None,
    )


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
    precomputed_reasons = record.get("eligibility_exclusion_reasons", [])
    if not isinstance(precomputed_reasons, list):
        precomputed_reasons = []
    definitive_overview_miss = (
        "OVERVIEW_UNAVAILABLE_AT_LAGGED_DATE" in precomputed_reasons
        or (
            isinstance(record.get("overview_failure"), dict)
            and record["overview_failure"].get("classification") == DEFINITIVE
        )
    )
    overview_no_share_count = (
        "OVERVIEW_NO_SHARE_COUNT" in precomputed_reasons
    )
    share_count = record.get("shares_outstanding")
    positive_share_count = (
        isinstance(share_count, (int, float))
        and not isinstance(share_count, bool)
        and float(share_count) > 0
    )
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
    if (
        listing is None
        and not definitive_overview_miss
        and not overview_no_share_count
        and not positive_share_count
    ):
        reasons.append("LISTING_DATE_MISSING")
    elif listing is not None and listing > day:
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
    if record.get("is_shell") is True:
        reasons.append("SHELL_COMPANY")
    if record.get("is_spac") is True:
        reasons.append("SPAC_SECURITY")
    if record.get("is_spac_suffix") is True:
        reasons.append("SPAC_SUFFIX_SECURITY")
    reasons.extend(
        str(reason) for reason in precomputed_reasons if str(reason)
    )

    market_cap = record.get("market_cap_usd", record.get("market_cap"))
    if not reasons:
        if not _known_by_cutoff(record, "market_cap", cutoff):
            reasons.append("MARKET_CAP_POINT_IN_TIME_UNPROVEN")
            LOGGER.warning(
                "MARKET_CAP_POINT_IN_TIME_UNPROVEN ticker=%s",
                ticker or "UNKNOWN",
            )
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
        "reference_data_as_of_timestamp": record.get(
            "reference_data_as_of_timestamp"
        ),
        "market_cap_as_of_timestamp": record.get(
            "market_cap_available_at"
        ),
        "shares_outstanding": record.get("shares_outstanding"),
        "shares_outstanding_lag_days": record.get(
            "shares_outstanding_lag_days"
        ),
        "shares_outstanding_lagged_date": record.get(
            "shares_outstanding_lagged_date"
        ),
        "shares_outstanding_provider_query_date": record.get(
            "shares_outstanding_provider_query_date"
        ),
        "shares_outstanding_provider_period_date": record.get(
            "shares_outstanding_provider_period_date"
        ),
        "prior_close": record.get("prior_close"),
        "prior_close_as_of_timestamp": record.get(
            "prior_close_as_of_timestamp"
        ),
        "market_cap_usd": market_cap,
        "cik": record.get("cik"),
        "composite_figi": record.get("composite_figi"),
        "prior_session_dollar_volume": record.get(
            "prior_session_dollar_volume"
        ),
        "duplicate_share_class_kept_ticker": record.get(
            "duplicate_share_class_kept_ticker"
        ),
        "overview_failure": record.get("overview_failure"),
        "provider_identifier_gap": _stable_id(record) is None,
        "deciding_definitive_reason": _deciding_definitive_reason(reasons),
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
    capabilities: dict[str, Any],
    freeze_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": "daily_universe_manifest_v1.0",
        "replay_id": replay_id,
        "trading_date": trading_date,
        "information_cutoff": information_cutoff(trading_date, freeze_config),
        "timezone": "America/New_York",
        "universe_mode": universe_mode,
        "massive_plan": load_massive_plan(),
        "ruleset": {"name": RULESET_NAME, "version": RULESET_VERSION},
        "capabilities": capabilities,
        "source": {
            "provider": provider,
            "feed_version": feed_version,
            "query_parameters": query_parameters,
            "retrieved_at": retrieved_at,
        },
        "coverage_status": "incomplete",
        "coverage_reasons": [],
        "provider_limitation_metrics": {
            "provider_identifier_gap_count": 0,
        },
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
    freeze_config: dict[str, Any] | None = None,
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
        capabilities={
            **{name: False for name in REQUIRED_CAPABILITIES},
            "provider_response_complete": True,
        },
        freeze_config=freeze_config,
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
            "deciding_definitive_reason": None,
            "provider_identifier_gap": False,
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
    capabilities: dict[str, Any],
    one_class_per_issuer: bool | None = None,
    retrieved_at: str | None = None,
    freeze_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = _base_manifest(
        replay_id=replay_id,
        trading_date=trading_date,
        universe_mode=HISTORICAL_RESEARCH,
        provider=provider,
        feed_version=feed_version,
        query_parameters=query_parameters,
        retrieved_at=retrieved_at or datetime.now(timezone.utc).isoformat(),
        capabilities=capabilities,
        freeze_config=freeze_config,
    )
    missing_capabilities = sorted(
        capability
        for capability in REQUIRED_CAPABILITIES
        if not _capability_available(
            capability, capabilities.get(capability)
        )
    )
    cutoff = _parse_timestamp(manifest["information_cutoff"])
    assert cutoff is not None
    evaluated = [_evaluate_record(record, trading_date, cutoff) for record in records]

    deduplicate_share_classes = (
        _UNIVERSE_CONFIG.get("one_class_per_issuer") is True
        if one_class_per_issuer is None
        else one_class_per_issuer
    )
    if deduplicate_share_classes:
        parent = list(range(len(evaluated)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        issuer_owner: dict[tuple[str, str], int] = {}
        for index, (item, _) in enumerate(evaluated):
            if not item["inclusion"]:
                continue
            identifiers = []
            if item.get("cik"):
                identifiers.append(("CIK", str(item["cik"]).strip().upper()))
            if item.get("composite_figi"):
                identifiers.append((
                    "COMPOSITE_FIGI",
                    str(item["composite_figi"]).strip().upper(),
                ))
            for identifier in identifiers:
                prior_index = issuer_owner.setdefault(identifier, index)
                union(index, prior_index)

        issuer_groups: dict[int, list[int]] = {}
        for index, (item, _) in enumerate(evaluated):
            if item["inclusion"] and (
                item.get("cik") or item.get("composite_figi")
            ):
                issuer_groups.setdefault(find(index), []).append(index)
        for indices in issuer_groups.values():
            if len(indices) < 2:
                continue
            kept_index = min(
                indices,
                key=lambda index: (
                    -float(
                        evaluated[index][0].get(
                            "prior_session_dollar_volume"
                        )
                        or 0
                    ),
                    evaluated[index][0]["ticker"],
                ),
            )
            kept_ticker = evaluated[kept_index][0]["ticker"]
            for index in indices:
                if index == kept_index:
                    continue
                item, reasons = evaluated[index]
                item["inclusion"] = False
                item["reason_codes"] = ["DUPLICATE_SHARE_CLASS"]
                item["duplicate_share_class_kept_ticker"] = kept_ticker
                item["deciding_definitive_reason"] = "DUPLICATE_SHARE_CLASS"
                reasons[:] = ["DUPLICATE_SHARE_CLASS"]

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
                item["deciding_definitive_reason"] = (
                    "DUPLICATE_STABLE_SECURITY_ID_FOR_DATE"
                )
                reasons[:] = ["DUPLICATE_STABLE_SECURITY_ID_FOR_DATE"]
            else:
                included_by_id[item["stable_security_id"]] = item
        definitive_reasons, gap_reasons = _classify_reasons(reasons)
        item["deciding_definitive_reason"] = (
            _deciding_definitive_reason(definitive_reasons)
        )
        if gap_reasons and not definitive_reasons:
            coverage_reasons.append(
                "SECURITY_METADATA_INCOMPLETE:"
                f"{item['stable_security_id']}:{item['ticker']}"
            )
        securities.append(item)

    manifest["securities"] = sorted(
        securities, key=lambda item: (item["stable_security_id"], item["ticker"])
    )
    manifest["eligible_symbol_count"] = sum(item["inclusion"] for item in securities)
    manifest["excluded_symbol_count"] = len(securities) - manifest["eligible_symbol_count"]
    manifest["provider_limitation_metrics"] = {
        "provider_identifier_gap_count": sum(
            bool(item["provider_identifier_gap"])
            for item in securities
        ),
    }
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
    freeze_config: dict[str, Any] | None = None,
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
            freeze_config=freeze_config,
        )
    else:
        capabilities = provider.historical_universe_capabilities()
        missing = [
            name
            for name in REQUIRED_CAPABILITIES
            if not _capability_available(name, capabilities.get(name))
        ]
        if missing:
            payload = {
                "records": [],
                "query_parameters": {
                    "date": trading_date,
                    "cutoff": information_cutoff(trading_date, freeze_config),
                    "request_skipped": "provider_capability_gate",
                },
            }
        else:
            try:
                payload = provider.get_historical_universe(
                    trading_date, information_cutoff(trading_date, freeze_config)
                )
            except ProviderError as exc:
                payload = {
                    "records": [],
                    "query_parameters": {
                        "date": trading_date,
                        "cutoff": information_cutoff(trading_date, freeze_config),
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
            freeze_config=freeze_config,
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
