from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trainer.providers.base import ProviderError
from trainer.providers.massive_flatfiles import (
    DAY_AGGS_DATASET,
    MassiveFlatFileStore,
)
from trainer.rate_control import load_massive_plan
from trainer.trading_calendar import generate_trading_dates
from trainer.universe_manifest import (
    build_research_manifest,
    calculate_manifest_hash,
    information_cutoff,
    persist_manifest,
    verify_manifest,
)


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "universe.json"
ET = ZoneInfo("America/New_York")
FEED_VERSION = "massive_reference_v3+massive_sip_flatfiles_v1"


class UniverseBuilderError(Exception):
    """Raised when a point-in-time universe cannot be constructed."""


def load_universe_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UniverseBuilderError(
            f"Unable to load universe config {path}: {exc}"
        ) from exc
    required = {
        "version",
        "allowed_exchanges",
        "included_security_types",
        "market_cap_usd",
        "excluded_symbol_suffixes",
        "spac_name_terms",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise UniverseBuilderError(
            f"Universe config must contain exactly {sorted(required)}."
        )
    return payload


def previous_trading_sessions(trading_date: str, count: int) -> list[str]:
    if count < 1:
        raise ValueError("count must be at least one.")
    target = date.fromisoformat(trading_date)
    span_days = max(45, count * 3)
    while True:
        start = target - timedelta(days=span_days)
        sessions = generate_trading_dates(
            start.isoformat(),
            (target - timedelta(days=1)).isoformat(),
        )
        if len(sessions) >= count:
            return sessions[-count:]
        span_days *= 2


def _positive_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def _stable_id(record: dict[str, Any]) -> str | None:
    for field in ("share_class_figi", "composite_figi", "cik"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _reference_as_of(trading_date: str) -> str:
    day = date.fromisoformat(trading_date)
    return datetime.combine(day, time(0), tzinfo=ET).isoformat()


def _timestamp_at_or_before(value: Any, cutoff: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        boundary = datetime.fromisoformat(cutoff)
    except ValueError:
        return None
    if observed.tzinfo is None or boundary.tzinfo is None or observed > boundary:
        return None
    return observed.isoformat()


def _latest_timestamp(*values: str) -> str:
    return max(
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in values
    ).isoformat()


def _exclusion_reasons(
    raw: dict[str, Any],
    overview: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[str], bool, bool, bool]:
    ticker = str(raw.get("ticker") or overview.get("ticker") or "").upper()
    name = str(overview.get("name") or raw.get("name") or "").upper()
    venue = str(
        overview.get("primary_exchange") or raw.get("primary_exchange") or ""
    )
    security_type = str(overview.get("type") or raw.get("type") or "UNKNOWN").upper()
    reasons: list[str] = []
    if venue not in set(config["allowed_exchanges"]):
        reasons.append("INELIGIBLE_LISTING_VENUE")
    if security_type not in set(config["included_security_types"]):
        reasons.append("SECURITY_TYPE_NOT_COMMON_EQUITY")
    is_shell = bool(overview.get("is_shell") or raw.get("is_shell"))
    is_spac_suffix = any(
        ticker.endswith(str(suffix).upper())
        for suffix in config["excluded_symbol_suffixes"]
    )
    is_spac = any(
        str(term).upper() in name for term in config["spac_name_terms"]
    )
    return sorted(set(reasons)), is_shell, is_spac, is_spac_suffix


def build_point_in_time_universe(
    reference_client: Any,
    flatfiles: MassiveFlatFileStore,
    trading_date: str,
    manifest_path: Path,
    *,
    replay_id: str | None = None,
    config_path: Path = CONFIG_PATH,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Build and persist the eligible universe using only date-D inputs."""
    date.fromisoformat(trading_date)
    config = load_universe_config(config_path)
    previous_session = previous_trading_sessions(trading_date, 1)[0]
    expected_replay_id = replay_id or f"{trading_date}-0700-flatfile-replay"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UniverseBuilderError(
                f"Unable to resume universe manifest {manifest_path}: {exc}"
            ) from exc
        verify_manifest(existing)
        query = existing["source"]["query_parameters"]
        if (
            existing["replay_id"] != expected_replay_id
            or existing["trading_date"] != trading_date
            or existing["source"]["provider"] != reference_client.provider_name
            or existing["source"]["feed_version"] != FEED_VERSION
            or existing["massive_plan"] != load_massive_plan()
            or query.get("reference_date") != trading_date
            or query.get("prior_close_trading_date") != previous_session
            or query.get("universe_config_version") != config["version"]
        ):
            raise UniverseBuilderError(
                "Existing universe manifest does not match this flat-file replay."
            )
        return existing
    prior_close_bars = {
        bar["ticker"]: bar
        for bar in flatfiles.iter_bars(DAY_AGGS_DATASET, previous_session)
    }
    reference_as_of = _reference_as_of(trading_date)
    try:
        ticker_rows = reference_client.get_tickers(
            trading_date,
            active=True,
            security_type=None,
        )
    except ProviderError as exc:
        raise UniverseBuilderError(
            f"Massive point-in-time ticker query failed for {trading_date}: {exc}"
        ) from exc

    records: list[dict[str, Any]] = []
    provider_complete = True
    point_in_time_market_cap = True
    cutoff = information_cutoff(trading_date)
    prior_close_as_of = datetime.combine(
        date.fromisoformat(previous_session), time(16), tzinfo=ET
    ).isoformat()
    for raw in ticker_rows:
        ticker = str(raw.get("ticker") or "").upper()
        try:
            overview = reference_client.get_ticker_overview(ticker, trading_date)
        except ProviderError:
            overview = {}
            provider_complete = False
        combined = {**raw, **overview}
        prior = prior_close_bars.get(ticker)
        prior_close = float(prior["close"]) if prior is not None else None
        shares = _positive_number(
            combined.get("weighted_shares_outstanding"),
            combined.get("share_class_shares_outstanding"),
            combined.get("shares_outstanding"),
        )
        # Massive's Ticker Overview `date` parameter selects SEC-derived data
        # by period-of-report date, which can predate the filing submission.
        # The raw endpoint therefore cannot, by itself, prove that a historical
        # share count was knowable at the replay cutoff. A provider adapter may
        # supply this normalized availability field when that proof exists.
        shares_available_at = _timestamp_at_or_before(
            combined.get("shares_outstanding_available_at"), cutoff
        )
        market_cap = (
            shares * prior_close
            if shares is not None and prior_close is not None
            else None
        )
        reasons, is_shell, is_spac, is_spac_suffix = _exclusion_reasons(
            raw, overview, config
        )
        requires_fundamental_proof = not reasons
        shell_status_known = (
            "is_shell" in raw or "is_shell" in overview
        )
        if requires_fundamental_proof and not shell_status_known:
            reasons.append("SHELL_STATUS_UNPROVEN")
        if requires_fundamental_proof and shares_available_at is None:
            reasons.append("SHARES_OUTSTANDING_AVAILABILITY_UNPROVEN")
            point_in_time_market_cap = False
        if market_cap is not None:
            limits = config["market_cap_usd"]
            if not float(limits["minimum"]) <= market_cap <= float(limits["maximum"]):
                reasons.append("MARKET_CAP_OUT_OF_RANGE")

        # `active=true&date=D` is the historical trading-status assertion.
        # A present-day active flag is deliberately not consulted.
        records.append({
            "ticker": ticker,
            "stable_security_id": _stable_id(combined),
            "share_class_figi": combined.get("share_class_figi"),
            "composite_figi": combined.get("composite_figi"),
            "primary_exchange": combined.get("primary_exchange"),
            "type": combined.get("type"),
            "locale": combined.get("locale", "us"),
            "market": combined.get("market", "stocks"),
            "list_date": combined.get("list_date"),
            "ticker_valid_from": combined.get("list_date"),
            "historical_active": True,
            "is_test_issue": bool(combined.get("is_test_issue", False)),
            "is_otc": bool(combined.get("is_otc", False)),
            "is_shell": is_shell,
            "is_spac": is_spac,
            "is_spac_suffix": is_spac_suffix,
            "eligibility_exclusion_reasons": sorted(set(reasons)),
            "metadata_available_at": reference_as_of,
            "reference_data_as_of_timestamp": reference_as_of,
            "shares_outstanding": shares,
            "shares_outstanding_available_at": shares_available_at,
            "prior_close": prior_close,
            "prior_close_as_of_timestamp": prior_close_as_of,
            "market_cap_usd": market_cap,
            "market_cap_available_at": (
                _latest_timestamp(shares_available_at, prior_close_as_of)
                if shares_available_at is not None and prior_close is not None
                else None
            ),
        })

    capabilities = {
        "point_in_time_listings": True,
        "delisted_securities": True,
        "stable_security_ids": True,
        "ticker_history": True,
        "point_in_time_exchange": True,
        "point_in_time_security_type": True,
        "historical_trading_status": True,
        "point_in_time_market_cap": point_in_time_market_cap,
        "provider_response_complete": provider_complete,
    }
    manifest = build_research_manifest(
        records,
        trading_date,
        replay_id=expected_replay_id,
        provider=reference_client.provider_name,
        feed_version=FEED_VERSION,
        query_parameters={
            "reference_endpoint": "/v3/reference/tickers",
            "reference_date": trading_date,
            "reference_active": True,
            "reference_type": None,
            "reference_data_as_of_timestamp": reference_as_of,
            "information_cutoff": information_cutoff(trading_date),
            "prior_close_dataset": DAY_AGGS_DATASET,
            "prior_close_trading_date": previous_session,
            "universe_config_version": config["version"],
        },
        capabilities=capabilities,
        retrieved_at=retrieved_at or datetime.now(timezone.utc).isoformat(),
    )
    if not provider_complete:
        manifest["coverage_reasons"] = sorted(
            set(manifest["coverage_reasons"] + ["PROVIDER_RESPONSE_INCOMPLETE"])
        )
        manifest["coverage_status"] = "incomplete"
        manifest["research_evidence"] = False
        manifest["promotion_eligible"] = False
        manifest["manifest_hash"] = calculate_manifest_hash(manifest)
    persist_manifest(manifest_path, manifest)
    return manifest
