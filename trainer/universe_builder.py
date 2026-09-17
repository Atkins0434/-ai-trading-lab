from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import sys
import time as time_module
from typing import Any
from zoneinfo import ZoneInfo

from trainer.providers.base import ProviderError
from trainer.providers.massive_flatfiles import (
    DAY_AGGS_DATASET,
    MassiveFlatFileStore,
)
from trainer.rate_control import (
    load_massive_plan,
    load_reference_fetch_workers,
)
from trainer.reference_cache import (
    OverviewCacheBatchItem,
    TickerOverviewCache,
)
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
        "shares_outstanding_lag_days",
        "max_share_price_usd",
        "one_class_per_issuer",
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
    if (
        not isinstance(payload["shares_outstanding_lag_days"], int)
        or payload["shares_outstanding_lag_days"] < 1
    ):
        raise UniverseBuilderError(
            "shares_outstanding_lag_days must be a positive integer."
        )
    if (
        not isinstance(payload["max_share_price_usd"], (int, float))
        or isinstance(payload["max_share_price_usd"], bool)
        or payload["max_share_price_usd"] <= 0
    ):
        raise UniverseBuilderError(
            "max_share_price_usd must be a positive number."
        )
    if not isinstance(payload["one_class_per_issuer"], bool):
        raise UniverseBuilderError(
            "one_class_per_issuer must be a boolean."
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


def _provider_period_date(overview: dict[str, Any]) -> str | None:
    for field in (
        "period_of_report_date",
        "period_end",
        "period_date",
    ):
        value = overview.get(field)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()[:10]
            try:
                return date.fromisoformat(candidate).isoformat()
            except ValueError:
                continue
    return None


def _exclusion_reasons(
    raw: dict[str, Any],
    overview: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[str], bool, bool, bool]:
    ticker = str(raw.get("ticker") or overview.get("ticker") or "").upper()
    name = str(raw.get("name") or overview.get("name") or "").upper()
    venue = str(
        raw.get("primary_exchange") or overview.get("primary_exchange") or ""
    )
    security_type = str(raw.get("type") or overview.get("type") or "UNKNOWN").upper()
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
    reference_cache_root: Path = ROOT / "data" / "reference_cache",
    reference_fetch_workers: int | None = None,
    max_tickers: int | None = None,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Build and persist the eligible universe using only date-D inputs."""
    date.fromisoformat(trading_date)
    if (
        max_tickers is not None
        and (
            not isinstance(max_tickers, int)
            or isinstance(max_tickers, bool)
            or max_tickers < 1
        )
    ):
        raise UniverseBuilderError("max_tickers must be a positive integer.")
    smoke_mode = max_tickers is not None
    config = load_universe_config(config_path)
    active_plan = load_massive_plan()
    configured_workers = (
        load_reference_fetch_workers()
        if reference_fetch_workers is None
        else reference_fetch_workers
    )
    if (
        not isinstance(configured_workers, int)
        or isinstance(configured_workers, bool)
        or configured_workers < 1
    ):
        raise UniverseBuilderError(
            "reference_fetch_workers must be a positive integer."
        )
    effective_workers = (
        1
        if active_plan["rest_calls_per_minute"] is not None
        else configured_workers
    )
    lag_days = int(config["shares_outstanding_lag_days"])
    lagged_date = (
        date.fromisoformat(trading_date) - timedelta(days=lag_days)
    ).isoformat()
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
            or existing["massive_plan"] != active_plan
            or query.get("reference_date") != trading_date
            or query.get("prior_close_trading_date") != previous_session
            or query.get("universe_config_version") != config["version"]
            or query.get("shares_outstanding_lag_days") != lag_days
            or query.get("shares_outstanding_lagged_date") != lagged_date
            or query.get("smoke_mode", False) is not smoke_mode
            or query.get("max_tickers") != max_tickers
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

    # Smoke runs deliberately choose the first N symbols in a stable ordering
    # before any overview request is made. The cap makes the universe
    # incomplete by construction, so its artifacts are never research evidence.
    if max_tickers is not None:
        ticker_rows = sorted(
            ticker_rows,
            key=lambda item: (
                str(item.get("ticker") or "").upper(),
                str(_stable_id(item) or ""),
            ),
        )[:max_tickers]

    records: list[dict[str, Any]] = []
    provider_complete = True
    overview_cache = TickerOverviewCache(reference_cache_root)
    cache_hits = 0
    cache_fetches = 0
    quarter_reuse_hits = 0
    cache_errors = 0
    definitive_misses = 0
    transient_retries = 0
    unresolved_failures = 0
    listed_after_lagged_date = 0
    tickers_processed = 0
    progress_started = time_module.perf_counter()

    def record_progress(item: OverviewCacheBatchItem) -> None:
        nonlocal cache_hits
        nonlocal cache_fetches
        nonlocal quarter_reuse_hits
        nonlocal cache_errors
        nonlocal definitive_misses
        nonlocal transient_retries
        nonlocal unresolved_failures
        nonlocal tickers_processed
        tickers_processed += 1
        if item.error is not None:
            cache_errors += 1
            unresolved_failures += 1
            transient_retries += int(
                getattr(item.error, "retry_count", 0)
            )
        else:
            assert item.result is not None
            transient_retries += item.result.transient_retries
            if item.result.definitive_miss:
                definitive_misses += 1
            if item.result.cache_hit:
                cache_hits += 1
            else:
                cache_fetches += 1
            if item.result.quarter_reuse:
                quarter_reuse_hits += 1
        if (
            tickers_processed % 250 == 0
            or tickers_processed == len(ticker_rows)
        ):
            elapsed = time_module.perf_counter() - progress_started
            print(
                "[universe] "
                f"tickers_processed={tickers_processed}/{len(ticker_rows)} "
                f"cache_hits={cache_hits} fetches={cache_fetches} "
                f"definitive_misses={definitive_misses} "
                f"transient_retries={transient_retries} "
                f"unresolved_failures={unresolved_failures} "
                f"errors={cache_errors} elapsed_seconds={elapsed:.2f}",
                file=sys.stderr,
                flush=True,
            )

    overview_rows: list[dict[str, Any]] = []
    skipped_overview_tickers: set[str] = set()
    lagged_day = date.fromisoformat(lagged_date)
    for raw in ticker_rows:
        ticker = str(raw.get("ticker") or "").upper()
        raw_list_date = raw.get("list_date")
        try:
            listed_day = (
                date.fromisoformat(str(raw_list_date)[:10])
                if raw_list_date
                else None
            )
        except ValueError:
            listed_day = None
        if listed_day is not None and listed_day > lagged_day:
            skipped_overview_tickers.add(ticker)
            listed_after_lagged_date += 1
        else:
            overview_rows.append(raw)

    tickers_processed = listed_after_lagged_date

    fetched_overviews = overview_cache.get_many(
        reference_client,
        [str(raw.get("ticker") or "").upper() for raw in overview_rows],
        lagged_date,
        max_workers=effective_workers,
        on_progress=record_progress,
    )
    overview_results = {
        item.ticker: item for item in fetched_overviews
    }
    prior_close_as_of = datetime.combine(
        date.fromisoformat(previous_session), time(16), tzinfo=ET
    ).isoformat()
    for raw in ticker_rows:
        ticker = str(raw.get("ticker") or "").upper()
        overview_failure = None
        listed_after_lag = ticker in skipped_overview_tickers
        overview_item = overview_results.get(ticker)
        if listed_after_lag:
            overview = {}
            provider_query_date = lagged_date
        elif overview_item is not None and overview_item.error is not None:
            overview = {}
            provider_complete = False
            provider_query_date = lagged_date
            overview_failure = {
                "classification": "TRANSIENT",
                "http_status": getattr(
                    overview_item.error, "http_status", None
                ),
                "exception_type": getattr(
                    overview_item.error,
                    "exception_type",
                    type(overview_item.error).__name__,
                ),
            }
        else:
            assert overview_item is not None
            assert overview_item.result is not None
            overview = overview_item.result.overview or {}
            provider_query_date = overview_item.result.provider_query_date
            if overview_item.result.definitive_miss:
                overview_failure = {
                    "classification": "DEFINITIVE",
                    "http_status": overview_item.result.http_status,
                    "exception_type": overview_item.result.exception_type
                    or "UNKNOWN",
                }
        combined = {**overview, **raw}
        prior = prior_close_bars.get(ticker)
        prior_close = float(prior["close"]) if prior is not None else None
        shares = _positive_number(
            overview.get("weighted_shares_outstanding"),
            overview.get("share_class_shares_outstanding"),
            overview.get("shares_outstanding"),
        )
        market_cap = (
            shares * prior_close
            if shares is not None and prior_close is not None
            else None
        )
        reasons, is_shell, is_spac, is_spac_suffix = _exclusion_reasons(
            raw, overview, config
        )
        if overview_failure is not None:
            reasons.append(
                "OVERVIEW_UNAVAILABLE_AT_LAGGED_DATE"
                if overview_failure["classification"] == "DEFINITIVE"
                else "OVERVIEW_FETCH_FAILED"
            )
        if listed_after_lag:
            reasons.append("LISTED_AFTER_LAGGED_DATE")
        provider_period_date = _provider_period_date(overview)
        if (
            provider_period_date is not None
            and date.fromisoformat(provider_period_date)
            > date.fromisoformat(trading_date)
        ):
            reasons.append("SHARES_PERIOD_AFTER_REPLAY_DATE")
        if market_cap is not None:
            limits = config["market_cap_usd"]
            if not float(limits["minimum"]) <= market_cap <= float(limits["maximum"]):
                reasons.append("MARKET_CAP_OUT_OF_RANGE")
        if (
            prior_close is not None
            and prior_close > float(config["max_share_price_usd"])
        ):
            reasons.append("SHARE_PRICE_ABOVE_CAP")

        prior_session_dollar_volume = (
            prior_close * float(prior.get("volume", 0))
            if prior is not None and prior_close is not None
            else None
        )

        # `active=true&date=D` is the historical trading-status assertion.
        # A present-day active flag is deliberately not consulted.
        records.append({
            "ticker": ticker,
            "stable_security_id": _stable_id(combined),
            "share_class_figi": combined.get("share_class_figi"),
            "composite_figi": combined.get("composite_figi"),
            "cik": combined.get("cik"),
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
            "shares_outstanding_lag_days": lag_days,
            "shares_outstanding_lagged_date": lagged_date,
            "shares_outstanding_provider_query_date": provider_query_date,
            "shares_outstanding_provider_period_date": provider_period_date,
            "prior_close": prior_close,
            "prior_close_as_of_timestamp": prior_close_as_of,
            "prior_session_dollar_volume": prior_session_dollar_volume,
            "market_cap_usd": market_cap,
            "overview_failure": overview_failure,
            "market_cap_available_at": (
                reference_as_of
                if shares is not None and prior_close is not None
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
        "point_in_time_market_cap": "LAGGED_PROXY",
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
            "shares_outstanding_lag_days": lag_days,
            "shares_outstanding_lagged_date": lagged_date,
            "smoke_mode": smoke_mode,
            "max_tickers": max_tickers,
            "ticker_overview_cache": {
                "hits": cache_hits,
                "fetches": cache_fetches,
                "quarter_reuse_hits": quarter_reuse_hits,
                "errors": cache_errors,
                "definitive_misses": definitive_misses,
                "transient_retries": transient_retries,
                "unresolved_failures": unresolved_failures,
                "listed_after_lagged_date": listed_after_lagged_date,
            },
            "prior_close_dataset": DAY_AGGS_DATASET,
            "prior_close_trading_date": previous_session,
            "universe_config_version": config["version"],
            "max_share_price_usd": config["max_share_price_usd"],
            "one_class_per_issuer": config["one_class_per_issuer"],
        },
        capabilities=capabilities,
        one_class_per_issuer=config["one_class_per_issuer"],
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
    if smoke_mode:
        manifest["research_evidence"] = False
        manifest["promotion_eligible"] = False
        manifest["manifest_hash"] = calculate_manifest_hash(manifest)
    persist_manifest(manifest_path, manifest)
    return manifest
