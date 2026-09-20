from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from statistics import mean
import sys
import time as wall_time
from typing import Any
from zoneinfo import ZoneInfo

from trainer.providers.massive_flatfiles import (
    DAY_AGGS_DATASET,
    MINUTE_AGGS_DATASET,
    SOURCE,
    MassiveFlatFileStore,
)
from trainer.sector_metrics import BENCHMARK_SYMBOLS
from trainer.rate_control import load_massive_plan
from trainer.replay_engine import configured_freeze_datetime
from trainer.universe_builder import previous_trading_sessions
from trainer.universe_manifest import evidence_metadata
from trainer.validate_contracts import validate_contract


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "flatfile_replay.json"
ET = ZoneInfo("America/New_York")


class FlatFileSnapshotError(Exception):
    """Raised when cached flat files cannot produce a replay snapshot."""


@dataclass(frozen=True)
class FlatFileSnapshotResult:
    snapshot: dict[str, Any]
    outcome_bars: dict[str, list[dict[str, Any]]]
    bar_statistics: dict[str, Any]
    lookback_dates: list[str]
    excluded_tickers: list[dict[str, Any]]


def load_flatfile_replay_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FlatFileSnapshotError(
            f"Unable to load flat-file replay config {path}: {exc}"
        ) from exc
    required = {
        "version",
        "cache_version",
        "baseline_lookback_sessions",
        "morning_freeze_time",
        "strategy_capital_usd",
        "comparison_policy_paths",
        "cache_root",
        "reference_cache_root",
        "output_root",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise FlatFileSnapshotError(
            f"Flat-file replay config must contain exactly {sorted(required)}."
        )
    if int(payload["baseline_lookback_sessions"]) < 1:
        raise FlatFileSnapshotError(
            "baseline_lookback_sessions must be at least one."
        )
    configured_freeze_datetime("2000-01-03", payload)
    if (
        not isinstance(payload["cache_version"], str)
        or not payload["cache_version"].strip()
    ):
        raise FlatFileSnapshotError(
            "cache_version must be a non-empty string."
        )
    comparison_paths = payload["comparison_policy_paths"]
    if (
        not isinstance(comparison_paths, list)
        or any(
            not isinstance(value, str) or not value.strip()
            for value in comparison_paths
        )
        or len(set(comparison_paths)) != len(comparison_paths)
    ):
        raise FlatFileSnapshotError(
            "comparison_policy_paths must contain unique non-empty strings."
        )
    return payload


def _observation(
    value: float | int | None,
    as_of: str,
    provider_field: str,
    *,
    missing_reason: str | None = None,
) -> dict[str, Any]:
    result = {
        "value": value,
        "as_of_timestamp": as_of,
        "source": SOURCE,
        "provider_field": provider_field,
    }
    if missing_reason is not None:
        result["missing_reason"] = missing_reason
    return result


def _average_true_range(
    prior_rows: list[dict[str, Any]],
    sessions: int = 14,
) -> tuple[float | None, int]:
    """Return simple ATR from completed sessions and the usable TR count."""
    ordered = sorted(prior_rows, key=lambda item: item["trading_date"])
    true_ranges: list[float] = []
    start = max(1, len(ordered) - sessions)
    for index in range(start, len(ordered)):
        bar = ordered[index]
        prior_close = float(ordered[index - 1]["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        true_ranges.append(
            max(
                high - low,
                abs(high - prior_close),
                abs(low - prior_close),
            )
        )
    sessions_used = len(true_ranges)
    return (
        mean(true_ranges) if sessions_used == sessions else None,
        sessions_used,
    )


def _benchmark_prior_close(rows: list[dict[str, Any]], prior_date: str) -> float | None:
    matches = [bar for bar in rows if bar["trading_date"] == prior_date]
    return float(matches[0]["close"]) if len(matches) == 1 else None


def _snapshot_bar(bar: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": bar["timestamp"],
        "open": bar["open"],
        "high": bar["high"],
        "low": bar["low"],
        "close": bar["close"],
        "volume": bar["volume"],
        "source": bar["source"],
        "session": bar["session"],
    }


class _TickerProgress:
    """Low-overhead progress reporting suitable for CI and interactive runs."""

    def __init__(self, phase: str, total: int, *, log_every: int = 250) -> None:
        self.phase = phase
        self.total = total
        self.log_every = log_every
        self.started = wall_time.perf_counter()

    def update(self, completed: int) -> None:
        elapsed = wall_time.perf_counter() - self.started
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = (self.total - completed) / rate if rate > 0 else 0.0
        message = (
            f"phase={self.phase} tickers={completed}/{self.total} "
            f"elapsed_seconds={elapsed:.1f} eta_seconds={eta:.1f}"
        )
        if sys.stderr.isatty():
            print(f"\r[{self.phase}] {message}", end="", file=sys.stderr, flush=True)
            if completed == self.total:
                print(file=sys.stderr, flush=True)
        if completed == self.total or completed % self.log_every == 0:
            print(f"[flatfile_snapshot] {message}", file=sys.stderr, flush=True)


def _log_load_progress(completed: int, total: int, started: float) -> None:
    elapsed = wall_time.perf_counter() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else 0.0
    print(
        "[flatfile_snapshot] phase=load_flatfiles "
        f"files={completed}/{total} elapsed_seconds={elapsed:.1f} "
        f"eta_seconds={eta:.1f}",
        file=sys.stderr,
        flush=True,
    )


def _identity_maps(
    universe_manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return eligible identities and unambiguous point-in-time ticker aliases."""
    eligible = {
        item.get("stable_security_id", f"TICKER:{item['ticker']}"): item
        for item in universe_manifest["securities"]
        if item["inclusion"]
    }
    ids_by_ticker: dict[str, set[str]] = defaultdict(set)
    for stable_id, item in eligible.items():
        aliases = {str(item["ticker"]).upper()}
        history = item.get("ticker_history") or []
        for value in history:
            if isinstance(value, str):
                aliases.add(value.upper())
            elif isinstance(value, dict) and value.get("ticker"):
                aliases.add(str(value["ticker"]).upper())
        for alias in aliases:
            ids_by_ticker[alias].add(stable_id)
    unique_ticker_ids = {
        ticker: next(iter(stable_ids))
        for ticker, stable_ids in ids_by_ticker.items()
        if len(stable_ids) == 1
    }
    return eligible, unique_ticker_ids


def _resolve_bar_identity(
    bar: dict[str, Any],
    eligible_by_id: dict[str, dict[str, Any]],
    unique_ticker_ids: dict[str, str],
) -> str | None:
    provider_id = bar.get("stable_security_id")
    if provider_id is not None:
        normalized = str(provider_id).strip()
        return normalized if normalized in eligible_by_id else None
    return unique_ticker_ids.get(str(bar["ticker"]).upper())


def _collision_row(bar: dict[str, Any]) -> dict[str, Any]:
    """Retain the provider values needed to audit a duplicate-minute choice."""
    return {
        "open": bar["open"],
        "high": bar["high"],
        "low": bar["low"],
        "close": bar["close"],
        "volume": bar["volume"],
        "transactions": bar["transactions"],
    }


def _deduplicate_regular_bars(
    ticker: str,
    bars: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sort regular bars and select one provider row per ticker-minute."""
    selected_by_minute: dict[datetime, dict[str, Any]] = {}
    collisions: list[dict[str, Any]] = []
    for bar in sorted(
        bars,
        key=lambda item: datetime.fromisoformat(item["timestamp"]),
    ):
        observed = datetime.fromisoformat(bar["timestamp"])
        minute = observed.replace(second=0, microsecond=0)
        existing = selected_by_minute.get(minute)
        if existing is None:
            selected_by_minute[minute] = bar
            continue

        existing_priority = (
            int(existing["transactions"]),
            float(existing["volume"]),
        )
        incoming_priority = (
            int(bar["transactions"]),
            float(bar["volume"]),
        )
        if incoming_priority > existing_priority:
            kept, discarded = bar, existing
            selected_by_minute[minute] = bar
        else:
            kept, discarded = existing, bar
        collisions.append(
            {
                "ticker": ticker,
                "timestamp": minute.isoformat(),
                "kept_row": _collision_row(kept),
                "discarded_row": _collision_row(discarded),
            }
        )

    selected = sorted(
        selected_by_minute.values(),
        key=lambda item: datetime.fromisoformat(item["timestamp"]),
    )
    return [_snapshot_bar(bar) for bar in selected], collisions


def _bar_discontinuity(
    ticker: str,
    bars: list[dict[str, Any]],
    collisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Identify an implausible adjacent-minute close discontinuity."""
    for collision in collisions or []:
        left = float(collision["kept_row"]["close"])
        right = float(collision["discarded_row"]["close"])
        smaller, larger = sorted((left, right))
        if larger / smaller - 1.0 > 0.50:
            return {
                "ticker": ticker,
                "reason": "BAR_DISCONTINUITY",
                "previous_timestamp": collision["timestamp"],
                "current_timestamp": collision["timestamp"],
                "previous_close": left,
                "current_close": right,
                "close_change_pct": (right / left - 1.0) * 100,
            }
    for previous, current in zip(bars, bars[1:]):
        previous_close = float(previous["close"])
        change = float(current["close"]) / previous_close - 1.0
        if abs(change) > 0.50:
            return {
                "ticker": ticker,
                "reason": "BAR_DISCONTINUITY",
                "previous_timestamp": previous["timestamp"],
                "current_timestamp": current["timestamp"],
                "previous_close": previous_close,
                "current_close": float(current["close"]),
                "close_change_pct": change * 100,
            }
    return None


def build_flatfile_snapshot(
    trading_date: str,
    universe_manifest: dict[str, Any],
    flatfiles: MassiveFlatFileStore,
    *,
    lookback_sessions: int = 20,
    config: dict[str, Any] | None = None,
) -> FlatFileSnapshotResult:
    """Build one frozen Research Alpha snapshot entirely from flat files."""
    target = date.fromisoformat(trading_date)
    active_config = config or load_flatfile_replay_config()
    freeze = configured_freeze_datetime(trading_date, active_config)
    if universe_manifest["trading_date"] != trading_date:
        raise FlatFileSnapshotError(
            "Universe manifest date does not match snapshot date."
        )
    if universe_manifest["coverage_status"] != "complete":
        raise FlatFileSnapshotError(
            "Incomplete universe manifests cannot produce research snapshots."
        )
    securities = [
        item for item in universe_manifest["securities"] if item["inclusion"]
    ]
    eligible_by_id, unique_ticker_ids = _identity_maps(universe_manifest)
    tickers = {item["ticker"] for item in securities}
    lookback_dates = previous_trading_sessions(trading_date, lookback_sessions)

    benchmark_daily = defaultdict(list)
    benchmark_premarket = defaultdict(list)
    daily_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    premarket_history: dict[str, dict[str, float]] = {
        stable_id: {value: 0.0 for value in lookback_dates}
        for stable_id in eligible_by_id
    }
    target_premarket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_regular: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved_bar_rows: dict[str, int] = defaultdict(int)

    load_started = wall_time.perf_counter()
    total_load_files = len(lookback_dates) * 2 + 1
    loaded_files = 0
    for session_date in lookback_dates:
        for bar in flatfiles.iter_bars(
            DAY_AGGS_DATASET, session_date, tickers=tickers | set(BENCHMARK_SYMBOLS)
        ):
            if bar["ticker"] in BENCHMARK_SYMBOLS:
                benchmark_daily[bar["ticker"]].append(bar)
                continue
            stable_id = _resolve_bar_identity(
                bar, eligible_by_id, unique_ticker_ids
            )
            if stable_id is None:
                unresolved_bar_rows[bar["ticker"]] += 1
                continue
            daily_history[stable_id].append(bar)
        loaded_files += 1
        _log_load_progress(loaded_files, total_load_files, load_started)
        for bar in flatfiles.iter_bars(
            MINUTE_AGGS_DATASET, session_date, tickers=tickers
        ):
            stable_id = _resolve_bar_identity(
                bar, eligible_by_id, unique_ticker_ids
            )
            if stable_id is None:
                unresolved_bar_rows[bar["ticker"]] += 1
                continue
            observed = datetime.fromisoformat(bar["timestamp"]).astimezone(ET)
            session_freeze = configured_freeze_datetime(
                session_date, active_config
            )
            if (
                observed.date() == date.fromisoformat(session_date)
                and observed < session_freeze
                and observed.time() >= time(4)
            ):
                premarket_history[stable_id][session_date] += float(
                    bar["volume"]
                )
        loaded_files += 1
        _log_load_progress(loaded_files, total_load_files, load_started)

    for bar in flatfiles.iter_bars(
        MINUTE_AGGS_DATASET, trading_date, tickers=tickers | set(BENCHMARK_SYMBOLS)
    ):
        if bar["ticker"] in BENCHMARK_SYMBOLS:
            observed = datetime.fromisoformat(bar["timestamp"]).astimezone(ET)
            if observed.date() == target and time(4) <= observed.time() and observed < freeze:
                benchmark_premarket[bar["ticker"]].append(_snapshot_bar(bar))
            continue
        stable_id = _resolve_bar_identity(
            bar, eligible_by_id, unique_ticker_ids
        )
        if stable_id is None:
            unresolved_bar_rows[bar["ticker"]] += 1
            continue
        observed = datetime.fromisoformat(bar["timestamp"]).astimezone(ET)
        if observed.date() != target:
            continue
        if time(4) <= observed.time() and observed < freeze:
            target_premarket[stable_id].append(bar)
        elif time(9, 30) <= observed.time() < time(16):
            target_regular[stable_id].append(bar)
    loaded_files += 1
    _log_load_progress(loaded_files, total_load_files, load_started)

    metadata = evidence_metadata(universe_manifest)
    snapshot = {
        "replay_id": universe_manifest["replay_id"],
        "trading_date": trading_date,
        "freeze_timestamp": freeze.isoformat(),
        "timezone": "America/New_York",
        "universe_version": universe_manifest["ruleset"]["version"],
        "scout_version": "research_scout_alpha_v1.3",
        "execution_policy_version": "execution_disabled",
        "feature_registry_version": "feature_registry_alpha_v1.0",
        "data_source": {
            "provider": flatfiles.provider_name,
            "feed_version": flatfiles.feed_version,
        },
        "massive_plan": load_massive_plan(),
        **metadata,
        "market_benchmarks": {symbol: {
            "premarket_bars": sorted(benchmark_premarket[symbol], key=lambda b: b["timestamp"]),
            "market_data": {"previous_close": _observation(
                _benchmark_prior_close(benchmark_daily[symbol], lookback_dates[-1]),
                datetime.combine(date.fromisoformat(lookback_dates[-1]), time(16), tzinfo=ET).isoformat(),
                "day_aggs_v1.close",
            )},
        } for symbol in BENCHMARK_SYMBOLS},
        "securities": [],
    }
    outcome_bars: dict[str, list[dict[str, Any]]] = {}
    per_ticker_counts: dict[str, dict[str, int]] = {}
    per_security_counts: dict[str, dict[str, int]] = {}
    duplicate_minute_rows: list[dict[str, Any]] = []
    excluded_tickers: list[dict[str, Any]] = []

    ordered_securities = sorted(
        securities,
        key=lambda item: (
            item["ticker"],
            item.get("stable_security_id", f"TICKER:{item['ticker']}"),
        ),
    )
    ticker_identity_counts = Counter(
        item["ticker"] for item in ordered_securities
    )
    progress = _TickerProgress("snapshot", len(ordered_securities))
    for completed, manifest_security in enumerate(ordered_securities, start=1):
        ticker = manifest_security["ticker"]
        stable_id = manifest_security.get(
            "stable_security_id", f"TICKER:{ticker}"
        )
        prior_rows = sorted(
            daily_history.get(stable_id, []),
            key=lambda item: item["trading_date"],
        )
        previous_close = (
            float(manifest_security["prior_close"])
            if manifest_security.get("prior_close") is not None
            else float(prior_rows[-1]["close"])
            if prior_rows
            else None
        )
        if previous_close is None:
            raise FlatFileSnapshotError(
                f"No prior close is available for eligible ticker {ticker}."
            )
        current = sorted(
            (_snapshot_bar(bar) for bar in target_premarket.get(stable_id, [])),
            key=lambda item: item["timestamp"],
        )
        premarket_bars = current
        last_hour_start = freeze - timedelta(minutes=60)
        real_bar_count_60m = sum(
            datetime.fromisoformat(bar["timestamp"]).astimezone(ET)
            >= last_hour_start
            for bar in premarket_bars
        )
        premarket_volume = sum(float(bar["volume"]) for bar in current)
        premarket_dollar_volume = sum(
            ((float(bar["high"]) + float(bar["low"]) + float(bar["close"])) / 3)
            * float(bar["volume"])
            for bar in current
        )
        average_premarket_volume = mean(
            premarket_history[stable_id][value] for value in lookback_dates
        )
        relative_volume = (
            premarket_volume / average_premarket_volume
            if average_premarket_volume > 0
            else 0.0
        )
        average_daily_dollar_volume = (
            mean(
                float(bar["close"]) * float(bar["volume"])
                for bar in prior_rows
            )
            if prior_rows
            else None
        )
        average_daily_volume = (
            mean(float(bar["volume"]) for bar in prior_rows)
            if prior_rows
            else None
        )
        average_daily_range_pct = (
            mean(
                (float(bar["high"]) - float(bar["low"]))
                / float(bar["close"])
                * 100
                for bar in prior_rows
            )
            if prior_rows
            else None
        )
        atr_14_usd, atr_sessions_used = _average_true_range(prior_rows)
        window_as_of = freeze.isoformat()
        prior_as_of = datetime.combine(
            date.fromisoformat(lookback_dates[-1]), time(16), tzinfo=ET
        ).isoformat()
        last_trade_as_of = (
            premarket_bars[-1]["timestamp"] if premarket_bars else prior_as_of
        )
        market_data = {
            "last_price": _observation(
                (
                    float(premarket_bars[-1]["close"])
                    if premarket_bars
                    else previous_close
                ),
                last_trade_as_of,
                "minute_aggs_v1.close",
            ),
            "premarket_volume": _observation(
                premarket_volume,
                window_as_of,
                f"sum(volume),04:00-{freeze.time().isoformat()}ET",
            ),
            "premarket_dollar_volume": _observation(
                premarket_dollar_volume,
                window_as_of,
                f"sum(typical_price*volume),04:00-{freeze.time().isoformat()}ET",
            ),
            "average_daily_dollar_volume": _observation(
                average_daily_dollar_volume,
                prior_as_of,
                f"mean(close*volume),{lookback_sessions}_sessions",
            ),
            "average_daily_volume": _observation(
                average_daily_volume,
                prior_as_of,
                f"mean(volume),{lookback_sessions}_sessions",
            ),
            "relative_volume": _observation(
                relative_volume,
                window_as_of,
                f"premarket_volume/{lookback_sessions}_session_mean",
            ),
            "real_bar_count": _observation(
                len(premarket_bars),
                window_as_of,
                f"count(real_trade_minutes),04:00-{freeze.time().isoformat()}ET",
            ),
            "real_bar_count_60m": _observation(
                real_bar_count_60m,
                window_as_of,
                "count(real_trade_minutes),last_60m_before_freeze",
            ),
            "previous_close": _observation(
                previous_close, prior_as_of, "day_aggs_v1.close"
            ),
            "average_daily_range_pct": _observation(
                average_daily_range_pct,
                prior_as_of,
                f"mean((high-low)/close),{lookback_sessions}_sessions",
            ),
            "atr_14_usd": _observation(
                atr_14_usd,
                prior_as_of,
                "mean(true_range),14_sessions",
                missing_reason=(
                    None
                    if atr_14_usd is not None
                    else "INSUFFICIENT_ATR_HISTORY"
                ),
            ),
            "atr_lookback_sessions": _observation(
                14,
                prior_as_of,
                "configured_atr_lookback_sessions",
            ),
            "atr_sessions_used": _observation(
                atr_sessions_used,
                prior_as_of,
                "count(usable_true_range_sessions)",
            ),
        }
        security = {
            "ticker": ticker,
            "stable_security_id": stable_id,
            "sic_code": manifest_security.get("sic_code"),
            "sic_description": manifest_security.get("sic_description"),
            "exchange": manifest_security["listing_venue"],
            "eligible": True,
            "eligibility_as_of_timestamp": manifest_security[
                "reference_data_as_of_timestamp"
            ],
            "eligibility_reasons": [],
            "market_data": market_data,
            "premarket_bars": premarket_bars,
            "news": [],
            "filings": [],
            "analyst_actions": [],
            "social_observations": [],
        }
        if manifest_security.get("market_cap_usd") is not None:
            security["market_cap_usd"] = _observation(
                float(manifest_security["market_cap_usd"]),
                manifest_security["market_cap_as_of_timestamp"],
                "shares_outstanding*prior_session_close",
            )
        snapshot["securities"].append(security)

        regular, collisions = _deduplicate_regular_bars(
            ticker,
            target_regular.get(stable_id, []),
        )
        duplicate_minute_rows.extend(collisions)
        discontinuity = _bar_discontinuity(ticker, regular, collisions)
        if discontinuity is not None:
            excluded_tickers.append(discontinuity)
        outcome_key = (
            ticker if ticker_identity_counts[ticker] == 1 else stable_id
        )
        outcome_bars[outcome_key] = regular
        counts = {
            "real_premarket": len(premarket_bars),
            "real_premarket_60m": real_bar_count_60m,
            "real_regular": len(regular),
        }
        per_security_counts[stable_id] = counts
        if ticker_identity_counts[ticker] == 1:
            per_ticker_counts[ticker] = counts
        progress.update(completed)

    validate_contract("historical_snapshot", snapshot)
    bar_stats = {
        "real_premarket_bar_count": sum(
            item["real_premarket"] for item in per_ticker_counts.values()
        ),
        "real_premarket_bar_count_60m": sum(
            item["real_premarket_60m"] for item in per_ticker_counts.values()
        ),
        "real_regular_bar_count": sum(
            item["real_regular"] for item in per_ticker_counts.values()
        ),
        "duplicate_minute_rows": duplicate_minute_rows,
        "unresolved_bar_rows": dict(sorted(unresolved_bar_rows.items())),
        "unresolved_bar_row_count": sum(unresolved_bar_rows.values()),
        "by_ticker": per_ticker_counts,
        "by_stable_security_id": per_security_counts,
    }
    return FlatFileSnapshotResult(
        snapshot=snapshot,
        outcome_bars=outcome_bars,
        bar_statistics=bar_stats,
        lookback_dates=lookback_dates,
        excluded_tickers=excluded_tickers,
    )
