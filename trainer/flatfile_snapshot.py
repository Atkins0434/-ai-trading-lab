from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

from trainer.providers.massive_flatfiles import (
    DAY_AGGS_DATASET,
    MINUTE_AGGS_DATASET,
    SOURCE,
    MassiveFlatFileStore,
)
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
    return payload


def _observation(
    value: float | int | None,
    as_of: str,
    provider_field: str,
) -> dict[str, Any]:
    return {
        "value": value,
        "as_of_timestamp": as_of,
        "source": SOURCE,
        "provider_field": provider_field,
    }


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
    tickers = {item["ticker"] for item in securities}
    lookback_dates = previous_trading_sessions(trading_date, lookback_sessions)

    daily_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    premarket_history: dict[str, dict[str, float]] = {
        ticker: {value: 0.0 for value in lookback_dates} for ticker in tickers
    }
    target_premarket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_regular: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for session_date in lookback_dates:
        for bar in flatfiles.iter_bars(
            DAY_AGGS_DATASET, session_date, tickers=tickers
        ):
            daily_history[bar["ticker"]].append(bar)
        for bar in flatfiles.iter_bars(
            MINUTE_AGGS_DATASET, session_date, tickers=tickers
        ):
            observed = datetime.fromisoformat(bar["timestamp"]).astimezone(ET)
            session_freeze = configured_freeze_datetime(
                session_date, active_config
            )
            if (
                observed.date() == date.fromisoformat(session_date)
                and observed < session_freeze
                and observed.time() >= time(4)
            ):
                premarket_history[bar["ticker"]][session_date] += float(
                    bar["volume"]
                )

    for bar in flatfiles.iter_bars(
        MINUTE_AGGS_DATASET, trading_date, tickers=tickers
    ):
        observed = datetime.fromisoformat(bar["timestamp"]).astimezone(ET)
        if observed.date() != target:
            continue
        if time(4) <= observed.time() and observed < freeze:
            target_premarket[bar["ticker"]].append(bar)
        elif time(9, 30) <= observed.time() < time(16):
            target_regular[bar["ticker"]].append(bar)

    metadata = evidence_metadata(universe_manifest)
    snapshot = {
        "replay_id": universe_manifest["replay_id"],
        "trading_date": trading_date,
        "freeze_timestamp": freeze.isoformat(),
        "timezone": "America/New_York",
        "universe_version": universe_manifest["ruleset"]["version"],
        "scout_version": "research_scout_alpha_v1.0",
        "execution_policy_version": "execution_disabled",
        "feature_registry_version": "feature_registry_alpha_v1.0",
        "data_source": {
            "provider": flatfiles.provider_name,
            "feed_version": flatfiles.feed_version,
        },
        "massive_plan": load_massive_plan(),
        **metadata,
        "securities": [],
    }
    outcome_bars: dict[str, list[dict[str, Any]]] = {}
    per_ticker_counts: dict[str, dict[str, int]] = {}
    duplicate_minute_rows: list[dict[str, Any]] = []

    manifest_by_ticker = {item["ticker"]: item for item in securities}
    for ticker in sorted(tickers):
        manifest_security = manifest_by_ticker[ticker]
        prior_rows = sorted(
            daily_history.get(ticker, []),
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
            (_snapshot_bar(bar) for bar in target_premarket.get(ticker, [])),
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
            premarket_history[ticker][value] for value in lookback_dates
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
        }
        security = {
            "ticker": ticker,
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
            target_regular.get(ticker, []),
        )
        duplicate_minute_rows.extend(collisions)
        outcome_bars[ticker] = regular
        per_ticker_counts[ticker] = {
            "real_premarket": len(premarket_bars),
            "real_premarket_60m": real_bar_count_60m,
            "real_regular": len(regular),
        }

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
        "by_ticker": per_ticker_counts,
    }
    return FlatFileSnapshotResult(
        snapshot=snapshot,
        outcome_bars=outcome_bars,
        bar_statistics=bar_stats,
        lookback_dates=lookback_dates,
    )
