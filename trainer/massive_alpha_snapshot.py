from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

from trainer.providers.base import ProviderError
from trainer.providers.massive import MassiveClient, canonical_price_bar, parse_massive_timestamp
from trainer.rate_control import load_massive_plan
from trainer.replay_engine import configured_freeze_datetime
from trainer.validate_contracts import load_json
from trainer.validate_contracts import validate_contract


MARKET_TIMEZONE = ZoneInfo("America/New_York")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "scout_alpha_v1.json"


def _observed(value: float, as_of: str, provider_field: str) -> dict[str, Any]:
    return {
        "value": value,
        "as_of_timestamp": as_of,
        "source": MassiveClient.provider_name,
        "provider_field": provider_field,
    }


def _market_datetime(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, tzinfo=MARKET_TIMEZONE)


def real_premarket_bars(
    records: list[dict[str, Any]],
    trading_date: str,
    *,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return chronological real trade-minute bars before the freeze."""
    day = date.fromisoformat(trading_date)
    freeze = configured_freeze_datetime(trading_date, config)
    bars: list[dict[str, Any]] = []
    for record in records:
        timestamp = datetime.fromisoformat(parse_massive_timestamp(record)).astimezone(MARKET_TIMEZONE)
        if timestamp.date() == day and time(4) <= timestamp.time() and timestamp < freeze:
            bars.append(canonical_price_bar(record, include_source=True))
    bars.sort(key=lambda bar: bar["timestamp"])
    return bars


def regular_session_bars(
    records: list[dict[str, Any]], trading_date: str
) -> list[dict[str, Any]]:
    """Return chronological 09:30-15:59 ET bars for outcome grading."""
    day = date.fromisoformat(trading_date)
    result = []
    for record in records:
        observed = datetime.fromisoformat(
            parse_massive_timestamp(record)
        ).astimezone(MARKET_TIMEZONE)
        if observed.date() == day and time(9, 30) <= observed.time() < time(16, 0):
            result.append(canonical_price_bar(record))
    result.sort(key=lambda bar: bar["timestamp"])
    if not result:
        raise ProviderError("Alpha target day has no regular-session records.")
    return result


def build_massive_alpha_snapshot(
    ticker: str,
    trading_date: str,
    daily_records: list[dict[str, Any]],
    intraday_records: list[dict[str, Any]],
    *,
    exchange: str = "UNKNOWN",
    eligible: bool = True,
    eligibility_reasons: list[str] | None = None,
    universe_metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a point-in-time Alpha snapshot from raw Massive aggregates."""
    day = date.fromisoformat(trading_date)
    active_config = config or load_json(CONFIG_PATH)
    freeze = configured_freeze_datetime(trading_date, active_config)
    freeze_label = freeze.strftime("%H%M")
    prior_daily: list[tuple[date, dict[str, Any]]] = []
    for record in daily_records:
        observed_day = datetime.fromisoformat(
            parse_massive_timestamp(record)
        ).astimezone(MARKET_TIMEZONE).date()
        if observed_day < day:
            prior_daily.append((observed_day, record))
    prior_daily.sort(key=lambda item: item[0])
    baseline = prior_daily[-20:]
    if len(baseline) < 20:
        raise ProviderError("Alpha requires 20 prior daily bars.")

    by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for record in intraday_records:
        observed = datetime.fromisoformat(
            parse_massive_timestamp(record)
        ).astimezone(MARKET_TIMEZONE)
        session_freeze = configured_freeze_datetime(
            observed.date().isoformat(), active_config
        )
        if time(4, 0) <= observed.time() and observed < session_freeze:
            by_day[observed.date()].append(record)
    current = by_day[day]

    previous_day, previous = baseline[-1]
    previous_close = float(previous["c"])
    average_daily_dollar_volume = mean(
        float(record["c"]) * float(record["v"]) for _, record in baseline
    )
    average_daily_range_pct = mean(
        (float(record["h"]) - float(record["l"])) / float(record["c"]) * 100
        for _, record in baseline
    )
    baseline_volumes = [sum(float(record["v"]) for record in by_day[d]) for d, _ in baseline]
    average_premarket_volume = mean(baseline_volumes)
    premarket_volume = sum(float(record["v"]) for record in current)
    premarket_dollar_volume = sum(
        ((float(record["h"]) + float(record["l"]) + float(record["c"])) / 3)
        * float(record["v"])
        for record in current
    )
    relative_volume = premarket_volume / average_premarket_volume if average_premarket_volume else 0.0
    bars = real_premarket_bars(
        intraday_records, trading_date, config=active_config
    )
    last_hour_start = freeze - timedelta(minutes=60)
    real_bar_count_60m = sum(
        datetime.fromisoformat(bar["timestamp"]).astimezone(MARKET_TIMEZONE)
        >= last_hour_start
        for bar in bars
    )
    as_of = freeze.isoformat()
    prior_close_as_of = _market_datetime(previous_day, time(16, 0)).isoformat()

    universe_metadata = universe_metadata or {
        "massive_plan": load_massive_plan(),
        "universe_mode": "ci_fixture",
        "universe_manifest_hash": "sha256:" + "0" * 64,
        "universe_coverage": "fixture",
        "research_evidence": False,
        "promotion_eligible": False,
    }
    snapshot = {
        "replay_id": f"{trading_date}-{freeze_label}-{ticker.upper()}-research-alpha",
        "trading_date": trading_date,
        "freeze_timestamp": freeze.isoformat(),
        "timezone": "America/New_York",
        "universe_version": "research_alpha_input_v1.0",
        "scout_version": "research_scout_alpha_v1.0",
        "execution_policy_version": "execution_disabled",
        "feature_registry_version": "feature_registry_alpha_v1.0",
        "data_source": {"provider": "MASSIVE", "feed_version": MassiveClient.feed_version},
        "massive_plan": universe_metadata.get(
            "massive_plan", load_massive_plan()
        ),
        **universe_metadata,
        "securities": [{
            "ticker": ticker.upper(),
            "exchange": exchange,
            "eligible": eligible,
            "eligibility_as_of_timestamp": prior_close_as_of,
            "eligibility_reasons": eligibility_reasons or [],
            "market_data": {
                "last_price": _observed(float(bars[-1]["close"]) if bars else previous_close, bars[-1]["timestamp"] if bars else prior_close_as_of, "c"),
                "premarket_volume": _observed(premarket_volume, as_of, f"sum(v),04:00-{freeze.time().isoformat()}ET"),
                "premarket_dollar_volume": _observed(premarket_dollar_volume, as_of, "sum(typical_price*v)"),
                "average_daily_dollar_volume": _observed(average_daily_dollar_volume, prior_close_as_of, "mean(c*v),20_sessions"),
                "relative_volume": _observed(relative_volume, as_of, "premarket_volume/20_session_mean"),
                "real_bar_count": _observed(
                    len(bars),
                    as_of,
                    "count(real_trade_minutes),04:00-freeze",
                ),
                "real_bar_count_60m": _observed(
                    real_bar_count_60m,
                    as_of,
                    "count(real_trade_minutes),last_60m_before_freeze",
                ),
                "previous_close": _observed(previous_close, prior_close_as_of, "c"),
                "average_daily_range_pct": _observed(average_daily_range_pct, prior_close_as_of, "mean((h-l)/c),20_sessions"),
            },
            "premarket_bars": bars,
            "news": [],
            "filings": [],
            "analyst_actions": [],
            "social_observations": [],
        }],
    }
    validate_contract("historical_snapshot", snapshot)
    return snapshot
