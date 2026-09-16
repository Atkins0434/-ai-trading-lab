from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

from trainer.providers.base import ProviderError
from trainer.providers.massive import MassiveClient, canonical_price_bar, parse_massive_timestamp
from trainer.rate_control import load_massive_plan
from trainer.validate_contracts import validate_contract


MARKET_TIMEZONE = ZoneInfo("America/New_York")


def _observed(value: float, as_of: str, provider_field: str) -> dict[str, Any]:
    return {
        "value": value,
        "as_of_timestamp": as_of,
        "source": MassiveClient.provider_name,
        "provider_field": provider_field,
    }


def _market_datetime(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, tzinfo=MARKET_TIMEZONE)


def regularize_last_premarket_hour(
    records: list[dict[str, Any]], trading_date: str
) -> list[dict[str, Any]]:
    """Return 60 auditable 06:00-06:59 ET bars using zero-volume carry-forward fills."""
    day = date.fromisoformat(trading_date)
    start = _market_datetime(day, time(6, 0))
    end = _market_datetime(day, time(7, 0))
    observed: dict[datetime, dict[str, Any]] = {}
    seed: float | None = None
    for record in records:
        timestamp = datetime.fromisoformat(parse_massive_timestamp(record)).astimezone(MARKET_TIMEZONE)
        if timestamp.date() != day:
            continue
        minute = timestamp.replace(second=0, microsecond=0)
        if minute < start:
            seed = float(record["c"])
        elif start <= minute < end:
            observed[minute] = record

    bars: list[dict[str, Any]] = []
    last_close = seed
    for offset in range(60):
        minute = start + timedelta(minutes=offset)
        record = observed.get(minute)
        if record is not None:
            bar = canonical_price_bar(record, include_source=True)
            last_close = float(bar["close"])
        else:
            if last_close is None:
                raise ProviderError("Cannot fill leading Alpha minute without an earlier trade.")
            bar = {
                "timestamp": minute.isoformat(),
                "open": last_close,
                "high": last_close,
                "low": last_close,
                "close": last_close,
                "volume": 0,
                "source": "MASSIVE_ZERO_VOLUME_CARRY_FORWARD",
            }
        bars.append(bar)
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
) -> dict[str, Any]:
    """Build a point-in-time Alpha snapshot from raw Massive aggregates."""
    day = date.fromisoformat(trading_date)
    freeze = _market_datetime(day, time(7, 0))
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
        if time(4, 0) <= observed.time() < time(7, 0):
            by_day[observed.date()].append(record)
    current = by_day[day]
    if not current:
        raise ProviderError("Alpha target day has no premarket records.")

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
    bars = regularize_last_premarket_hour(intraday_records, trading_date)
    padded_bar_count = sum(
        bar["source"] == "MASSIVE_ZERO_VOLUME_CARRY_FORWARD"
        for bar in bars
    )
    as_of = bars[-1]["timestamp"]
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
        "replay_id": f"{trading_date}-0700-{ticker.upper()}-research-alpha",
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
                "last_price": _observed(float(bars[-1]["close"]), as_of, "c"),
                "premarket_volume": _observed(premarket_volume, as_of, "sum(v),04:00-07:00ET"),
                "premarket_dollar_volume": _observed(premarket_dollar_volume, as_of, "sum(typical_price*v)"),
                "average_daily_dollar_volume": _observed(average_daily_dollar_volume, prior_close_as_of, "mean(c*v),20_sessions"),
                "relative_volume": _observed(relative_volume, as_of, "premarket_volume/20_session_mean"),
                "padded_bar_count": _observed(
                    padded_bar_count,
                    as_of,
                    "count(MASSIVE_ZERO_VOLUME_CARRY_FORWARD),06:00-07:00ET",
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
