from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from trainer.momentum import MomentumError, calculate_momentum


class PriceVolumeError(Exception):
    """Raised when premarket price/volume inputs are malformed."""


@dataclass(frozen=True)
class MetricValue:
    """One calculated, but not yet contract-formatted, Scout metric."""

    raw_value: Any
    score: int
    as_of_timestamp: str
    reason_code: str


def _parse_timestamp(raw: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise PriceVolumeError(f"Invalid bar timestamp: {raw}") from exc
    if parsed.tzinfo is None:
        raise PriceVolumeError("Premarket bar timestamps require a timezone.")
    return parsed


def validate_minute_bars(bars: list[dict[str, Any]]) -> None:
    """Validate chronological, contiguous, one-minute OHLCV bars."""
    if len(bars) < 60:
        raise PriceVolumeError("At least 60 premarket minute bars are required.")

    previous_timestamp: datetime | None = None
    for index, bar in enumerate(bars):
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(bar)
        if missing:
            raise PriceVolumeError(
                f"Bar {index} is missing fields: {sorted(missing)}"
            )

        timestamp = _parse_timestamp(bar["timestamp"])
        if previous_timestamp is not None:
            seconds = (timestamp - previous_timestamp).total_seconds()
            if seconds != 60:
                raise PriceVolumeError(
                    "Premarket bars must be ordered and contiguous at "
                    "one-minute intervals."
                )
        previous_timestamp = timestamp

        values = [bar["open"], bar["high"], bar["low"], bar["close"]]
        if any(not isinstance(value, (int, float)) or value <= 0 for value in values):
            raise PriceVolumeError(f"Bar {index} contains an invalid price.")
        if bar["high"] < max(bar["open"], bar["low"], bar["close"]):
            raise PriceVolumeError(f"Bar {index} high is inconsistent with OHLC.")
        if bar["low"] > min(bar["open"], bar["high"], bar["close"]):
            raise PriceVolumeError(f"Bar {index} low is inconsistent with OHLC.")
        if not isinstance(bar["volume"], (int, float)) or bar["volume"] < 0:
            raise PriceVolumeError(f"Bar {index} contains invalid volume.")


def score_thresholds(value: float, thresholds: dict[str, float]) -> int:
    """Score a value against inclusive, monotonically increasing cutoffs."""
    score = 0
    previous = float("-inf")
    for points in range(1, 5):
        threshold = float(thresholds[str(points)])
        if threshold < previous:
            raise PriceVolumeError("Metric thresholds must be monotonic.")
        previous = threshold
        if value >= threshold:
            score = points
    return score


def _observation_value(
    market_data: dict[str, Any],
    field: str,
) -> float | None:
    observation = market_data.get(field)
    if observation is None:
        return None
    value = observation.get("value")
    return float(value) if value is not None else None


def calculate_price_volume_metrics(
    security: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, MetricValue]:
    """Calculate Scout metrics 2-12 from frozen premarket inputs.

    Relative volume (metric 1) remains calculated from its explicitly
    timestamped snapshot observation. This function owns the remaining
    Price & Volume Dynamics family.
    """
    bars = security.get("premarket_bars", [])
    if not bars:
        return {}
    validate_minute_bars(bars)

    try:
        momentum = calculate_momentum(bars)
    except MomentumError as exc:
        raise PriceVolumeError(str(exc)) from exc

    as_of = bars[-1]["timestamp"]
    thresholds = config["price_volume_metric_thresholds"]
    results: dict[str, MetricValue] = {}

    slopes = {
        "price_slope_15m": momentum.price_15.slope,
        "price_slope_30m": momentum.price_30.slope,
        "price_slope_60m": momentum.price_60.slope,
        "volume_slope_15m": momentum.volume_15.slope,
        "volume_slope_30m": momentum.volume_30.slope,
        "volume_slope_60m": momentum.volume_60.slope,
    }
    for metric_id, value in slopes.items():
        metric_thresholds = (
            config["price_momentum"]["score_thresholds_pct_per_minute"]
            if metric_id.startswith("price")
            else thresholds["volume_slope"]
        )
        score = score_thresholds(value, metric_thresholds)
        if (
            metric_id.startswith("price_slope")
            and momentum.short_term_override
            and metric_id in {"price_slope_15m", "price_slope_30m"}
        ):
            score = 4
        results[metric_id] = MetricValue(
            raw_value=value,
            score=score,
            as_of_timestamp=as_of,
            reason_code=f"{metric_id.upper()}_SCORE",
        )

    market_data = security["market_data"]
    previous_close = _observation_value(market_data, "previous_close")
    average_daily_range_pct = _observation_value(
        market_data, "average_daily_range_pct"
    )

    if previous_close is not None and previous_close > 0:
        gap_pct = ((bars[-1]["close"] - previous_close) / previous_close) * 100
        results["premarket_gap_strength"] = MetricValue(
            raw_value=gap_pct,
            score=score_thresholds(gap_pct, thresholds["premarket_gap_pct"]),
            as_of_timestamp=as_of,
            reason_code="PREMARKET_GAP_STRENGTH_SCORE",
        )

    total_volume = sum(float(bar["volume"]) for bar in bars)
    if total_volume > 0:
        vwap = sum(
            ((bar["high"] + bar["low"] + bar["close"]) / 3)
            * float(bar["volume"])
            for bar in bars
        ) / total_volume
        price_vs_vwap_pct = ((bars[-1]["close"] - vwap) / vwap) * 100
        results["price_vs_premarket_vwap"] = MetricValue(
            raw_value={"vwap": vwap, "price_vs_vwap_pct": price_vs_vwap_pct},
            score=score_thresholds(
                price_vs_vwap_pct, thresholds["price_vs_vwap_pct"]
            ),
            as_of_timestamp=as_of,
            reason_code="PRICE_VS_PREMARKET_VWAP_SCORE",
        )

    deltas = [
        float(current["close"]) - float(previous["close"])
        for previous, current in zip(bars, bars[1:])
    ]
    positive_share_pct = (
        sum(delta > 0 for delta in deltas) / len(deltas) * 100
    )
    results["premarket_trend_consistency"] = MetricValue(
        raw_value=positive_share_pct,
        score=score_thresholds(
            positive_share_pct, thresholds["trend_consistency_pct"]
        ),
        as_of_timestamp=as_of,
        reason_code="PREMARKET_TREND_CONSISTENCY_SCORE",
    )

    aligned_windows = sum(
        price.slope > 0 and volume.slope > 0
        for price, volume in (
            (momentum.price_15, momentum.volume_15),
            (momentum.price_30, momentum.volume_30),
            (momentum.price_60, momentum.volume_60),
        )
    )
    confirmation_score = int(aligned_windows)
    if aligned_windows == 3 and momentum.short_term_override:
        confirmation_score = 4
    results["price_volume_confirmation"] = MetricValue(
        raw_value={
            "aligned_positive_windows": aligned_windows,
            "short_term_override": momentum.short_term_override,
        },
        score=confirmation_score,
        as_of_timestamp=as_of,
        reason_code="PRICE_VOLUME_CONFIRMATION_SCORE",
    )

    if average_daily_range_pct is not None and average_daily_range_pct > 0:
        reference_price = previous_close or float(bars[0]["open"])
        premarket_range_pct = (
            (max(bar["high"] for bar in bars) - min(bar["low"] for bar in bars))
            / reference_price
            * 100
        )
        expansion_ratio = premarket_range_pct / average_daily_range_pct
        results["premarket_range_expansion"] = MetricValue(
            raw_value={
                "premarket_range_pct": premarket_range_pct,
                "average_daily_range_pct": average_daily_range_pct,
                "expansion_ratio": expansion_ratio,
            },
            score=score_thresholds(
                expansion_ratio, thresholds["range_expansion_ratio"]
            ),
            as_of_timestamp=as_of,
            reason_code="PREMARKET_RANGE_EXPANSION_SCORE",
        )

    return results
