from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


class MomentumError(Exception):
    """Raised when momentum calculations cannot be completed."""


@dataclass(frozen=True)
class WindowSlope:
    window_minutes: int
    raw_slope: float
    slope: float
    first_value: float
    last_value: float
    observation_count: int


@dataclass(frozen=True)
class MomentumResult:
    price_15: WindowSlope | None
    price_30: WindowSlope | None
    price_60: WindowSlope | None
    volume_15: WindowSlope | None
    volume_30: WindowSlope | None
    volume_60: WindowSlope | None
    short_term_override: bool


def _parse_timestamp(raw: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise MomentumError(f"Invalid bar timestamp: {raw}") from exc
    if parsed.tzinfo is None:
        raise MomentumError("Bar timestamps require timezone information.")
    return parsed


def linear_regression_slope(
    values: list[float], x_values: list[float] | None = None
) -> float:
    """Calculate an OLS slope over explicit elapsed-minute coordinates."""
    if len(values) < 2:
        raise MomentumError("At least two observations are required.")
    xs = (
        [float(index) for index in range(len(values))]
        if x_values is None
        else x_values
    )
    if len(xs) != len(values):
        raise MomentumError("Regression x and y values must have equal length.")
    if len(set(xs)) != len(xs):
        raise MomentumError("Regression x values must be unique.")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(values) / len(values)
    numerator = sum(
        (x - mean_x) * (value - mean_y)
        for x, value in zip(xs, values)
    )
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise MomentumError("Regression denominator cannot be zero.")
    return numerator / denominator


def _window_slope(
    bars: list[dict[str, Any]],
    field: str,
    window_minutes: int,
    freeze: datetime,
    minimum_count: int,
) -> WindowSlope | None:
    start = freeze - timedelta(minutes=window_minutes)
    window = [
        bar
        for bar in bars
        if start <= _parse_timestamp(bar["timestamp"]) < freeze
    ]
    if len(window) < minimum_count:
        return None
    values = [float(bar[field]) for bar in window]
    x_values = [
        (_parse_timestamp(bar["timestamp"]) - start).total_seconds() / 60
        for bar in window
    ]
    raw_slope = linear_regression_slope(values, x_values)
    baseline = sum(values) / len(values)
    if baseline == 0 and raw_slope == 0:
        normalized = 0.0
    elif baseline <= 0:
        raise MomentumError("Normalized slope requires a positive window mean.")
    else:
        normalized = raw_slope / baseline * 100
    return WindowSlope(
        window_minutes=window_minutes,
        raw_slope=raw_slope,
        slope=normalized,
        first_value=values[0],
        last_value=values[-1],
        observation_count=len(values),
    )


def _validate_bars(bars: list[dict[str, Any]]) -> None:
    required = {"timestamp", "close", "volume"}
    for index, bar in enumerate(bars):
        missing = required - set(bar)
        if missing:
            raise MomentumError(f"Bar {index} is missing fields: {sorted(missing)}")
        _parse_timestamp(bar["timestamp"])
        if bar["close"] <= 0:
            raise MomentumError(f"Bar {index} has invalid close price.")
        if bar["volume"] < 0:
            raise MomentumError(f"Bar {index} has invalid volume.")


def calculate_short_term_override(
    price_15_slope: float,
    price_30_slope: float,
    price_60_slope: float,
    override_multiple: float = 2.0,
) -> bool:
    if override_multiple <= 0:
        raise MomentumError("Override multiple must be positive.")
    if price_60_slope <= 0:
        return False
    return (
        price_15_slope >= price_60_slope * override_multiple
        and price_30_slope >= price_60_slope * override_multiple
    )


def calculate_momentum(
    bars: list[dict[str, Any]],
    *,
    freeze_timestamp: str | None = None,
    minimum_bars_per_window: dict[str, int] | None = None,
    override_multiple: float = 2.0,
) -> MomentumResult:
    """Calculate sparse-aware normalized slopes in elapsed clock minutes."""
    _validate_bars(bars)
    if not bars:
        raise MomentumError("At least one real premarket bar is required.")
    freeze = (
        _parse_timestamp(freeze_timestamp)
        if freeze_timestamp is not None
        else _parse_timestamp(bars[-1]["timestamp"]) + timedelta(minutes=1)
    )
    minimums = minimum_bars_per_window or {"15": 5, "30": 10, "60": 20}
    for window in (15, 30, 60):
        minimum = minimums.get(str(window))
        if (
            not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or minimum < 2
        ):
            raise MomentumError(
                f"minimum_bars_per_window[{window}] must be an integer >= 2."
            )

    slopes: dict[str, WindowSlope | None] = {}
    for field in ("close", "volume"):
        for window in (15, 30, 60):
            slopes[f"{field}_{window}"] = _window_slope(
                bars, field, window, freeze, minimums[str(window)]
            )
    p15 = slopes["close_15"]
    p30 = slopes["close_30"]
    p60 = slopes["close_60"]
    override = bool(
        p15 is not None
        and p30 is not None
        and p60 is not None
        and calculate_short_term_override(
            p15.slope, p30.slope, p60.slope, override_multiple
        )
    )
    return MomentumResult(
        price_15=p15,
        price_30=p30,
        price_60=p60,
        volume_15=slopes["volume_15"],
        volume_30=slopes["volume_30"],
        volume_60=slopes["volume_60"],
        short_term_override=override,
    )
