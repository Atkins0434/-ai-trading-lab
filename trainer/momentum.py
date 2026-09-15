from __future__ import annotations

from dataclasses import dataclass
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
    price_15: WindowSlope
    price_30: WindowSlope
    price_60: WindowSlope
    volume_15: WindowSlope
    volume_30: WindowSlope
    volume_60: WindowSlope
    short_term_override: bool


def _validate_series(
    values: list[float],
    window_minutes: int,
) -> None:
    if window_minutes <= 1:
        raise MomentumError(
            "Window must contain at least two observations."
        )

    if len(values) < window_minutes:
        raise MomentumError(
            f"Need at least {window_minutes} observations, "
            f"received {len(values)}."
        )


def linear_regression_slope(
    values: list[float],
) -> float:
    """
    Calculate ordinary least-squares slope.

    X values are 0, 1, 2, ... n-1.
    """
    if len(values) < 2:
        raise MomentumError(
            "At least two observations are required."
        )

    n = len(values)

    mean_x = (n - 1) / 2
    mean_y = sum(values) / n

    numerator = 0.0
    denominator = 0.0

    for index, value in enumerate(values):
        x_delta = index - mean_x
        y_delta = value - mean_y

        numerator += x_delta * y_delta
        denominator += x_delta * x_delta

    if denominator == 0:
        raise MomentumError(
            "Regression denominator cannot be zero."
        )

    return numerator / denominator


def calculate_window_slope(
    values: list[float],
    window_minutes: int,
) -> WindowSlope:
    """
    Calculate slope using the most recent N observations.
    """
    _validate_series(
        values,
        window_minutes,
    )

    window = values[-window_minutes:]

    raw_slope = linear_regression_slope(window)
    baseline = sum(window) / len(window)

    if baseline == 0 and raw_slope == 0:
        normalized_slope = 0.0
    elif baseline <= 0:
        raise MomentumError(
            "Normalized slope requires a positive window mean."
        )
    else:
        normalized_slope = (raw_slope / baseline) * 100

    return WindowSlope(
        window_minutes=window_minutes,
        raw_slope=raw_slope,
        slope=normalized_slope,
        first_value=window[0],
        last_value=window[-1],
        observation_count=len(window),
    )


def _validate_bars(
    bars: list[dict[str, Any]],
) -> None:
    if len(bars) < 60:
        raise MomentumError(
            "At least 60 one-minute bars are required."
        )

    required = {
        "timestamp",
        "close",
        "volume",
    }

    for index, bar in enumerate(bars):
        missing = required - set(bar)

        if missing:
            raise MomentumError(
                f"Bar {index} is missing fields: "
                f"{sorted(missing)}"
            )

        if bar["close"] <= 0:
            raise MomentumError(
                f"Bar {index} has invalid close price."
            )

        if bar["volume"] < 0:
            raise MomentumError(
                f"Bar {index} has invalid volume."
            )


def calculate_short_term_override(
    price_15_slope: float,
    price_30_slope: float,
    price_60_slope: float,
    override_multiple: float = 2.0,
) -> bool:
    """
    Allow legitimate sudden acceleration when:

    - 60-minute price slope is positive
    - 15-minute slope is at least 2x the 60-minute slope
    - 30-minute slope is at least 2x the 60-minute slope

    This matches the Scout design rule that strong recent
    acceleration may outweigh a slower positive 60-minute trend.
    """
    if override_multiple <= 0:
        raise MomentumError(
            "Override multiple must be positive."
        )

    if price_60_slope <= 0:
        return False

    return (
        price_15_slope
        >= price_60_slope * override_multiple
        and price_30_slope
        >= price_60_slope * override_multiple
    )


def calculate_momentum(
    bars: list[dict[str, Any]],
    override_multiple: float = 2.0,
) -> MomentumResult:
    """
    Calculate Scout's normalized 15/30/60 minute price momentum
    and volume acceleration slopes as percent change per minute.

    Bars must be one-minute bars ordered oldest to newest.
    """
    _validate_bars(bars)

    closes = [
        float(bar["close"])
        for bar in bars
    ]

    volumes = [
        float(bar["volume"])
        for bar in bars
    ]

    price_15 = calculate_window_slope(
        closes,
        15,
    )

    price_30 = calculate_window_slope(
        closes,
        30,
    )

    price_60 = calculate_window_slope(
        closes,
        60,
    )

    volume_15 = calculate_window_slope(
        volumes,
        15,
    )

    volume_30 = calculate_window_slope(
        volumes,
        30,
    )

    volume_60 = calculate_window_slope(
        volumes,
        60,
    )

    short_term_override = (
        calculate_short_term_override(
            price_15.slope,
            price_30.slope,
            price_60.slope,
            override_multiple,
        )
    )

    return MomentumResult(
        price_15=price_15,
        price_30=price_30,
        price_60=price_60,
        volume_15=volume_15,
        volume_30=volume_30,
        volume_60=volume_60,
        short_term_override=short_term_override,
    )
