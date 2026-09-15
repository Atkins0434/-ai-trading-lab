import pytest

from trainer.momentum import (
    MomentumError,
    calculate_momentum,
    calculate_short_term_override,
    linear_regression_slope,
)


def make_bars(
    closes: list[float],
    volumes: list[float],
) -> list[dict]:
    return [
        {
            "timestamp": (
                f"2018-01-02T"
                f"{7 + index // 60:02d}:"
                f"{index % 60:02d}:00-05:00"
            ),
            "close": close,
            "volume": volume,
        }
        for index, (close, volume) in enumerate(
            zip(closes, volumes)
        )
    ]


def test_linear_regression_slope_positive():
    values = [1, 2, 3, 4, 5]

    slope = linear_regression_slope(values)

    assert slope == pytest.approx(1.0)


def test_linear_regression_slope_negative():
    values = [5, 4, 3, 2, 1]

    slope = linear_regression_slope(values)

    assert slope == pytest.approx(-1.0)


def test_linear_regression_slope_flat():
    values = [5, 5, 5, 5, 5]

    slope = linear_regression_slope(values)

    assert slope == pytest.approx(0.0)


def test_momentum_detects_rising_price_and_volume():
    closes = [
        10 + index * 0.05
        for index in range(60)
    ]

    volumes = [
        1000 + index * 100
        for index in range(60)
    ]

    result = calculate_momentum(
        make_bars(closes, volumes)
    )

    assert result.price_15.slope > 0
    assert result.price_30.slope > 0
    assert result.price_60.slope > 0

    assert result.volume_15.slope > 0
    assert result.volume_30.slope > 0
    assert result.volume_60.slope > 0
    assert result.price_15.raw_slope == pytest.approx(0.05)
    assert result.price_15.slope == pytest.approx(
        result.price_15.raw_slope
        / (
            sum(closes[-15:])
            / 15
        )
        * 100
    )


def test_momentum_detects_falling_price():
    closes = [
        20 - index * 0.05
        for index in range(60)
    ]

    volumes = [
        5000
        for _ in range(60)
    ]

    result = calculate_momentum(
        make_bars(closes, volumes)
    )

    assert result.price_15.slope < 0
    assert result.price_30.slope < 0
    assert result.price_60.slope < 0


def test_momentum_detects_flat_price():
    closes = [
        15.0
        for _ in range(60)
    ]

    volumes = [
        2000
        for _ in range(60)
    ]

    result = calculate_momentum(
        make_bars(closes, volumes)
    )

    assert result.price_15.slope == pytest.approx(
        0.0
    )
    assert result.price_30.slope == pytest.approx(
        0.0
    )
    assert result.price_60.slope == pytest.approx(
        0.0
    )


def test_short_term_override_triggers():
    assert calculate_short_term_override(
        price_15_slope=0.06,
        price_30_slope=0.05,
        price_60_slope=0.02,
        override_multiple=2.0,
    )


def test_short_term_override_does_not_trigger_with_negative_60():
    assert not calculate_short_term_override(
        price_15_slope=0.06,
        price_30_slope=0.05,
        price_60_slope=-0.01,
        override_multiple=2.0,
    )


def test_short_term_override_does_not_trigger_when_only_15_is_strong():
    assert not calculate_short_term_override(
        price_15_slope=0.06,
        price_30_slope=0.03,
        price_60_slope=0.02,
        override_multiple=2.0,
    )


def test_insufficient_bars_are_rejected():
    closes = [
        10 + index * 0.05
        for index in range(59)
    ]

    volumes = [
        1000
        for _ in range(59)
    ]

    with pytest.raises(
        MomentumError,
        match="At least 60 one-minute bars are required",
    ):
        calculate_momentum(
            make_bars(closes, volumes)
        )


def test_negative_volume_is_rejected():
    closes = [
        10 + index * 0.01
        for index in range(60)
    ]

    volumes = [
        1000
        for _ in range(60)
    ]

    volumes[-1] = -1

    with pytest.raises(
        MomentumError,
        match="invalid volume",
    ):
        calculate_momentum(
            make_bars(closes, volumes)
        )


def test_missing_required_bar_field_is_rejected():
    closes = [
        10 + index * 0.01
        for index in range(60)
    ]

    volumes = [
        1000
        for _ in range(60)
    ]

    bars = make_bars(
        closes,
        volumes,
    )

    del bars[-1]["volume"]

    with pytest.raises(
        MomentumError,
        match="missing fields",
    ):
        calculate_momentum(bars)
