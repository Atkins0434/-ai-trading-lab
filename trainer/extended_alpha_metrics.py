"""Research-only metrics calculated entirely from frozen snapshot observations."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from trainer.price_volume import score_thresholds

EXTENDED_METRIC_IDS = (
    "relative_strength_index_15m", "relative_strength_index_30m",
    "relative_strength_index_60m", "atr_pct_opportunity",
    "current_range_pct_opportunity", "average_daily_dollar_volume_quality",
    "premarket_dollar_volume_quality",
)


def cutler_window_rsi(closes: list[float]) -> float:
    if len(closes) < 2:
        raise ValueError("RSI needs at least one observed close-to-close change")
    changes = [b - a for a, b in zip(closes, closes[1:])]
    gain = sum(max(change, 0.0) for change in changes) / len(changes)
    loss = sum(max(-change, 0.0) for change in changes) / len(changes)
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


def _observation(security: dict, key: str) -> tuple[Any, str | None]:
    observation = security.get("market_data", {}).get(key, {})
    return observation.get("value"), observation.get("as_of_timestamp")


def extended_components(security: dict, config: dict, freeze_timestamp: str) -> dict:
    rules = config["extended_metric_thresholds"]
    if rules["rsi_method"] != "cutler_window":
        raise ValueError("Unsupported research RSI method")
    freeze = datetime.fromisoformat(freeze_timestamp.replace("Z", "+00:00"))
    # Snapshot inputs contain only real trade bars; never pad missing minutes.
    bars = security.get("premarket_bars", [])
    values = {}
    for window in (15, 30, 60):
        start = freeze - timedelta(minutes=window)
        selected = [bar for bar in bars if start <= datetime.fromisoformat(bar["timestamp"].replace("Z", "+00:00")) < freeze]
        enough = len(selected) >= config["price_momentum"]["minimum_bars_per_window"][str(window)]
        values[f"relative_strength_index_{window}m"] = (
            cutler_window_rsi([float(bar["close"]) for bar in selected]) if enough else None,
            selected[-1]["timestamp"] if enough else None,
        )
    prior, prior_at = _observation(security, "previous_close")
    atr, atr_at = _observation(security, "atr_14_usd")
    valid_prior = prior is not None and float(prior) > 0
    values["atr_pct_opportunity"] = (
        float(atr) / float(prior) * 100 if atr is not None and valid_prior else None,
        max(atr_at, prior_at, key=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")))
        if atr_at and prior_at else atr_at or prior_at,
    )
    # Same absolute range as expansion's numerator, even if its historical
    # average denominator is unavailable (so expansion itself is MISSING).
    values["current_range_pct_opportunity"] = (
        (max(float(bar["high"]) for bar in bars) - min(float(bar["low"]) for bar in bars)) / float(prior) * 100
        if bars and valid_prior else None,
        bars[-1]["timestamp"] if bars else None,
    )
    values["average_daily_dollar_volume_quality"] = _observation(security, "average_daily_dollar_volume")
    values["premarket_dollar_volume_quality"] = _observation(security, "premarket_dollar_volume")
    result = {}
    for metric_id, (raw, timestamp) in values.items():
        missing = raw is None
        result[metric_id] = {
            "raw_value": raw,
            "score": None if missing else score_thresholds(raw, rules[metric_id]),
            "status": "MISSING" if missing else "SCORED",
            "maximum_score": 4,
            "as_of_timestamp": None if missing else timestamp,
            "reason_code": "REQUIRED_ALPHA_INPUT_MISSING" if missing else f"{metric_id.upper()}_SCORE",
            "calculation_version": "rsi_cutler_window_v1.0" if metric_id.startswith("relative_strength_index_") else "research_alpha_extended_v1.0",
        }
    return result
