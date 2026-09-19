from __future__ import annotations

from typing import Iterable


METRIC_IDS = (
    "relative_volume",
    "price_slope_15m",
    "price_slope_30m",
    "price_slope_60m",
    "volume_slope_15m",
    "volume_slope_30m",
    "volume_slope_60m",
    "premarket_gap_strength",
    "price_vs_premarket_vwap",
    "premarket_trend_consistency",
    "price_volume_confirmation",
    "premarket_range_expansion",
    "relative_strength_index_15m",
    "relative_strength_index_30m",
    "relative_strength_index_60m",
    "atr_pct_opportunity",
    "current_range_pct_opportunity",
    "average_daily_dollar_volume_quality",
    "premarket_dollar_volume_quality",
)

SCORABLE_PREFIX_COLUMNS = (
    "trading_date", "ticker", "stable_security_id", "shadow",
    "premarket_real_bars_60m", "premarket_real_bars_total",
    "prior_close", "price_at_freeze", "premarket_dollar_volume_usd",
    "average_daily_dollar_volume_usd", "market_cap_usd",
)
SCORABLE_SCORE_COLUMNS = tuple(
    column
    for metric_id in METRIC_IDS
    for column in (f"{metric_id}_raw", f"{metric_id}_score")
)
SCORABLE_OUTCOME_COLUMNS = (
    "total_score", "score_pct", "alpha12_total_score", "alpha12_score_pct",
    "rubric_version", "reversal_score_pct", "selected",
    "selection_basis", "rejection_reasons", "liquidity_action",
    "top_10_mover", "benchmark_rank", "open_0930", "day_high",
    "day_low", "close", "day_mfe_pct", "day_mae_pct",
    "close_vs_open_pct", "time_of_day_high", "high_0930_1000_pct",
    "return_0930_1000_pct", "no_regular_session_path",
)

ELIGIBLE_OUTCOMES_COLUMNS = (
    "trading_date", "ticker", "stable_security_id",
    "premarket_real_bars_60m", "scorable", "open_0930", "close",
    "day_mfe_pct", "close_vs_open_pct", "top_10_mover",
    "realized_return_pct",
)


def policy_columns(policy_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        column
        for policy_id in policy_ids
        for column in (
            f"{policy_id}_exit_reason",
            f"{policy_id}_realized_return_pct",
            f"{policy_id}_capture_ratio",
            f"{policy_id}_stop_distance_pct",
        )
    )


def scorable_outcomes_columns(policy_ids: Iterable[str]) -> tuple[str, ...]:
    return (
        SCORABLE_PREFIX_COLUMNS
        + SCORABLE_SCORE_COLUMNS
        + SCORABLE_OUTCOME_COLUMNS
        + policy_columns(policy_ids)
    )
