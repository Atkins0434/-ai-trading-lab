from __future__ import annotations

from typing import Any

from trainer.outcome_grader import grade_replay_outcomes
from trainer.replay_engine import (
    validate_freeze_timestamp,
    validate_point_in_time_inputs,
)
from trainer.scout_engine import run_scout


def run_single_day_replay(
    snapshot: dict[str, Any],
    outcome_bars_by_ticker: dict[str, list[dict[str, Any]]],
    *,
    threshold_pct: float = 85.0,
    strategy_capital: float = 2500.0,
) -> dict[str, Any]:
    """Run the deterministic Scout-to-outcome path for one frozen day."""
    validate_freeze_timestamp(snapshot)
    validate_point_in_time_inputs(snapshot)
    scout_result = run_scout(snapshot, threshold_pct=threshold_pct)
    outcomes = grade_replay_outcomes(
        snapshot,
        scout_result,
        outcome_bars_by_ticker,
        strategy_capital,
    )
    return {
        "snapshot": snapshot,
        "scout_output": scout_result,
        "end_of_day_outcome": outcomes,
    }
