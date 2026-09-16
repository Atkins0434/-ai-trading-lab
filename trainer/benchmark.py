from __future__ import annotations

import random
from statistics import mean
from typing import Any

from trainer.evidence_eligibility import EvidenceEligibilityError, assert_matching_universe
from trainer.trade_engine import load_execution_policy, simulate_trade
from trainer.validate_contracts import ContractError, validate_contract


RANDOM_BASELINE_SEED = 20260916
RANDOM_BASELINE_DRAWS = 200
QUALIFYING_THRESHOLD = "QUALIFYING_THRESHOLD"
EXPLORATION_TOP_K = "EXPLORATION_TOP_K"
NOT_SELECTED = "NOT_SELECTED"


class BenchmarkError(Exception):
    """Raised when a same-universe benchmark cannot be built."""


def _summary(
    executions: list[dict[str, Any]],
    candidate_count: int,
    strategy_capital: float,
) -> dict[str, Any]:
    completed = [item for item in executions if item.get("trade_executed")]
    returns = [float(item["realized_return_pct"]) for item in completed]
    pnls = [float(item["realized_pnl_usd"]) for item in completed]
    gains = sum(value for value in pnls if value > 0)
    losses = abs(sum(value for value in pnls if value < 0))
    ratios = [
        float(item["capture_ratio"])
        for item in completed
        if item.get("capture_ratio") is not None
    ]
    drawdowns = [
        float(item["maximum_position_drawdown_pct"])
        for item in completed
        if item.get("maximum_position_drawdown_pct") is not None
    ]
    return {
        "candidate_count": candidate_count,
        "trades_executed": len(completed),
        "realized_return_pct": sum(pnls) / strategy_capital * 100,
        "realized_pnl_usd": sum(pnls),
        "win_rate_pct": (
            sum(value > 0 for value in returns) / len(returns) * 100
            if returns
            else None
        ),
        "profit_factor": gains / losses if losses else None,
        "max_drawdown_pct": min(drawdowns) if drawdowns else None,
        "average_capture_ratio": mean(ratios) if ratios else None,
    }


def _universe_eligibility(
    snapshot: dict[str, Any], scout_result: dict[str, Any]
) -> dict[str, bool]:
    """Return pre-Scout universe eligibility, not Scout guardrail eligibility."""
    snapshot_securities = snapshot.get("securities", [])
    if snapshot_securities:
        return {
            item["ticker"]: bool(item.get("eligible", False))
            for item in snapshot_securities
        }
    # Compatibility for synthetic callers that predate snapshot securities.
    # Production historical snapshots always take the first branch.
    return {
        item["ticker"]: bool(
            item.get("universe_eligible", item.get("research_eligible", False))
        )
        for item in scout_result["candidates"]
    }


def _simulate_outcome(
    outcome: dict[str, Any], strategy_capital: float
) -> dict[str, Any]:
    bars = outcome["intraday_path"]
    raw = simulate_trade(
        outcome["ticker"], float(bars[0]["open"]), bars, strategy_capital
    )
    maximum_move = float(outcome["maximum_capturable_move_pct"])
    return {
        **raw,
        "capture_ratio": (
            raw["realized_return_pct"] / maximum_move
            if maximum_move > 0
            else None
        ),
        "maximum_position_drawdown_pct": outcome["mae_pct"],
    }


def _return_baselines(
    eligible_outcomes: list[dict[str, Any]],
    selected_count: int,
    strategy_capital: float,
) -> dict[str, Any]:
    """Build deterministic same-policy baselines from the eligible universe."""
    policy = load_execution_policy()
    ordered = sorted(eligible_outcomes, key=lambda item: item["ticker"])
    executions = [_simulate_outcome(item, strategy_capital) for item in ordered]
    if selected_count > len(executions):
        raise BenchmarkError(
            "Scout selected count exceeds the same-day eligible universe."
        )

    per_ticker_pnls = [float(item["realized_pnl_usd"]) for item in executions]
    eligible_mean_pnl = mean(per_ticker_pnls) if per_ticker_pnls else 0.0
    eligible_single_position_mean_return = (
        eligible_mean_pnl / strategy_capital * 100
    )
    eligible_basket_expected_pnl = eligible_mean_pnl * min(
        selected_count, policy.max_positions
    )
    eligible_basket_expected_return = (
        eligible_basket_expected_pnl / strategy_capital * 100
    )

    rng = random.Random(RANDOM_BASELINE_SEED)
    draw_pnls: list[float] = []
    for _ in range(RANDOM_BASELINE_DRAWS):
        if selected_count == 0:
            draw_pnls.append(0.0)
            continue
        draw = rng.sample(executions, selected_count)
        # The draw contains N names, but the same execution policy still caps
        # the number of positions that can actually consume capital.
        executed_draw = draw[: policy.max_positions]
        draw_pnls.append(
            sum(float(item["realized_pnl_usd"]) for item in executed_draw)
        )
    random_mean_pnl = mean(draw_pnls)
    random_mean_return = random_mean_pnl / strategy_capital * 100
    return {
        "return_basis": "GROSS_STRATEGY_RETURN_PCT",
        "eligible_ticker_count": len(executions),
        "eligible_single_position_mean_return_pct": (
            eligible_single_position_mean_return
        ),
        "eligible_ticker_mean_realized_pnl_usd": eligible_mean_pnl,
        "eligible_basket_expected_return_pct": (
            eligible_basket_expected_return
        ),
        "eligible_basket_expected_pnl_usd": eligible_basket_expected_pnl,
        "random_draw_count": RANDOM_BASELINE_DRAWS,
        "random_draw_size": selected_count,
        "random_seed": RANDOM_BASELINE_SEED,
        "random_draw_mean_realized_return_pct": random_mean_return,
        "random_draw_mean_realized_pnl_usd": random_mean_pnl,
    }


def _day_high_timestamp(outcome: dict[str, Any]) -> str:
    highest = max(float(bar["high"]) for bar in outcome["intraday_path"])
    return next(
        bar["timestamp"]
        for bar in outcome["intraday_path"]
        if float(bar["high"]) == highest
    )


def _selection_basis(outcome: dict[str, Any]) -> str:
    """Read the explicit research basis with legacy qualifying compatibility."""
    basis = outcome.get("selection_basis")
    if basis is None:
        return QUALIFYING_THRESHOLD if outcome.get("selected") else NOT_SELECTED
    if basis not in {QUALIFYING_THRESHOLD, EXPLORATION_TOP_K, NOT_SELECTED}:
        raise BenchmarkError(
            f"Unknown selection basis for {outcome['ticker']}: {basis}"
        )
    return basis


def build_same_universe_benchmark(
    snapshot: dict[str, Any],
    scout_result: dict[str, Any],
    outcome_result: dict[str, Any],
    strategy_capital: float,
) -> dict[str, Any]:
    """Grade Scout against random same-universe baskets; retain top-10 diagnostics."""
    try:
        universe_metadata = assert_matching_universe(
            snapshot, scout_result, outcome_result
        )
    except EvidenceEligibilityError as exc:
        raise BenchmarkError(str(exc)) from exc

    policy = load_execution_policy()
    eligibility = _universe_eligibility(snapshot, scout_result)
    eligible_outcomes = [
        item
        for item in outcome_result["outcomes"]
        if eligibility.get(item["ticker"], False)
    ]

    top_outcomes = sorted(
        outcome_result["outcomes"],
        key=lambda item: (-item["mfe_pct"], item["ticker"]),
    )[:10]
    benchmark_candidates: list[dict[str, Any]] = []
    benchmark_executions: list[dict[str, Any]] = []
    simulated_positions = 0
    for rank, outcome in enumerate(top_outcomes, start=1):
        execution = None
        exclusion = None
        universe_eligible = eligibility.get(outcome["ticker"], False)
        if not universe_eligible:
            exclusion = "NOT_IN_ELIGIBLE_UNIVERSE"
        elif simulated_positions < policy.max_positions:
            execution = _simulate_outcome(outcome, strategy_capital)
            benchmark_executions.append(execution)
            simulated_positions += 1
        else:
            exclusion = "MAX_POSITIONS_REACHED"

        scout_execution = outcome["execution_result"]
        benchmark_candidates.append(
            {
                "ticker": outcome["ticker"],
                "benchmark_rank": rank,
                "raw_move_pct": outcome["mfe_pct"],
                "maximum_capturable_move_pct": outcome[
                    "maximum_capturable_move_pct"
                ],
                "day_high_timestamp": _day_high_timestamp(outcome),
                "universe_eligible": universe_eligible,
                "scout_selected": bool(outcome["selected"]),
                "scout_selection_basis": _selection_basis(outcome),
                "scout_trade_executed": bool(
                    scout_execution.get("trade_executed", False)
                ),
                "scout_realized_return_pct": scout_execution.get(
                    "realized_return_pct"
                ),
                "scout_exit_reason": scout_execution.get("exit_reason"),
                "scout_exit_timestamp": scout_execution.get("exit_timestamp"),
                "fair_simulation_possible": execution is not None,
                "simulation_exclusion_reason": exclusion,
                "trade_executed": (
                    execution["trade_executed"] if execution else None
                ),
                "realized_return_pct": (
                    execution["realized_return_pct"] if execution else None
                ),
                "realized_pnl_usd": (
                    execution["realized_pnl_usd"] if execution else None
                ),
                "capture_ratio": (
                    execution["capture_ratio"] if execution else None
                ),
                "exit_reason": execution["exit_reason"] if execution else None,
            }
        )

    qualifying_outcomes = [
        item
        for item in outcome_result["outcomes"]
        if _selection_basis(item) == QUALIFYING_THRESHOLD
    ]
    exploration_outcomes = [
        item
        for item in outcome_result["outcomes"]
        if _selection_basis(item) == EXPLORATION_TOP_K
    ]
    qualifying_executions = [
        item["execution_result"]
        for item in qualifying_outcomes
    ]
    exploration_executions = [
        item["execution_result"]
        for item in exploration_outcomes
    ]
    qualifying_count = len(qualifying_outcomes)
    exploration_count = len(exploration_outcomes)
    scout_summary = _summary(
        qualifying_executions, qualifying_count, strategy_capital
    )
    exploration_summary = _summary(
        exploration_executions, exploration_count, strategy_capital
    )
    combined_summary = _summary(
        qualifying_executions + exploration_executions,
        qualifying_count + exploration_count,
        strategy_capital,
    )
    # This summary is retained only for the top-10 diagnostic table. It no
    # longer supplies WIN/TIE/MISS.
    benchmark_summary = _summary(
        benchmark_executions, len(benchmark_candidates), strategy_capital
    )
    baselines = _return_baselines(
        eligible_outcomes, qualifying_count, strategy_capital
    )

    qualifying_selected = {
        item["ticker"]
        for item in outcome_result["outcomes"]
        if _selection_basis(item) == QUALIFYING_THRESHOLD
    }
    exploration_selected = {
        item["ticker"]
        for item in outcome_result["outcomes"]
        if _selection_basis(item) == EXPLORATION_TOP_K
    }
    combined_selected = {
        item["ticker"]
        for item in outcome_result["outcomes"]
        if item["selected"]
    }
    benchmark_tickers = {item["ticker"] for item in benchmark_candidates}
    qualifying_capture_rate = (
        len(qualifying_selected & benchmark_tickers)
        / len(benchmark_tickers)
        * 100
        if benchmark_tickers
        else 0.0
    )
    exploration_capture_rate = (
        len(exploration_selected & benchmark_tickers)
        / len(benchmark_tickers)
        * 100
        if benchmark_tickers
        else 0.0
    )
    combined_capture_rate = (
        len(combined_selected & benchmark_tickers)
        / len(benchmark_tickers)
        * 100
        if benchmark_tickers
        else 0.0
    )
    random_return = baselines["random_draw_mean_realized_return_pct"]
    random_pnl = baselines["random_draw_mean_realized_pnl_usd"]
    return_difference = scout_summary["realized_return_pct"] - random_return
    pnl_difference = scout_summary["realized_pnl_usd"] - random_pnl
    eligible_difference = (
        scout_summary["realized_return_pct"]
        - baselines["eligible_basket_expected_return_pct"]
    )
    tolerance = 1e-9
    if return_difference > tolerance:
        result_code = "SCOUT_OUTPERFORMED"
    elif return_difference < -tolerance:
        result_code = "SCOUT_UNDERPERFORMED"
    else:
        result_code = "SCOUT_TIED"

    result = {
        "replay_id": snapshot["replay_id"],
        "trading_date": snapshot["trading_date"],
        "universe_version": snapshot["universe_version"],
        **universe_metadata,
        "execution_policy_version": outcome_result["execution_policy_version"],
        "benchmark_method": "RANDOM_DRAW_BASELINE_SAME_UNIVERSE",
        "benchmark_candidates": benchmark_candidates,
        "scout_summary": scout_summary,
        "exploration_summary": exploration_summary,
        "combined_summary": combined_summary,
        "benchmark_summary": benchmark_summary,
        "return_baselines": baselines,
        "comparison": {
            "scout_won": result_code == "SCOUT_OUTPERFORMED",
            "scout_net_realized_return_pct": scout_summary[
                "realized_return_pct"
            ],
            "eligible_ticker_mean_return_difference_pct": eligible_difference,
            "random_draw_mean_return_difference_pct": return_difference,
            "return_difference_pct": return_difference,
            "pnl_difference_usd": pnl_difference,
            "top_10_capture_rate_pct": qualifying_capture_rate,
            "exploration_top_10_capture_rate_pct": exploration_capture_rate,
            "combined_top_10_capture_rate_pct": combined_capture_rate,
            "result_code": result_code,
        },
    }
    try:
        validate_contract("benchmark_result", result)
    except ContractError as exc:
        raise BenchmarkError(
            f"Benchmark contract validation failed: {exc}"
        ) from exc
    return result
