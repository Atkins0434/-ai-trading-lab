from __future__ import annotations

from statistics import mean
from typing import Any

from trainer.trade_engine import load_execution_policy, simulate_trade
from trainer.validate_contracts import ContractError, validate_contract
from trainer.evidence_eligibility import EvidenceEligibilityError, assert_matching_universe


class BenchmarkError(Exception):
    """Raised when a same-universe benchmark cannot be built."""


def _summary(executions: list[dict[str, Any]], candidate_count: int, strategy_capital: float) -> dict[str, Any]:
    completed = [item for item in executions if item.get("trade_executed")]
    returns = [float(item["realized_return_pct"]) for item in completed]
    pnls = [float(item["realized_pnl_usd"]) for item in completed]
    gains = sum(value for value in pnls if value > 0)
    losses = abs(sum(value for value in pnls if value < 0))
    ratios = [float(item["capture_ratio"]) for item in completed if item.get("capture_ratio") is not None]
    drawdowns = [float(item["maximum_position_drawdown_pct"]) for item in completed if item.get("maximum_position_drawdown_pct") is not None]
    return {
        "candidate_count": candidate_count,
        "trades_executed": len(completed),
        "realized_return_pct": sum(pnls) / strategy_capital * 100,
        "realized_pnl_usd": sum(pnls),
        "win_rate_pct": (sum(value > 0 for value in returns) / len(returns) * 100) if returns else None,
        "profit_factor": gains / losses if losses else None,
        "max_drawdown_pct": min(drawdowns) if drawdowns else None,
        "average_capture_ratio": mean(ratios) if ratios else None,
    }


def build_same_universe_benchmark(
    snapshot: dict[str, Any],
    scout_result: dict[str, Any],
    outcome_result: dict[str, Any],
    strategy_capital: float,
) -> dict[str, Any]:
    """Compare Scout with the top MFE movers from its exact screened universe."""
    try:
        universe_metadata = assert_matching_universe(
            snapshot, scout_result, outcome_result
        )
    except EvidenceEligibilityError as exc:
        raise BenchmarkError(str(exc)) from exc
    policy = load_execution_policy()
    outcomes = sorted(
        outcome_result["outcomes"],
        key=lambda item: (-item["mfe_pct"], item["ticker"]),
    )[:10]
    benchmark_candidates = []
    benchmark_executions = []
    scout_candidates = {item["ticker"]: item for item in scout_result["candidates"]}
    simulated_positions = 0
    for rank, outcome in enumerate(outcomes, start=1):
        execution = None
        exclusion = None
        scout_candidate = scout_candidates.get(outcome["ticker"])
        research_eligible = bool(
            scout_candidate
            and scout_candidate.get(
                "research_eligible", scout_candidate.get("eligible", False)
            )
        )
        if not research_eligible:
            exclusion = "PRE_ENTRY_GUARDRAIL_REJECTED"
        elif simulated_positions < policy.max_positions:
            bars = outcome["intraday_path"]
            raw = simulate_trade(
                outcome["ticker"], float(bars[0]["open"]), bars, strategy_capital
            )
            maximum_move = float(outcome["maximum_capturable_move_pct"])
            capture_ratio = raw["realized_return_pct"] / maximum_move if maximum_move > 0 else None
            execution = {
                **raw,
                "capture_ratio": capture_ratio,
                "maximum_position_drawdown_pct": outcome["mae_pct"],
            }
            benchmark_executions.append(execution)
            simulated_positions += 1
        else:
            exclusion = "MAX_POSITIONS_REACHED"
        benchmark_candidates.append({
            "ticker": outcome["ticker"],
            "benchmark_rank": rank,
            "raw_move_pct": outcome["mfe_pct"],
            "fair_simulation_possible": execution is not None,
            "simulation_exclusion_reason": exclusion,
            "trade_executed": execution["trade_executed"] if execution else None,
            "realized_return_pct": execution["realized_return_pct"] if execution else None,
            "realized_pnl_usd": execution["realized_pnl_usd"] if execution else None,
            "capture_ratio": execution["capture_ratio"] if execution else None,
            "exit_reason": execution["exit_reason"] if execution else None,
        })

    scout_executions = [item["execution_result"] for item in outcome_result["outcomes"] if item["selected"]]
    scout_summary = _summary(scout_executions, sum(item["selected"] for item in outcome_result["outcomes"]), strategy_capital)
    benchmark_summary = _summary(benchmark_executions, len(benchmark_candidates), strategy_capital)
    selected = {item["ticker"] for item in outcome_result["outcomes"] if item["selected"]}
    benchmark_tickers = {item["ticker"] for item in benchmark_candidates}
    capture_rate = len(selected & benchmark_tickers) / len(benchmark_tickers) * 100 if benchmark_tickers else 0.0
    return_difference = scout_summary["realized_return_pct"] - benchmark_summary["realized_return_pct"]
    pnl_difference = scout_summary["realized_pnl_usd"] - benchmark_summary["realized_pnl_usd"]
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
        "benchmark_method": "TOP_10_MOVERS_SAME_UNIVERSE",
        "benchmark_candidates": benchmark_candidates,
        "scout_summary": scout_summary,
        "benchmark_summary": benchmark_summary,
        "comparison": {
            "scout_won": result_code == "SCOUT_OUTPERFORMED",
            "return_difference_pct": return_difference,
            "pnl_difference_usd": pnl_difference,
            "top_10_capture_rate_pct": capture_rate,
            "target_driver_capture_pct": 70,
            "target_driver_capture_met": capture_rate >= 70,
            "result_code": result_code,
        },
    }
    try:
        validate_contract("benchmark_result", result)
    except ContractError as exc:
        raise BenchmarkError(f"Benchmark contract validation failed: {exc}") from exc
    return result
