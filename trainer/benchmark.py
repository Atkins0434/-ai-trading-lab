from __future__ import annotations

import random
from datetime import datetime
from statistics import mean
import sys
import time
from typing import Any

from trainer.evidence_eligibility import EvidenceEligibilityError, assert_matching_universe
from trainer.rate_control import load_massive_plan
from trainer.trade_engine import load_execution_policy, simulate_trade
from trainer.validate_contracts import ContractError, validate_contract
from trainer.execution_costs import load_execution_costs, net_summary
from trainer.validate_contracts import load_json
from pathlib import Path


RANDOM_BASELINE_SEED = 20260916
RANDOM_BASELINE_DRAWS = 200
QUALIFYING_THRESHOLD = "QUALIFYING_THRESHOLD"
EXPLORATION_TOP_K = "EXPLORATION_TOP_K"
REVERSAL_EXPLORATION = "REVERSAL_EXPLORATION"
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
        **net_summary(completed, strategy_capital),
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
    outcome: dict[str, Any], strategy_capital: float, policy=None, atr_14_usd=None,
) -> dict[str, Any]:
    bars = outcome["intraday_path"]
    if not bars or outcome.get("execution_exclusion_reason"):
        return {
            **outcome["execution_result"],
            "exit_reason": (
                outcome.get("execution_exclusion_reason")
                or "NO_REGULAR_SESSION_PATH"
            ),
            "capture_ratio": None,
            "maximum_position_drawdown_pct": None,
        }
    raw = simulate_trade(
        outcome["ticker"], float(bars[0]["open"]), bars, strategy_capital,
        policy=policy, atr_14_usd=atr_14_usd,
    )
    if not raw["trade_executed"]:
        return {**raw, "capture_ratio": None, "maximum_position_drawdown_pct": None}
    maximum_move = float(outcome["maximum_capturable_move_pct"])
    exit_at = datetime.fromisoformat(
        raw["exit_timestamp"].replace("Z", "+00:00")
    )
    held = [
        bar for bar in bars
        if datetime.fromisoformat(
            bar["timestamp"].replace("Z", "+00:00")
        ) <= exit_at
    ]
    held_mae = (
        min(float(bar["low"]) for bar in held) / float(bars[0]["open"])
        - 1
    ) * 100
    return {
        **raw,
        "entry_timestamp": bars[0]["timestamp"],
        "capture_ratio": (
            raw["realized_return_pct"] / maximum_move
            if maximum_move > 0
            else None
        ),
        "maximum_position_drawdown_pct": held_mae,
    }


def _return_baselines(
    eligible_outcomes: list[dict[str, Any]],
    selected_count: int,
    strategy_capital: float,
    policy=None,
    atr_values=None,
) -> dict[str, Any]:
    """Build deterministic same-policy baselines from the eligible universe."""
    policy = policy or load_execution_policy()
    ordered = sorted(eligible_outcomes, key=lambda item: item["ticker"])
    executions = []
    started = time.perf_counter()
    total = len(ordered)
    for completed, item in enumerate(ordered, start=1):
        executions.append(_simulate_outcome(item, strategy_capital, policy, (atr_values or {}).get(item["ticker"])))
        if completed == total or completed % 250 == 0:
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta = (total - completed) / rate if rate > 0 else 0.0
            print(
                "[benchmark] phase=random_baseline_inputs "
                f"tickers={completed}/{total} elapsed_seconds={elapsed:.1f} "
                f"eta_seconds={eta:.1f}",
                file=sys.stderr,
                flush=True,
            )
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
    net_draw_pnls: list[float] = []
    for _ in range(RANDOM_BASELINE_DRAWS):
        if selected_count == 0:
            draw_pnls.append(0.0)
            net_draw_pnls.append(0.0)
            continue
        draw = rng.sample(executions, selected_count)
        # The draw contains N names, but the same execution policy still caps
        # the number of positions that can actually consume capital.
        executed_draw = draw[: policy.max_positions]
        net_draw_pnls.append(sum(float(item.get("net_realized_pnl_usd") or 0.0) for item in executed_draw))
        draw_pnls.append(
            sum(float(item["realized_pnl_usd"]) for item in executed_draw)
        )
    random_mean_pnl = mean(draw_pnls)
    random_mean_return = random_mean_pnl / strategy_capital * 100
    net_eligible = mean([float(item.get("net_realized_pnl_usd") or 0.0) for item in executions]) if executions else 0.0
    net_basket = net_eligible * min(selected_count, policy.max_positions)
    return {
        "cost_model_id": load_execution_costs()["cost_model_id"],
        "net_eligible_ticker_mean_realized_pnl_usd": net_eligible,
        "net_eligible_single_position_mean_return_pct": net_eligible / strategy_capital * 100,
        "net_eligible_basket_expected_pnl_usd": net_basket,
        "net_eligible_basket_expected_return_pct": net_basket / strategy_capital * 100,
        "net_random_draw_mean_realized_pnl_usd": mean(net_draw_pnls),
        "net_random_draw_mean_realized_return_pct": mean(net_draw_pnls) / strategy_capital * 100,
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
    if basis not in {
        QUALIFYING_THRESHOLD,
        EXPLORATION_TOP_K,
        REVERSAL_EXPLORATION,
        NOT_SELECTED,
    }:
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
        (
            item
            for item in outcome_result["outcomes"]
            if item["intraday_path"]
            and not item.get("execution_exclusion_reason")
        ),
        key=lambda item: (
            -item.get("day_mfe_pct", item["mfe_pct"]), item["ticker"]
        ),
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
                **{key: execution.get(key) if execution else None for key in (
                    "gross_realized_pnl_usd", "gross_realized_return_pct",
                    "net_realized_pnl_usd", "net_realized_return_pct", "net_capture_ratio",
                    "entry_slippage_usd", "exit_slippage_usd", "commissions_usd",
                    "regulatory_fees_usd", "total_cost_usd", "cost_bps_of_position",
                )},
                "scout_net_realized_return_pct": scout_execution.get("net_realized_return_pct"),
                "ticker": outcome["ticker"],
                "benchmark_rank": rank,
                "raw_move_pct": outcome.get(
                    "day_mfe_pct", outcome["mfe_pct"]
                ),
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
                "entry_timestamp": (
                    execution.get("entry_timestamp") if execution else None
                ),
                "entry_price": (
                    execution.get("entry_price") if execution else None
                ),
                "exit_timestamp": (
                    execution.get("exit_timestamp") if execution else None
                ),
                "exit_price": (
                    execution.get("exit_price") if execution else None
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
                "maximum_position_drawdown_pct": (
                    execution.get("maximum_position_drawdown_pct")
                    if execution else None
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
        if _selection_basis(item) in {
            EXPLORATION_TOP_K, REVERSAL_EXPLORATION
        }
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
        if _selection_basis(item) in {
            EXPLORATION_TOP_K, REVERSAL_EXPLORATION
        }
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

    gross_result_code = result_code
    net_difference = (scout_summary["net_realized_return_pct"] - baselines["net_random_draw_mean_realized_return_pct"]
                      if scout_summary["net_realized_return_pct"] is not None else None)
    net_result_code = (None if net_difference is None else "SCOUT_OUTPERFORMED" if net_difference > tolerance else
                       "SCOUT_UNDERPERFORMED" if net_difference < -tolerance else "SCOUT_TIED")
    verdict_basis = load_verdict_basis()
    if verdict_basis == "NET":
        if net_result_code is None:
            raise BenchmarkError("NET verdict requires costed outcomes; rerun the day")
        result_code = net_result_code

    result = {
        "cost_model_id": load_execution_costs()["cost_model_id"],
        "replay_id": snapshot["replay_id"],
        "trading_date": snapshot["trading_date"],
        "universe_version": snapshot["universe_version"],
        "massive_plan": snapshot.get("massive_plan", load_massive_plan()),
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
            "verdict_basis": verdict_basis,
            "gross_result_code": gross_result_code,
            "net_result_code": net_result_code,
            "net_scout_won": net_result_code == "SCOUT_OUTPERFORMED" if net_result_code else None,
            "net_return_difference_pct": net_difference,
            "scout_won": result_code == "SCOUT_OUTPERFORMED",
            "scout_net_realized_return_pct": scout_summary[
                "net_realized_return_pct"
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


def load_verdict_basis(path=None):
    config = load_json(Path(path) if path else Path(__file__).resolve().parent.parent / "config" / "benchmark.json")
    basis = config["verdict_basis"]
    if basis not in {"GROSS", "NET"}:
        raise BenchmarkError("benchmark.verdict_basis must be GROSS or NET")
    return basis
