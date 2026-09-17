from __future__ import annotations

from datetime import datetime
from collections import Counter
from pathlib import Path
from statistics import mean
import sys
import time
from typing import Any

from trainer.trade_engine import (
    ExecutionError,
    load_execution_policy,
    simulate_trade,
)
from trainer.rate_control import load_massive_plan
from trainer.validate_contracts import ContractError, validate_contract


class OutcomeError(Exception):
    """Raised when an end-of-day path cannot be graded deterministically."""


QUALIFYING_THRESHOLD = "QUALIFYING_THRESHOLD"
EXPLORATION_TOP_K = "EXPLORATION_TOP_K"
NOT_SELECTED = "NOT_SELECTED"


def _log_grading_progress(
    policy_id: str,
    completed: int,
    total: int,
    started: float,
    *,
    log_every: int = 250,
) -> None:
    elapsed = time.perf_counter() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else 0.0
    message = (
        f"phase=grading policy_id={policy_id} tickers={completed}/{total} "
        f"elapsed_seconds={elapsed:.1f} eta_seconds={eta:.1f}"
    )
    if sys.stderr.isatty():
        print(f"\r[outcome_grader] {message}", end="", file=sys.stderr, flush=True)
        if completed == total:
            print(file=sys.stderr, flush=True)
    if completed == total or completed % log_every == 0:
        print(f"[outcome_grader] {message}", file=sys.stderr, flush=True)


def _parse_timestamp(raw: str, ticker: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise OutcomeError(
            f"ticker={ticker}: invalid outcome timestamp: {raw}"
        ) from exc
    if parsed.tzinfo is None:
        raise OutcomeError(
            f"ticker={ticker}: outcome timestamps require a timezone."
        )
    return parsed


def validate_outcome_bars(
    bars: list[dict[str, Any]],
    *,
    ticker: str = "UNKNOWN",
) -> None:
    if not bars:
        raise OutcomeError(
            f"ticker={ticker}: at least one regular-session bar is required."
        )
    previous: datetime | None = None
    previous_raw: str | None = None
    for index, bar in enumerate(bars):
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(bar)
        if missing:
            raise OutcomeError(
                f"ticker={ticker}: outcome bar {index} is missing "
                f"{sorted(missing)}."
            )
        timestamp = _parse_timestamp(bar["timestamp"], ticker)
        if previous is not None and timestamp <= previous:
            raise OutcomeError(
                f"ticker={ticker}: outcome bars must be strictly chronological; "
                f"previous_timestamp={previous_raw}, "
                f"current_timestamp={bar['timestamp']}."
            )
        previous = timestamp
        previous_raw = bar["timestamp"]
        prices = [bar["open"], bar["high"], bar["low"], bar["close"]]
        if any(not isinstance(value, (int, float)) or value <= 0 for value in prices):
            raise OutcomeError(
                f"ticker={ticker}: outcome bar {index} has an invalid price."
            )
        if bar["high"] < max(prices) or bar["low"] > min(prices):
            raise OutcomeError(
                f"ticker={ticker}: outcome bar {index} has inconsistent OHLC."
            )
        if not isinstance(bar["volume"], (int, float)) or bar["volume"] < 0:
            raise OutcomeError(
                f"ticker={ticker}: outcome bar {index} has invalid volume."
            )


def path_statistics(
    bars: list[dict[str, Any]],
    reference_price: float,
    *,
    ticker: str = "UNKNOWN",
) -> dict[str, float]:
    """Calculate MFE, MAE, and the best chronological low-to-high move."""
    validate_outcome_bars(bars, ticker=ticker)
    if reference_price <= 0:
        raise OutcomeError(
            f"ticker={ticker}: reference price must be positive."
        )

    day_mfe_pct = (max(bar["high"] for bar in bars) / reference_price - 1) * 100
    day_mae_pct = (min(bar["low"] for bar in bars) / reference_price - 1) * 100

    running_low = float(bars[0]["low"])
    maximum_move_pct = 0.0
    for bar in bars:
        running_low = min(running_low, float(bar["low"]))
        move_pct = (float(bar["high"]) / running_low - 1) * 100
        maximum_move_pct = max(maximum_move_pct, move_pct)

    return {
        "day_mfe_pct": day_mfe_pct,
        "day_mae_pct": day_mae_pct,
        "maximum_capturable_move_pct": maximum_move_pct,
    }


def _held_statistics(
    bars: list[dict[str, Any]],
    reference_price: float,
    exit_timestamp: str | None,
) -> dict[str, float | None]:
    if exit_timestamp is None:
        return {"mfe_pct": None, "mae_pct": None}
    exit_at = datetime.fromisoformat(exit_timestamp.replace("Z", "+00:00"))
    held = [
        bar
        for bar in bars
        if datetime.fromisoformat(bar["timestamp"].replace("Z", "+00:00"))
        <= exit_at
    ]
    if not held:
        return {"mfe_pct": None, "mae_pct": None}
    return {
        "mfe_pct": (
            max(float(bar["high"]) for bar in held) / reference_price - 1
        ) * 100,
        "mae_pct": (
            min(float(bar["low"]) for bar in held) / reference_price - 1
        ) * 100,
    }


def _execution_contract(
    execution: dict[str, Any],
    entry_timestamp: str,
    maximum_move_pct: float,
    bars: list[dict[str, Any]],
) -> dict[str, Any]:
    realized_return = float(execution["realized_return_pct"])
    capture_ratio = (
        realized_return / maximum_move_pct
        if maximum_move_pct > 0
        else None
    )
    held = _held_statistics(
        bars,
        float(execution["entry_price"]),
        execution.get("exit_timestamp"),
    )
    return {
        "trade_executed": execution["trade_executed"],
        "entry_timestamp": (
            entry_timestamp if execution["trade_executed"] else None
        ),
        "entry_price": execution.get("entry_price"),
        "position_size_shares": execution.get("position_size_shares"),
        "position_value_usd": execution.get("position_value_usd"),
        "exit_timestamp": execution.get("exit_timestamp"),
        "exit_price": execution.get("exit_price"),
        "exit_reason": execution.get("exit_reason"),
        "realized_pnl_usd": execution.get("realized_pnl_usd"),
        "realized_return_pct": realized_return,
        "capture_ratio": capture_ratio,
        "mfe_pct": held["mfe_pct"],
        "mae_pct": held["mae_pct"],
        "maximum_position_drawdown_pct": held["mae_pct"],
        "policy_id": execution["policy_id"],
        "exit_mode": execution["exit_mode"],
        "sizing_mode": execution["sizing_mode"],
        "stop_distance_usd": execution.get("stop_distance_usd"),
        "stop_distance_pct": execution.get("stop_distance_pct"),
        "target_distance_usd": execution.get("target_distance_usd"),
        "target_distance_pct": execution.get("target_distance_pct"),
        "risk_usd_at_entry": execution.get("risk_usd_at_entry"),
        "entry_rejection_reason": execution.get("entry_rejection_reason"),
    }


def _no_trade_contract(
    policy: Any,
    reason: str | None = None,
    *,
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "trade_executed": False,
        "entry_timestamp": None,
        "entry_price": None,
        "position_size_shares": 0,
        "position_value_usd": 0.0,
        "exit_timestamp": None,
        "exit_price": None,
        "exit_reason": reason,
        "realized_pnl_usd": 0.0,
        "realized_return_pct": 0.0,
        "capture_ratio": None,
        "mfe_pct": None,
        "mae_pct": None,
        "maximum_position_drawdown_pct": None,
        "policy_id": policy.policy_id,
        "exit_mode": policy.exit_mode,
        "sizing_mode": policy.sizing_mode,
        "stop_distance_usd": None,
        "stop_distance_pct": None,
        "target_distance_usd": None,
        "target_distance_pct": None,
        "risk_usd_at_entry": None,
        "entry_rejection_reason": rejection_reason,
    }


def _atr_by_ticker(snapshot: dict[str, Any]) -> dict[str, float | None]:
    return {
        security["ticker"]: security.get("market_data", {})
        .get("atr_14_usd", {})
        .get("value")
        for security in snapshot.get("securities", [])
    }


def _simulate_contract(
    *,
    ticker: str,
    bars: list[dict[str, Any]],
    policy: Any,
    atr_14_usd: float | None,
    strategy_capital: float,
    maximum_move_pct: float,
) -> dict[str, Any]:
    try:
        raw_execution = simulate_trade(
            ticker=ticker,
            entry_price=float(bars[0]["open"]),
            intraday_bars=bars,
            strategy_capital=strategy_capital,
            policy=policy,
            atr_14_usd=atr_14_usd,
        )
    except ExecutionError as exc:
        raise OutcomeError(
            f"ticker={ticker}: execution failed: {exc}"
        ) from exc
    return _execution_contract(
        raw_execution,
        bars[0]["timestamp"],
        maximum_move_pct,
        bars,
    )


def _execution_summary(
    executions: list[dict[str, Any]],
    strategy_capital: float,
) -> dict[str, Any]:
    completed = [item for item in executions if item["trade_executed"]]
    returns = [float(item["realized_return_pct"]) for item in completed]
    winners = [value for value in returns if value > 0]
    losers = [value for value in returns if value < 0]
    captures = [
        float(item["capture_ratio"])
        for item in completed
        if item.get("capture_ratio") is not None
    ]
    drawdowns = [
        float(item["maximum_position_drawdown_pct"])
        for item in completed
        if item.get("maximum_position_drawdown_pct") is not None
    ]
    reason_counts = Counter(
        item.get("exit_reason")
        for item in executions
        if item.get("exit_reason") in {
            "TRAILING_STOP", "PROFIT_TARGET", "SESSION_END"
        }
    )
    rejected = Counter(
        item.get("entry_rejection_reason") or "UNSPECIFIED"
        for item in executions
        if item.get("exit_reason") == "ENTRY_REJECTED"
    )
    net_pnl = sum(float(item["realized_pnl_usd"]) for item in completed)
    return {
        "net_realized_pnl_usd": net_pnl,
        "realized_return_pct": net_pnl / strategy_capital * 100,
        "win_rate_pct": (
            sum(value > 0 for value in returns) / len(returns) * 100
            if returns
            else None
        ),
        "average_winner_pct": mean(winners) if winners else None,
        "average_loser_pct": mean(losers) if losers else None,
        "average_capture_ratio": mean(captures) if captures else None,
        "max_drawdown_pct": min(drawdowns) if drawdowns else None,
        "exit_reason_counts": {
            "TRAILING_STOP": reason_counts["TRAILING_STOP"],
            "PROFIT_TARGET": reason_counts["PROFIT_TARGET"],
            "SESSION_END": reason_counts["SESSION_END"],
            "ENTRY_REJECTED": dict(sorted(rejected.items())),
        },
    }


def grade_replay_outcomes(
    snapshot: dict[str, Any],
    scout_result: dict[str, Any],
    bars_by_ticker: dict[str, list[dict[str, Any]]],
    strategy_capital: float,
    comparison_policy_paths: list[Path | str] | None = None,
    excluded_tickers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Grade every Scout candidate and execute only selected candidates."""
    policy = load_execution_policy()
    atr_values = _atr_by_ticker(snapshot)
    execution_exclusions = {
        item["ticker"]: item["reason"] for item in (excluded_tickers or [])
    }
    primary_started = time.perf_counter()
    def is_selected(candidate: dict[str, Any]) -> bool:
        return bool(candidate.get("selected", candidate.get("research_selected", False)))

    def selection_basis(candidate: dict[str, Any]) -> str:
        selected = is_selected(candidate)
        basis = candidate.get(
            "selection_basis",
            QUALIFYING_THRESHOLD if selected else NOT_SELECTED,
        )
        allowed = {QUALIFYING_THRESHOLD, EXPLORATION_TOP_K, NOT_SELECTED}
        if basis not in allowed:
            raise OutcomeError(
                f"ticker={candidate['ticker']}: unknown selection basis: {basis}"
            )
        if selected != (basis != NOT_SELECTED):
            raise OutcomeError(
                f"ticker={candidate['ticker']}: selection basis disagrees "
                "with selected state."
            )
        return basis

    qualifying_in_rank_order = sorted(
        (
            candidate
            for candidate in scout_result["candidates"]
            if selection_basis(candidate) == QUALIFYING_THRESHOLD
        ),
        key=lambda candidate: candidate["rank"] or 999999,
    )
    exploration_in_rank_order = sorted(
        (
            candidate
            for candidate in scout_result["candidates"]
            if selection_basis(candidate) == EXPLORATION_TOP_K
        ),
        key=lambda candidate: candidate["rank"] or 999999,
    )
    selected_in_execution_order = (
        qualifying_in_rank_order + exploration_in_rank_order
    )
    executable_tickers = {
        candidate["ticker"]
        for candidate in selected_in_execution_order[: policy.max_positions]
    }

    outcomes = []
    candidates = scout_result["candidates"]
    primary_progress_started = time.perf_counter()
    for completed, candidate in enumerate(candidates, start=1):
        ticker = candidate["ticker"]
        bars = bars_by_ticker.get(ticker) or []
        selected = is_selected(candidate)
        basis = selection_basis(candidate)
        exclusion_reason = execution_exclusions.get(ticker)
        if not bars or exclusion_reason is not None:
            outcomes.append(
                {
                    "ticker": ticker,
                    "selected": selected,
                    "selection_basis": basis,
                    "scout_rank": candidate["rank"],
                    "scout_score_pct": candidate["score_pct"],
                    "intraday_path": [],
                    "mfe_pct": 0.0,
                    "mae_pct": 0.0,
                    "day_mfe_pct": 0.0,
                    "day_mae_pct": 0.0,
                    "maximum_capturable_move_pct": None,
                    "execution_exclusion_reason": exclusion_reason,
                    "execution_result": _no_trade_contract(
                        policy,
                        exclusion_reason or "NO_REGULAR_SESSION_PATH"
                    ),
                }
            )
            _log_grading_progress(
                policy.policy_id, completed, len(candidates), primary_progress_started
            )
            continue
        validate_outcome_bars(bars, ticker=ticker)
        entry_price = float(bars[0]["open"])
        stats = path_statistics(bars, entry_price, ticker=ticker)

        if selected and ticker in executable_tickers:
            execution = _simulate_contract(
                ticker=ticker,
                bars=bars,
                policy=policy,
                atr_14_usd=atr_values.get(ticker),
                strategy_capital=strategy_capital,
                maximum_move_pct=stats["maximum_capturable_move_pct"],
            )
        elif selected:
            execution = _no_trade_contract(
                policy,
                "ENTRY_REJECTED",
                rejection_reason="MAX_POSITIONS_REACHED",
            )
        else:
            execution = _no_trade_contract(policy)

        held = {
            "mfe_pct": execution.get("mfe_pct"),
            "mae_pct": execution.get("mae_pct"),
        }
        outcomes.append(
            {
                "ticker": ticker,
                "selected": selected,
                "selection_basis": basis,
                "scout_rank": candidate["rank"],
                "scout_score_pct": candidate["score_pct"],
                "intraday_path": bars,
                **stats,
                "mfe_pct": (
                    held["mfe_pct"]
                    if held["mfe_pct"] is not None
                    else stats["day_mfe_pct"]
                ),
                "mae_pct": (
                    held["mae_pct"]
                    if held["mae_pct"] is not None
                    else stats["day_mae_pct"]
                ),
                "execution_exclusion_reason": None,
                "execution_result": execution,
            }
        )
        _log_grading_progress(
            policy.policy_id, completed, len(candidates), primary_progress_started
        )

    print(
        "[outcome_grader] phase=grading "
        f"policy_id={policy.policy_id} ticker_count={len(outcomes)} "
        f"elapsed_seconds={time.perf_counter() - primary_started:.3f}",
        file=sys.stderr,
        flush=True,
    )

    outcome_by_ticker = {item["ticker"]: item for item in outcomes}
    top_outcomes = sorted(
        (
            item for item in outcomes
            if item["intraday_path"] and not item["execution_exclusion_reason"]
        ),
        key=lambda item: (-item["day_mfe_pct"], item["ticker"]),
    )[:10]
    policy_comparisons = []
    seen_policy_ids = {policy.policy_id}
    for policy_path in comparison_policy_paths or []:
        comparison_started = time.perf_counter()
        comparison_policy = load_execution_policy(policy_path)
        if comparison_policy.policy_id in seen_policy_ids:
            raise OutcomeError(
                "ticker=ALL: duplicate execution policy_id: "
                f"{comparison_policy.policy_id}"
            )
        seen_policy_ids.add(comparison_policy.policy_id)
        comparison_executions = []
        selected_results = []
        comparison_executable = {
            item["ticker"]
            for item in selected_in_execution_order[
                : comparison_policy.max_positions
            ]
        }
        for candidate in selected_in_execution_order:
            ticker = candidate["ticker"]
            observed = outcome_by_ticker[ticker]
            bars = observed["intraday_path"]
            if not bars:
                execution = _no_trade_contract(
                    comparison_policy, "NO_REGULAR_SESSION_PATH"
                )
            elif ticker not in comparison_executable:
                execution = _no_trade_contract(
                    comparison_policy,
                    "ENTRY_REJECTED",
                    rejection_reason="MAX_POSITIONS_REACHED",
                )
            else:
                execution = _simulate_contract(
                    ticker=ticker,
                    bars=bars,
                    policy=comparison_policy,
                    atr_14_usd=atr_values.get(ticker),
                    strategy_capital=strategy_capital,
                    maximum_move_pct=observed[
                        "maximum_capturable_move_pct"
                    ],
                )
            selected_results.append(execution)
            comparison_executions.append({
                "ticker": ticker,
                "cohort": "SCOUT_SELECTION",
                "benchmark_rank": None,
                "execution_result": execution,
            })

        for rank, observed in enumerate(top_outcomes, start=1):
            ticker = observed["ticker"]
            if rank > comparison_policy.max_positions:
                execution = _no_trade_contract(
                    comparison_policy,
                    "ENTRY_REJECTED",
                    rejection_reason="MAX_POSITIONS_REACHED",
                )
            else:
                execution = _simulate_contract(
                    ticker=ticker,
                    bars=observed["intraday_path"],
                    policy=comparison_policy,
                    atr_14_usd=atr_values.get(ticker),
                    strategy_capital=strategy_capital,
                    maximum_move_pct=observed[
                        "maximum_capturable_move_pct"
                    ],
                )
            comparison_executions.append({
                "ticker": ticker,
                "cohort": "TOP_10_MOVER",
                "benchmark_rank": rank,
                "execution_result": execution,
            })

        policy_comparisons.append({
            "policy_id": comparison_policy.policy_id,
            "exit_mode": comparison_policy.exit_mode,
            "sizing_mode": comparison_policy.sizing_mode,
            "executions": comparison_executions,
            "summary": _execution_summary(
                selected_results, strategy_capital
            ),
        })
        print(
            "[outcome_grader] phase=grading "
            f"policy_id={comparison_policy.policy_id} "
            f"ticker_count={len({item['ticker'] for item in comparison_executions})} "
            f"execution_records={len(comparison_executions)} "
            f"elapsed_seconds={time.perf_counter() - comparison_started:.3f}",
            file=sys.stderr,
            flush=True,
        )

    result = {
        "replay_id": snapshot["replay_id"],
        "trading_date": snapshot["trading_date"],
        "scout_version": snapshot["scout_version"],
        "execution_policy_version": snapshot["execution_policy_version"],
        "massive_plan": snapshot.get("massive_plan", load_massive_plan()),
        "universe_mode": snapshot["universe_mode"],
        "universe_manifest_hash": snapshot["universe_manifest_hash"],
        "universe_coverage": snapshot["universe_coverage"],
        "research_evidence": snapshot["research_evidence"],
        "promotion_eligible": snapshot["promotion_eligible"],
        "outcomes": outcomes,
        "policy_comparisons": policy_comparisons,
    }
    try:
        validate_contract("end_of_day_outcome", result)
    except ContractError as exc:
        raise OutcomeError(
            f"ticker=ALL: outcome contract validation failed: {exc}"
        ) from exc
    return result
