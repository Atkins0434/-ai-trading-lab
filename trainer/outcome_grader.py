from __future__ import annotations

from datetime import datetime
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

    mfe_pct = (max(bar["high"] for bar in bars) / reference_price - 1) * 100
    mae_pct = (min(bar["low"] for bar in bars) / reference_price - 1) * 100

    running_low = float(bars[0]["low"])
    maximum_move_pct = 0.0
    for bar in bars:
        running_low = min(running_low, float(bar["low"]))
        move_pct = (float(bar["high"]) / running_low - 1) * 100
        maximum_move_pct = max(maximum_move_pct, move_pct)

    return {
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "maximum_capturable_move_pct": maximum_move_pct,
    }


def _execution_contract(
    execution: dict[str, Any],
    entry_timestamp: str,
    maximum_move_pct: float,
    mae_pct: float,
) -> dict[str, Any]:
    realized_return = float(execution["realized_return_pct"])
    capture_ratio = (
        realized_return / maximum_move_pct
        if maximum_move_pct > 0
        else None
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
        "maximum_position_drawdown_pct": mae_pct,
    }


def _no_trade_contract(reason: str | None = None) -> dict[str, Any]:
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
        "maximum_position_drawdown_pct": None,
    }


def grade_replay_outcomes(
    snapshot: dict[str, Any],
    scout_result: dict[str, Any],
    bars_by_ticker: dict[str, list[dict[str, Any]]],
    strategy_capital: float,
) -> dict[str, Any]:
    """Grade every Scout candidate and execute only selected candidates."""
    policy = load_execution_policy()
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
    for candidate in scout_result["candidates"]:
        ticker = candidate["ticker"]
        bars = bars_by_ticker.get(ticker) or []
        selected = is_selected(candidate)
        basis = selection_basis(candidate)
        if not bars:
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
                    "maximum_capturable_move_pct": None,
                    "execution_result": _no_trade_contract(
                        "NO_REGULAR_SESSION_PATH"
                    ),
                }
            )
            continue
        validate_outcome_bars(bars, ticker=ticker)
        entry_price = float(bars[0]["open"])
        stats = path_statistics(bars, entry_price, ticker=ticker)

        if selected and ticker in executable_tickers:
            try:
                raw_execution = simulate_trade(
                    ticker=ticker,
                    entry_price=entry_price,
                    intraday_bars=bars,
                    strategy_capital=strategy_capital,
                )
            except ExecutionError as exc:
                raise OutcomeError(
                    f"ticker={ticker}: execution failed: {exc}"
                ) from exc
            execution = _execution_contract(
                raw_execution,
                bars[0]["timestamp"],
                stats["maximum_capturable_move_pct"],
                stats["mae_pct"],
            )
        elif selected:
            execution = _no_trade_contract("ENTRY_REJECTED")
        else:
            execution = _no_trade_contract()

        outcomes.append(
            {
                "ticker": ticker,
                "selected": selected,
                "selection_basis": basis,
                "scout_rank": candidate["rank"],
                "scout_score_pct": candidate["score_pct"],
                "intraday_path": bars,
                **stats,
                "execution_result": execution,
            }
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
    }
    try:
        validate_contract("end_of_day_outcome", result)
    except ContractError as exc:
        raise OutcomeError(
            f"ticker=ALL: outcome contract validation failed: {exc}"
        ) from exc
    return result
