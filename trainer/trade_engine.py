from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainer.validate_contracts import ContractError, load_json


ROOT = Path(__file__).resolve().parent.parent
EXECUTION_CONFIG_PATH = ROOT / "config" / "execution_policy.json"


class ExecutionError(Exception):
    """Raised when deterministic execution cannot proceed."""


@dataclass(frozen=True)
class ExecutionPolicy:
    max_position_pct: float
    max_positions: int
    trailing_stop_pct: float
    profit_take_pct: float
    abnormal_put_volume_multiple: float
    mandatory_end_of_session_liquidation: bool


def load_execution_policy() -> ExecutionPolicy:
    """Load the versioned execution policy from config."""
    try:
        config = load_json(EXECUTION_CONFIG_PATH)
    except ContractError as exc:
        raise ExecutionError(
            f"Unable to load execution policy: {exc}"
        ) from exc

    return ExecutionPolicy(
        max_position_pct=config["capital"]["max_position_pct"],
        max_positions=config["capital"]["max_positions"],
        trailing_stop_pct=config[
            "position_management"
        ]["trailing_stop_pct"],
        profit_take_pct=config[
            "position_management"
        ]["profit_take_pct"],
        abnormal_put_volume_multiple=config[
            "options_risk"
        ]["abnormal_put_volume_multiple"],
        mandatory_end_of_session_liquidation=config[
            "session"
        ]["mandatory_end_of_session_liquidation"],
    )


def calculate_position_value(
    strategy_capital: float,
    policy: ExecutionPolicy,
) -> float:
    """Maximum dollar allocation for one position."""
    if strategy_capital <= 0:
        raise ExecutionError(
            "Strategy capital must be greater than zero."
        )

    return strategy_capital * policy.max_position_pct


def calculate_share_quantity(
    position_value_usd: float,
    entry_price: float,
) -> int:
    """Return whole-share position size."""
    if position_value_usd <= 0:
        raise ExecutionError(
            "Position value must be greater than zero."
        )

    if entry_price <= 0:
        raise ExecutionError(
            "Entry price must be greater than zero."
        )

    return int(position_value_usd // entry_price)


def trailing_stop_price(
    highest_price_since_entry: float,
    trailing_stop_pct: float,
) -> float:
    """Calculate current trailing-stop trigger."""
    if highest_price_since_entry <= 0:
        raise ExecutionError(
            "Highest price must be greater than zero."
        )

    return highest_price_since_entry * (
        1 - trailing_stop_pct / 100
    )


def profit_target_price(
    entry_price: float,
    profit_take_pct: float,
) -> float:
    """Calculate full-position profit target."""
    if entry_price <= 0:
        raise ExecutionError(
            "Entry price must be greater than zero."
        )

    return entry_price * (
        1 + profit_take_pct / 100
    )


def simulate_trade(
    ticker: str,
    entry_price: float,
    intraday_bars: list[dict[str, Any]],
    strategy_capital: float,
    abnormal_put_events: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Simulate one deterministic position.

    Exit priority:
    1. abnormal put-volume liquidation
    2. trailing stop
    3. profit target
    4. session-end liquidation

    Intraday bars must already be ordered chronologically.
    """
    if not intraday_bars:
        raise ExecutionError(
            f"No intraday bars supplied for {ticker}."
        )

    policy = load_execution_policy()

    position_value = calculate_position_value(
        strategy_capital,
        policy,
    )

    shares = calculate_share_quantity(
        position_value,
        entry_price,
    )

    if shares <= 0:
        return {
            "ticker": ticker,
            "trade_executed": False,
            "entry_price": entry_price,
            "position_size_shares": 0,
            "position_value_usd": 0.0,
            "exit_timestamp": None,
            "exit_price": None,
            "exit_reason": "ENTRY_REJECTED",
            "realized_pnl_usd": 0.0,
            "realized_return_pct": 0.0,
            "highest_price_since_entry": entry_price,
        }

    actual_position_value = shares * entry_price

    target_price = profit_target_price(
        entry_price,
        policy.profit_take_pct,
    )

    highest_price = entry_price
    exit_price = None
    exit_timestamp = None
    exit_reason = None

    abnormal_put_events = abnormal_put_events or {}

    for bar in intraday_bars:
        timestamp = bar["timestamp"]

        put_multiple = abnormal_put_events.get(timestamp)

        if (
            put_multiple is not None
            and put_multiple
            >= policy.abnormal_put_volume_multiple
        ):
            exit_price = bar["open"]
            exit_timestamp = timestamp
            exit_reason = "ABNORMAL_PUT_VOLUME"
            break

        if bar["high"] > highest_price:
            highest_price = bar["high"]

        stop_price = trailing_stop_price(
            highest_price,
            policy.trailing_stop_pct,
        )

        # Conservative same-bar assumption:
        # stop gets priority over profit target if both could
        # have been touched inside the same OHLC bar.
        if bar["low"] <= stop_price:
            exit_price = stop_price
            exit_timestamp = timestamp
            exit_reason = "TRAILING_STOP"
            break

        if bar["high"] >= target_price:
            exit_price = target_price
            exit_timestamp = timestamp
            exit_reason = "PROFIT_TARGET"
            break

    if exit_price is None:
        if not policy.mandatory_end_of_session_liquidation:
            raise ExecutionError(
                "Trade remained open but session liquidation "
                "is disabled."
            )

        final_bar = intraday_bars[-1]
        exit_price = final_bar["close"]
        exit_timestamp = final_bar["timestamp"]
        exit_reason = "SESSION_END"

    realized_pnl = (
        exit_price - entry_price
    ) * shares

    realized_return_pct = (
        (exit_price - entry_price)
        / entry_price
    ) * 100

    return {
        "ticker": ticker,
        "trade_executed": True,
        "entry_price": entry_price,
        "position_size_shares": shares,
        "position_value_usd": actual_position_value,
        "exit_timestamp": exit_timestamp,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "realized_pnl_usd": realized_pnl,
        "realized_return_pct": realized_return_pct,
        "highest_price_since_entry": highest_price,
    }
