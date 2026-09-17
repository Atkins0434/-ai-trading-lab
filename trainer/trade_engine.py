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
    policy_id: str
    max_position_pct: float
    max_positions: int
    exit_mode: str
    trailing_stop_pct: float | None
    profit_take_pct: float | None
    atr_stop_multiple: float | None
    atr_target_multiple: float | None
    sizing_mode: str
    risk_per_trade_pct: float | None
    abnormal_put_volume_multiple: float
    mandatory_end_of_session_liquidation: bool


def load_execution_policy(path: Path | str | None = None) -> ExecutionPolicy:
    """Load the versioned execution policy from config."""
    policy_path = Path(path) if path is not None else EXECUTION_CONFIG_PATH
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    try:
        config = load_json(policy_path)
    except ContractError as exc:
        raise ExecutionError(
            f"Unable to load execution policy: {exc}"
        ) from exc

    management = config.get("position_management", {})
    exit_mode = management.get("exit_mode")
    sizing_mode = management.get("sizing_mode")
    if exit_mode not in {"PERCENT", "ATR"}:
        raise ExecutionError(
            f"Unsupported exit_mode in {policy_path}: {exit_mode}"
        )
    if sizing_mode not in {"FIXED_FRACTION", "RISK_PER_TRADE"}:
        raise ExecutionError(
            f"Unsupported sizing_mode in {policy_path}: {sizing_mode}"
        )
    if exit_mode == "PERCENT" and any(
        not isinstance(management.get(name), (int, float))
        or float(management[name]) <= 0
        for name in ("trailing_stop_pct", "profit_take_pct")
    ):
        raise ExecutionError(
            f"PERCENT policy {policy_path} requires positive percent exits."
        )
    if exit_mode == "ATR" and any(
        not isinstance(management.get(name), (int, float))
        or float(management[name]) <= 0
        for name in ("atr_stop_multiple", "atr_target_multiple")
    ):
        raise ExecutionError(
            f"ATR policy {policy_path} requires positive ATR multiples."
        )
    if (
        sizing_mode == "RISK_PER_TRADE"
        and (
            not isinstance(management.get("risk_per_trade_pct"), (int, float))
            or float(management["risk_per_trade_pct"]) <= 0
        )
    ):
        raise ExecutionError(
            f"RISK_PER_TRADE policy {policy_path} requires positive risk."
        )

    return ExecutionPolicy(
        policy_id=str(config["policy_id"]),
        max_position_pct=config["capital"]["max_position_pct"],
        max_positions=config["capital"]["max_positions"],
        exit_mode=exit_mode,
        trailing_stop_pct=management.get("trailing_stop_pct"),
        profit_take_pct=management.get("profit_take_pct"),
        atr_stop_multiple=management.get("atr_stop_multiple"),
        atr_target_multiple=management.get("atr_target_multiple"),
        sizing_mode=sizing_mode,
        risk_per_trade_pct=management.get("risk_per_trade_pct"),
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
    *,
    policy: ExecutionPolicy | None = None,
    atr_14_usd: float | None = None,
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

    policy = policy or load_execution_policy()

    if policy.exit_mode == "PERCENT":
        if policy.trailing_stop_pct is None or policy.profit_take_pct is None:
            raise ExecutionError(
                f"PERCENT policy {policy.policy_id} is missing percent exits."
            )
        stop_distance = entry_price * policy.trailing_stop_pct / 100
        target_distance = entry_price * policy.profit_take_pct / 100
    else:
        if (
            atr_14_usd is None
            or atr_14_usd <= 0
            or policy.atr_stop_multiple is None
            or policy.atr_target_multiple is None
        ):
            return _rejected_trade(
                ticker,
                entry_price,
                policy,
                "ATR_UNAVAILABLE",
            )
        stop_distance = policy.atr_stop_multiple * atr_14_usd
        target_distance = policy.atr_target_multiple * atr_14_usd

    position_value = calculate_position_value(
        strategy_capital,
        policy,
    )

    if policy.sizing_mode == "FIXED_FRACTION":
        shares = calculate_share_quantity(position_value, entry_price)
    else:
        if policy.risk_per_trade_pct is None or policy.risk_per_trade_pct <= 0:
            raise ExecutionError(
                f"RISK_PER_TRADE policy {policy.policy_id} has invalid risk."
            )
        risk_budget = strategy_capital * policy.risk_per_trade_pct / 100
        risk_sized_shares = int(risk_budget // stop_distance)
        position_cap_shares = calculate_share_quantity(
            position_value, entry_price
        )
        shares = min(risk_sized_shares, position_cap_shares)

    if shares <= 0:
        return _rejected_trade(
            ticker,
            entry_price,
            policy,
            "POSITION_TOO_SMALL",
            stop_distance=stop_distance,
            target_distance=target_distance,
        )

    actual_position_value = shares * entry_price

    target_price = (
        profit_target_price(entry_price, policy.profit_take_pct)
        if policy.exit_mode == "PERCENT"
        else entry_price + target_distance
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

        # The bar's internal high/low order is unknowable from OHLC alone.
        # Test the stop derived from information available before this bar,
        # then admit the current high. This avoids assuming the high happened
        # before the low and tightening the stop with future same-bar data.
        stop_price = (
            trailing_stop_price(highest_price, policy.trailing_stop_pct)
            if policy.exit_mode == "PERCENT"
            else highest_price - stop_distance
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

        if bar["high"] > highest_price:
            highest_price = bar["high"]

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
        **_policy_fields(
            policy,
            entry_price,
            shares,
            stop_distance,
            target_distance,
        ),
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


def _policy_fields(
    policy: ExecutionPolicy,
    entry_price: float | None,
    shares: int,
    stop_distance: float | None,
    target_distance: float | None,
) -> dict[str, Any]:
    return {
        "policy_id": policy.policy_id,
        "exit_mode": policy.exit_mode,
        "sizing_mode": policy.sizing_mode,
        "stop_distance_usd": stop_distance,
        "stop_distance_pct": (
            stop_distance / entry_price * 100
            if stop_distance is not None and entry_price
            else None
        ),
        "target_distance_usd": target_distance,
        "target_distance_pct": (
            target_distance / entry_price * 100
            if target_distance is not None and entry_price
            else None
        ),
        "risk_usd_at_entry": (
            shares * stop_distance if stop_distance is not None else None
        ),
    }


def _rejected_trade(
    ticker: str,
    entry_price: float,
    policy: ExecutionPolicy,
    rejection_reason: str,
    *,
    stop_distance: float | None = None,
    target_distance: float | None = None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        **_policy_fields(
            policy, entry_price, 0, stop_distance, target_distance
        ),
        "trade_executed": False,
        "entry_price": entry_price,
        "position_size_shares": 0,
        "position_value_usd": 0.0,
        "exit_timestamp": None,
        "exit_price": None,
        "exit_reason": "ENTRY_REJECTED",
        "entry_rejection_reason": rejection_reason,
        "realized_pnl_usd": 0.0,
        "realized_return_pct": 0.0,
        "highest_price_since_entry": entry_price,
    }
