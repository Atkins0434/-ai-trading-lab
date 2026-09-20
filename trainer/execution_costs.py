"""Post-fill research costs; never used to choose a fill or size a position."""
from datetime import datetime, time
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from trainer.validate_contracts import load_json

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "execution_costs.json"
COST_FIELDS = (
    "cost_model_id", "gross_realized_pnl_usd", "gross_realized_return_pct",
    "entry_slippage_usd", "exit_slippage_usd", "commissions_usd",
    "regulatory_fees_usd", "total_cost_usd", "cost_bps_of_position",
    "net_realized_pnl_usd", "net_realized_return_pct", "net_capture_ratio",
)


def load_execution_costs(path=None):
    config = load_json(Path(path) if path else CONFIG_PATH)
    if config["slippage"]["mode"] != "TIERED_BY_PRICE_AND_TIME":
        raise ValueError("Unsupported slippage mode")
    values = [config[key] for key in (
        "commission_per_share_usd", "commission_minimum_per_order_usd",
        "regulatory_fees_per_share_sold_usd",
    )]
    for key, tiers in config["slippage"].items():
        if key != "mode":
            values.extend(tiers.values())
    if any(not isinstance(value, (int, float)) or value < 0 for value in values):
        raise ValueError("Execution costs must be nonnegative numbers")
    return config


def price_tier(price):
    for ceiling, tier in ((2, "under_2"), (5, "2_to_5"), (10, "5_to_10"), (25, "10_to_25")):
        if price < ceiling:
            return tier
    return "25_plus"


def cost_fields(
    trade,
    day_mfe_pct=None,
    config=None,
    *,
    entry_phase="entry_at_open_bps",
):
    config = config if config is not None else load_execution_costs()
    components = dict.fromkeys(("entry_slippage_usd", "exit_slippage_usd", "commissions_usd", "regulatory_fees_usd"), 0.0)
    if trade.get("trade_executed"):
        shares = trade["position_size_shares"]
        entry, exit_price = trade["entry_price"], trade["exit_price"]
        exit_at = datetime.fromisoformat(trade["exit_timestamp"].replace("Z", "+00:00"))
        local_time = exit_at.astimezone(ZoneInfo("America/New_York")).time()
        phase = ("session_end_exit_bps" if trade["exit_reason"] == "SESSION_END" else
                 "exit_first_5_minutes_bps" if local_time < time(9, 35) else "exit_after_5_minutes_bps")
        if entry_phase not in config["slippage"]:
            raise ValueError(f"Unknown entry cost phase: {entry_phase}")
        components["entry_slippage_usd"] = entry * shares * config["slippage"][entry_phase][price_tier(entry)] / 10000
        components["exit_slippage_usd"] = exit_price * shares * config["slippage"][phase][price_tier(exit_price)] / 10000
        components["commissions_usd"] = 2 * max(shares * config["commission_per_share_usd"], config["commission_minimum_per_order_usd"])
        components["regulatory_fees_usd"] = shares * config["regulatory_fees_per_share_sold_usd"]
    total = sum(components.values())
    position = trade.get("position_value_usd") or 0.0
    net_pnl = trade["realized_pnl_usd"] - total
    net_return = net_pnl / position * 100 if position else 0.0
    return {
        "cost_model_id": config["cost_model_id"],
        "gross_realized_pnl_usd": trade["realized_pnl_usd"],
        "gross_realized_return_pct": trade["realized_return_pct"],
        **components,
        "total_cost_usd": total,
        "cost_bps_of_position": total / position * 10000 if position else 0.0,
        "net_realized_pnl_usd": net_pnl,
        "net_realized_return_pct": net_return,
        "net_capture_ratio": net_return / day_mfe_pct if day_mfe_pct and day_mfe_pct > 0 and trade.get("trade_executed") else None,
    }


def net_summary(executions, strategy_capital=None):
    """Mirror each caller's gross denominator; legacy missing costs stay null."""
    completed = [item for item in executions if item.get("trade_executed")]
    if any(item.get("net_realized_pnl_usd") is None or item.get("net_realized_return_pct") is None for item in completed):
        return dict.fromkeys(("net_realized_pnl_usd", "net_realized_return_pct", "net_average_capture_ratio"))
    pnl = sum(item["net_realized_pnl_usd"] for item in completed)
    returns = [item["net_realized_return_pct"] for item in completed]
    captures = [item["net_capture_ratio"] for item in completed if item.get("net_capture_ratio") is not None]
    return {
        "net_realized_pnl_usd": pnl,
        "net_realized_return_pct": pnl / strategy_capital * 100 if strategy_capital else mean(returns) if returns else 0.0,
        "net_average_capture_ratio": mean(captures) if captures else None,
    }
