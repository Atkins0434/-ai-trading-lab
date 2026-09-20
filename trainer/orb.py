"""Independent opening-range-breakout research for flat-file replays."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import date, datetime, time
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
import time as wall_time
from typing import Any
from zoneinfo import ZoneInfo

from trainer.execution_costs import cost_fields, load_execution_costs
from trainer.flatfile_snapshot import (
    _average_true_range,
    _bar_discontinuity,
    _deduplicate_regular_bars,
)
from trainer.providers.massive_flatfiles import (
    DAY_AGGS_DATASET,
    MINUTE_AGGS_DATASET,
    MassiveFlatFileStore,
)
from trainer.universe_builder import previous_trading_sessions


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "orb_v1.json"
ET = ZoneInfo("America/New_York")
OPENING_START = time(9, 30)
OPENING_END = time(9, 35)
SESSION_END = time(16)


class OrbError(Exception):
    """Raised when ORB evidence cannot be produced deterministically."""


def _progress(phase: str, completed: int, started: float, total: int | None = None) -> None:
    elapsed = wall_time.perf_counter() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (
        (total - completed) / rate
        if total is not None and rate > 0 else None
    )
    total_label = str(total) if total is not None else "unknown"
    eta_label = f"{eta:.1f}" if eta is not None else "unknown"
    print(
        f"[orb] phase={phase} items={completed}/{total_label} "
        f"elapsed_seconds={elapsed:.1f} eta_seconds={eta_label}",
        file=sys.stderr,
        flush=True,
    )


def load_orb_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrbError(f"Unable to load ORB config {path}: {exc}") from exc
    required = {
        "orb_version", "universe", "opening_volume_lookback_sessions",
        "minimum_baseline_sessions", "minimum_relative_volume", "top_n",
        "entry_cutoff_time", "atr_stop_fraction", "risk_per_trade_pct",
        "paper_leverage_cap", "cash_leverage_cap",
        "opening_volume_cache_root",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise OrbError(f"ORB config must contain exactly {sorted(required)}")
    universe_required = {
        "common_stock_types", "allowed_exchanges", "minimum_open_price_usd",
        "minimum_average_daily_volume", "minimum_atr_14_usd",
    }
    if set(config["universe"]) != universe_required:
        raise OrbError(
            f"ORB universe config must contain exactly {sorted(universe_required)}"
        )
    datetime.strptime(config["entry_cutoff_time"], "%H:%M:%S")
    if config["minimum_baseline_sessions"] > config["opening_volume_lookback_sessions"]:
        raise OrbError("minimum_baseline_sessions exceeds the configured lookback")
    for key in (
        "minimum_relative_volume", "atr_stop_fraction", "risk_per_trade_pct",
        "paper_leverage_cap", "cash_leverage_cap",
    ):
        if not isinstance(config[key], (int, float)) or config[key] <= 0:
            raise OrbError(f"{key} must be positive")
    return config


def orb_config_hash(path: Path = CONFIG_PATH) -> str:
    payload = Path(path).read_bytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def opening_cache_path(cache_root: Path, trading_date: str) -> Path:
    normalized = date.fromisoformat(trading_date).isoformat()
    return Path(cache_root) / f"{normalized}.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _bar_priority(bar: dict[str, Any]) -> tuple[int, float]:
    return int(bar.get("transactions") or 0), float(bar.get("volume") or 0)


def _opening_candle(bars: list[dict[str, Any]]) -> dict[str, Any]:
    if not bars:
        return {
            "opening_volume": 0.0,
            "first_candle_open": None,
            "first_candle_high": None,
            "first_candle_low": None,
            "first_candle_close": None,
        }
    ordered = sorted(bars, key=lambda item: item["timestamp"])
    return {
        "opening_volume": sum(float(item["volume"]) for item in ordered),
        "first_candle_open": float(ordered[0]["open"]),
        "first_candle_high": max(float(item["high"]) for item in ordered),
        "first_candle_low": min(float(item["low"]) for item in ordered),
        "first_candle_close": float(ordered[-1]["close"]),
    }


def write_opening_volume_cache(
    flatfiles: MassiveFlatFileStore,
    trading_date: str,
    *,
    cache_root: Path,
    orb_version: str,
) -> Path:
    """Persist a five-minute opening candle for every ticker in the file."""
    target = date.fromisoformat(trading_date)
    seen: set[str] = set()
    by_ticker_minute: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    started = wall_time.perf_counter()
    row_count = 0
    for row_count, bar in enumerate(
        flatfiles.iter_bars(MINUTE_AGGS_DATASET, trading_date), start=1
    ):
        ticker = str(bar["ticker"]).upper()
        seen.add(ticker)
        observed = datetime.fromisoformat(bar["timestamp"]).astimezone(ET)
        if observed.date() != target or not OPENING_START <= observed.time() < OPENING_END:
            continue
        minute = observed.replace(second=0, microsecond=0).isoformat()
        existing = by_ticker_minute[ticker].get(minute)
        if existing is None or _bar_priority(bar) > _bar_priority(existing):
            by_ticker_minute[ticker][minute] = bar
        if row_count % 250000 == 0:
            _progress("opening_cache", row_count, started)
    _progress("opening_cache", row_count, started)
    entries = {
        ticker: _opening_candle(list(by_ticker_minute[ticker].values()))
        for ticker in sorted(seen)
    }
    payload = {
        "orb_version": orb_version,
        "trading_date": trading_date,
        "window": "09:30:00-09:34:59 America/New_York",
        "source": "MASSIVE_FLATFILE",
        "tickers": entries,
    }
    path = opening_cache_path(cache_root, trading_date)
    _atomic_json(path, payload)
    return path


def opening_volume_baseline(
    ticker: str,
    trading_date: str,
    *,
    cache_root: Path,
    lookback_sessions: int = 14,
    minimum_sessions: int = 7,
) -> tuple[float | None, int]:
    values: list[float] = []
    for prior_date in previous_trading_sessions(trading_date, lookback_sessions):
        path = opening_cache_path(cache_root, prior_date)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            item = payload.get("tickers", {}).get(ticker.upper())
            if item is not None:
                values.append(float(item.get("opening_volume") or 0.0))
        except (OSError, ValueError, TypeError):
            continue
    if len(values) < minimum_sessions:
        return None, len(values)
    return mean(values), len(values)


def _manifest_base_reason(
    security: dict[str, Any], config: dict[str, Any]
) -> str | None:
    universe = config["universe"]
    security_type = str(security.get("security_type") or "").upper()
    if security_type not in set(universe["common_stock_types"]):
        return "ORB_SECURITY_TYPE"
    if security.get("listing_venue") not in set(universe["allowed_exchanges"]):
        return "ORB_LISTING_VENUE"
    reasons = set(security.get("reason_codes", []))
    if "SPAC_SUFFIX_SECURITY" in reasons:
        return "ORB_SPAC_SUFFIX"
    if "SPAC_SECURITY" in reasons:
        return "ORB_SPAC"
    stable_id = str(security.get("stable_security_id") or "").strip()
    if not stable_id:
        return "ORB_STABLE_SECURITY_ID_MISSING"
    return None


def _identity_maps(
    securities: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], set[str]]:
    by_id = {str(item["stable_security_id"]): item for item in securities}
    ids_by_ticker: dict[str, set[str]] = defaultdict(set)
    for stable_id, item in by_id.items():
        aliases = {str(item["ticker"]).upper()}
        for value in item.get("ticker_history") or []:
            if isinstance(value, str):
                aliases.add(value.upper())
            elif isinstance(value, dict) and value.get("ticker"):
                aliases.add(str(value["ticker"]).upper())
        for ticker in aliases:
            ids_by_ticker[ticker].add(stable_id)
    unique = {
        ticker: next(iter(ids))
        for ticker, ids in ids_by_ticker.items()
        if len(ids) == 1
    }
    ambiguous = {ticker for ticker, ids in ids_by_ticker.items() if len(ids) > 1}
    return by_id, unique, ambiguous


def _resolve_identity(
    bar: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    unique_tickers: dict[str, str],
) -> str | None:
    provider_id = str(bar.get("stable_security_id") or "").strip()
    if provider_id:
        return provider_id if provider_id in by_id else None
    return unique_tickers.get(str(bar["ticker"]).upper())


def simulate_orb_signal(
    bars: list[dict[str, Any]],
    *,
    direction: str,
    first_candle_high: float,
    first_candle_low: float,
    atr_14_usd: float,
    atr_stop_fraction: float = 0.10,
    entry_cutoff_time: str = "15:30:00",
) -> dict[str, Any]:
    """Simulate an unsized ORB signal without affecting Scout execution."""
    if direction not in {"LONG", "SHORT"}:
        raise OrbError(f"Unsupported ORB direction: {direction}")
    cutoff = datetime.strptime(entry_cutoff_time, "%H:%M:%S").time()
    ordered = sorted(bars, key=lambda item: item["timestamp"])
    trigger = float(first_candle_high if direction == "LONG" else first_candle_low)
    entry_index: int | None = None
    entry_price: float | None = None
    for index, bar in enumerate(ordered):
        observed = datetime.fromisoformat(bar["timestamp"]).astimezone(ET)
        if observed.time() < OPENING_END:
            continue
        if observed.time() > cutoff:
            break
        crossed = (
            float(bar["high"]) > trigger
            if direction == "LONG"
            else float(bar["low"]) < trigger
        )
        if crossed:
            entry_index = index
            entry_price = (
                max(trigger, float(bar["open"]))
                if direction == "LONG"
                else min(trigger, float(bar["open"]))
            )
            break
    if entry_index is None or entry_price is None:
        return {
            "trade_executed": False,
            "status": "NOT_TRIGGERED",
            "exit_reason": "NOT_TRIGGERED",
            "direction": direction,
            "entry_timestamp": None,
            "entry_price": None,
            "exit_timestamp": None,
            "exit_price": None,
            "stop_price": None,
            "stop_distance_usd": None,
            "r_multiple": None,
            "mfe_pct": None,
            "mae_pct": None,
            "gross_realized_return_pct": 0.0,
        }
    stop_distance = float(atr_14_usd) * float(atr_stop_fraction)
    stop = entry_price - stop_distance if direction == "LONG" else entry_price + stop_distance
    exit_index = len(ordered) - 1
    exit_price = float(ordered[-1]["close"])
    exit_reason = "SESSION_END"
    for index in range(entry_index, len(ordered)):
        bar = ordered[index]
        crossed = (
            float(bar["low"]) <= stop
            if direction == "LONG"
            else float(bar["high"]) >= stop
        )
        if crossed:
            opened_through = (
                float(bar["open"]) < stop
                if direction == "LONG"
                else float(bar["open"]) > stop
            )
            exit_price = float(bar["open"]) if opened_through else stop
            exit_index = index
            exit_reason = "STOP"
            break
    held = ordered[entry_index:exit_index + 1]
    if direction == "LONG":
        pnl_per_share = exit_price - entry_price
        mfe = (max(float(bar["high"]) for bar in held) / entry_price - 1) * 100
        mae = (min(float(bar["low"]) for bar in held) / entry_price - 1) * 100
    else:
        pnl_per_share = entry_price - exit_price
        mfe = (
            entry_price - min(float(bar["low"]) for bar in held)
        ) / entry_price * 100
        mae = (
            entry_price - max(float(bar["high"]) for bar in held)
        ) / entry_price * 100
    return {
        "trade_executed": True,
        "status": "TRIGGERED",
        "exit_reason": exit_reason,
        "direction": direction,
        "entry_timestamp": ordered[entry_index]["timestamp"],
        "entry_price": entry_price,
        "exit_timestamp": ordered[exit_index]["timestamp"],
        "exit_price": exit_price,
        "stop_price": stop,
        "stop_distance_usd": stop_distance,
        "r_multiple": pnl_per_share / stop_distance,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "gross_realized_return_pct": pnl_per_share / entry_price * 100,
    }


def _sized_results(
    rows: list[dict[str, Any]],
    *,
    strategy_capital_usd: float,
    leverage_cap: float,
    allow_shorts: bool,
    exhausted_reason: str,
    costs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    risk_budget = strategy_capital_usd * rows[0]["risk_per_trade_pct"] / 100 if rows else 0
    for row in rows:
        key = row["stable_security_id"]
        signal = row["signal"]
        if not row["selected"]:
            results[key] = {"trade_executed": False, "status": row["selection_status"]}
            continue
        if row["direction"] == "SHORT" and not allow_shorts:
            results[key] = {"trade_executed": False, "status": "SHORT_NOT_ALLOWED"}
            continue
        if not signal["trade_executed"]:
            results[key] = dict(signal)
            continue
        shares = math.floor(risk_budget / float(signal["stop_distance_usd"]))
        if shares <= 0:
            results[key] = {"trade_executed": False, "status": "POSITION_TOO_SMALL"}
            continue
        exposure = shares * float(signal["entry_price"])
        entry_at = datetime.fromisoformat(signal["entry_timestamp"])
        candidate_exit = datetime.fromisoformat(signal["exit_timestamp"])
        intervals = [
            (
                datetime.fromisoformat(item["entry_timestamp"]),
                datetime.fromisoformat(item["exit_timestamp"]),
                item["exposure"],
            )
            for item in accepted
        ] + [(entry_at, candidate_exit, exposure)]
        event_times = sorted({value for start, end, _ in intervals for value in (start, end)})
        maximum_exposure = max(
            sum(value for start, end, value in intervals if start <= moment <= end)
            for moment in event_times
        )
        if maximum_exposure > leverage_cap * strategy_capital_usd:
            results[key] = {"trade_executed": False, "status": exhausted_reason}
            continue
        gross_pnl = (
            (float(signal["exit_price"]) - float(signal["entry_price"])) * shares
            if row["direction"] == "LONG"
            else (float(signal["entry_price"]) - float(signal["exit_price"])) * shares
        )
        trade = {
            **signal,
            "position_size_shares": shares,
            "position_value_usd": exposure,
            "realized_pnl_usd": gross_pnl,
            "realized_return_pct": float(signal["gross_realized_return_pct"]),
        }
        trade.update(cost_fields(
            trade,
            config=costs,
            entry_phase="entry_after_open_bps",
        ))
        accepted.append({
            "entry_timestamp": signal["entry_timestamp"],
            "exit_timestamp": signal["exit_timestamp"],
            "exposure": exposure,
        })
        results[key] = trade
    return results


def _summary(
    executions: list[dict[str, Any]],
    strategy_capital_usd: float,
    *,
    signal_triggered: int | None = None,
) -> dict[str, Any]:
    completed = [item for item in executions if item.get("trade_executed")]
    returns = [float(item["gross_realized_return_pct"]) for item in completed]
    multiples = [float(item["r_multiple"]) for item in completed]
    return {
        "candidates": len(executions),
        "triggered": len(completed) if signal_triggered is None else signal_triggered,
        "trades_executed": len(completed),
        "hit_rate_pct": (
            sum(value > 0 for value in returns) / len(returns) * 100
            if returns else None
        ),
        "mean_r": mean(multiples) if multiples else None,
        "sum_r": sum(multiples),
        "gross_realized_pnl_usd": sum(float(item["realized_pnl_usd"]) for item in completed),
        "gross_realized_return_pct": sum(float(item["realized_pnl_usd"]) for item in completed) / strategy_capital_usd * 100,
        "net_realized_pnl_usd": sum(float(item["net_realized_pnl_usd"]) for item in completed),
        "net_realized_return_pct": sum(float(item["net_realized_pnl_usd"]) for item in completed) / strategy_capital_usd * 100,
        "exit_reason_counts": dict(sorted(Counter(
            item.get("exit_reason") or item.get("status") for item in executions
        ).items())),
    }


def _write_outcomes_csv(path: Path, result: dict[str, Any]) -> None:
    base_fields = [
        "orb_version", "trading_date", "rank", "ticker", "stable_security_id",
        "relative_volume_open", "baseline_sessions_used", "direction", "selected",
        "selection_status", "outside_scout_band", "scout_scorable",
        "scout_selected", "top_10_mover", "opening_volume",
        "opening_volume_baseline", "average_daily_volume_14", "atr_14_usd",
        "first_candle_open", "first_candle_high", "first_candle_low",
        "first_candle_close", "signal_triggered",
    ]
    execution_fields = [
        "status", "trade_executed", "entry_timestamp", "entry_price",
        "exit_timestamp", "exit_price", "exit_reason", "position_size_shares",
        "r_multiple", "mfe_pct", "mae_pct", "gross_realized_return_pct",
        "gross_realized_pnl_usd", "net_realized_return_pct",
        "net_realized_pnl_usd",
    ]
    variants = ("paper_long_only", "paper_long_short", "cash_long_only")
    fields = base_fields + [f"{variant}_{field}" for variant in variants for field in execution_fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in result["ranked_candidates"]:
            row = {key: item.get(key) for key in base_fields}
            for variant in variants:
                execution = item["executions"].get(variant, {})
                for field in execution_fields:
                    row[f"{variant}_{field}"] = execution.get(field)
            writer.writerow(row)


def build_orb_research(
    trading_date: str,
    universe_manifest: dict[str, Any],
    flatfiles: MassiveFlatFileStore,
    scout_result: dict[str, Any],
    benchmark_result: dict[str, Any],
    *,
    strategy_capital_usd: float,
    output_path: Path,
    config: dict[str, Any] | None = None,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    active = config or load_orb_config()
    orb_version = active["orb_version"]
    cache_root = Path(cache_root or active["opening_volume_cache_root"])
    write_opening_volume_cache(
        flatfiles, trading_date, cache_root=cache_root, orb_version=orb_version
    )
    target_cache = json.loads(
        opening_cache_path(cache_root, trading_date).read_text(encoding="utf-8")
    )["tickers"]

    universe_exclusions: list[dict[str, Any]] = []
    base: list[dict[str, Any]] = []
    for security in universe_manifest["securities"]:
        reason = _manifest_base_reason(security, active)
        if reason is not None:
            universe_exclusions.append({"ticker": security["ticker"], "reason": reason})
        else:
            base.append(security)
    by_id, unique_tickers, ambiguous_tickers = _identity_maps(base)
    tickers = {str(item["ticker"]).upper() for item in base}
    prior_dates = previous_trading_sessions(trading_date, 15)
    daily: dict[str, list[dict[str, Any]]] = defaultdict(list)
    regular_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved = Counter()
    history_started = wall_time.perf_counter()
    for completed_date, prior_date in enumerate(prior_dates, start=1):
        for bar in flatfiles.iter_bars(DAY_AGGS_DATASET, prior_date, tickers=tickers):
            stable_id = _resolve_identity(bar, by_id, unique_tickers)
            if stable_id is None:
                unresolved[bar["ticker"]] += 1
            else:
                daily[stable_id].append(bar)
        _progress("daily_history", completed_date, history_started, len(prior_dates))
    target_day = date.fromisoformat(trading_date)
    minute_started = wall_time.perf_counter()
    row_count = 0
    for row_count, bar in enumerate(
        flatfiles.iter_bars(MINUTE_AGGS_DATASET, trading_date, tickers=tickers),
        start=1,
    ):
        stable_id = _resolve_identity(bar, by_id, unique_tickers)
        if stable_id is None:
            unresolved[bar["ticker"]] += 1
            continue
        observed = datetime.fromisoformat(bar["timestamp"]).astimezone(ET)
        if observed.date() == target_day and OPENING_START <= observed.time() < SESSION_END:
            regular_raw[stable_id].append(bar)
        if row_count % 250000 == 0:
            _progress("regular_paths", row_count, minute_started)
    _progress("regular_paths", row_count, minute_started)

    scout_by_ticker = {item["ticker"]: item for item in scout_result.get("candidates", [])}
    scout_selected = {
        item["ticker"] for item in scout_result.get("candidates", [])
        if item.get("research_selected") or item.get("qualification_selected")
    }
    top_10 = {item["ticker"] for item in benchmark_result.get("benchmark_candidates", [])}
    candidates: list[dict[str, Any]] = []
    candidate_started = wall_time.perf_counter()
    ordered_base = sorted(base, key=lambda item: (item["ticker"], item["stable_security_id"]))
    for completed_security, security in enumerate(ordered_base, start=1):
        ticker = str(security["ticker"]).upper()
        stable_id = str(security["stable_security_id"])
        failures: list[str] = []
        if ticker in ambiguous_tickers:
            failures.append("IDENTITY_AMBIGUOUS")
        path, collisions = _deduplicate_regular_bars(
            ticker, regular_raw.get(stable_id, [])
        )
        if _bar_discontinuity(ticker, path, collisions) is not None:
            failures.append("BAR_DISCONTINUITY")
        opening = target_cache.get(ticker, {})
        open_price = opening.get("first_candle_open")
        exact_open = next((bar for bar in path if datetime.fromisoformat(bar["timestamp"]).astimezone(ET).time() == OPENING_START), None)
        if exact_open is None:
            failures.append("OPEN_0930_UNAVAILABLE")
        elif float(exact_open["open"]) < float(active["universe"]["minimum_open_price_usd"]):
            failures.append("OPEN_PRICE_BELOW_MINIMUM")
        prior_rows = sorted(daily.get(stable_id, []), key=lambda item: item["trading_date"])
        last_14 = prior_rows[-14:]
        adv = mean(float(item["volume"]) for item in last_14) if len(last_14) == 14 else None
        if adv is None:
            failures.append("AVERAGE_DAILY_VOLUME_UNAVAILABLE")
        elif adv < float(active["universe"]["minimum_average_daily_volume"]):
            failures.append("AVERAGE_DAILY_VOLUME_BELOW_MINIMUM")
        atr, atr_used = _average_true_range(prior_rows, 14)
        if atr is None:
            failures.append("ATR_14_UNAVAILABLE")
        elif atr < float(active["universe"]["minimum_atr_14_usd"]):
            failures.append("ATR_14_BELOW_MINIMUM")
        if failures:
            universe_exclusions.append({
                "ticker": ticker, "stable_security_id": stable_id,
                "reasons": failures,
            })
            if completed_security % 250 == 0 or completed_security == len(ordered_base):
                _progress("universe", completed_security, candidate_started, len(ordered_base))
            continue
        baseline, baseline_used = opening_volume_baseline(
            ticker,
            trading_date,
            cache_root=cache_root,
            lookback_sessions=int(active["opening_volume_lookback_sessions"]),
            minimum_sessions=int(active["minimum_baseline_sessions"]),
        )
        opening_volume = float(opening.get("opening_volume") or 0.0)
        relative = opening_volume / baseline if baseline and baseline > 0 else None
        candle_open = float(opening["first_candle_open"])
        candle_close = float(opening["first_candle_close"])
        direction = "LONG" if candle_close > candle_open else "SHORT" if candle_close < candle_open else "DOJI"
        scout_candidate = scout_by_ticker.get(ticker)
        candidates.append({
            "orb_version": orb_version,
            "trading_date": trading_date,
            "ticker": ticker,
            "stable_security_id": stable_id,
            "opening_volume": opening_volume,
            "opening_volume_baseline": baseline,
            "baseline_sessions_used": baseline_used,
            "relative_volume_open": relative,
            "direction": direction,
            "outside_scout_band": bool(
                {"MARKET_CAP_OUT_OF_RANGE", "SHARE_PRICE_ABOVE_CAP"}
                & set(security.get("reason_codes", []))
            ),
            "scout_scorable": bool(scout_candidate and scout_candidate.get("status") == "SCORED"),
            "scout_selected": ticker in scout_selected,
            "top_10_mover": ticker in top_10,
            "average_daily_volume_14": adv,
            "atr_14_usd": atr,
            "atr_sessions_used": atr_used,
            "first_candle_open": candle_open,
            "first_candle_high": float(opening["first_candle_high"]),
            "first_candle_low": float(opening["first_candle_low"]),
            "first_candle_close": candle_close,
            "path": path,
            "risk_per_trade_pct": float(active["risk_per_trade_pct"]),
        })
        if completed_security % 250 == 0 or completed_security == len(ordered_base):
            _progress("universe", completed_security, candidate_started, len(ordered_base))

    rankable = [item for item in candidates if item["relative_volume_open"] is not None]
    rankable.sort(key=lambda item: (-item["relative_volume_open"], item["ticker"], item["stable_security_id"]))
    qualified_seen = 0
    ranked: list[dict[str, Any]] = []
    for rank, item in enumerate(rankable, start=1):
        item["rank"] = rank
        if item["relative_volume_open"] < float(active["minimum_relative_volume"]):
            status = "BELOW_MINIMUM_RELATIVE_VOLUME"
            selected = False
        elif item["direction"] == "DOJI":
            status = "DOJI_SKIPPED"
            selected = False
        else:
            qualified_seen += 1
            selected = qualified_seen <= int(active["top_n"])
            status = "SELECTED" if selected else "BEYOND_TOP_N"
        item["selected"] = selected
        item["selection_status"] = status
        item["signal"] = (
            simulate_orb_signal(
                item["path"], direction=item["direction"],
                first_candle_high=item["first_candle_high"],
                first_candle_low=item["first_candle_low"],
                atr_14_usd=item["atr_14_usd"],
                atr_stop_fraction=float(active["atr_stop_fraction"]),
                entry_cutoff_time=active["entry_cutoff_time"],
            )
            if item["direction"] != "DOJI" else {
                "trade_executed": False, "status": "DOJI_SKIPPED"
            }
        )
        item["signal_triggered"] = bool(item["signal"].get("trade_executed"))
        ranked.append(item)
    not_rankable = [item for item in candidates if item["relative_volume_open"] is None]
    for item in not_rankable:
        item.update({
            "rank": None, "selected": False,
            "selection_status": "NOT_RANKABLE",
            "signal": {"trade_executed": False, "status": "NOT_RANKABLE"},
        })
    rows = ranked
    costs = load_execution_costs()
    paper_long_only = _sized_results(
        rows, strategy_capital_usd=strategy_capital_usd,
        leverage_cap=float(active["paper_leverage_cap"]), allow_shorts=False,
        exhausted_reason="LEVERAGE_CAP_EXCEEDED", costs=costs,
    )
    paper_long_short = _sized_results(
        rows, strategy_capital_usd=strategy_capital_usd,
        leverage_cap=float(active["paper_leverage_cap"]), allow_shorts=True,
        exhausted_reason="LEVERAGE_CAP_EXCEEDED", costs=costs,
    )
    cash_long_only = _sized_results(
        rows, strategy_capital_usd=strategy_capital_usd,
        leverage_cap=float(active["cash_leverage_cap"]), allow_shorts=False,
        exhausted_reason="CASH_EXHAUSTED", costs=costs,
    )
    for item in rows:
        key = item["stable_security_id"]
        item["executions"] = {
            "paper_long_only": paper_long_only[key],
            "paper_long_short": paper_long_short[key],
            "cash_long_only": cash_long_only[key],
        }
        item.pop("path", None)
        item.pop("signal", None)
        item.pop("risk_per_trade_pct", None)

    selected_rows = [item for item in rows if item["selected"]]
    long_triggers = sum(
        item["signal_triggered"] and item["direction"] == "LONG"
        for item in selected_rows
    )
    all_triggers = sum(item["signal_triggered"] for item in selected_rows)
    summaries = {
        "paper": {
            "long_only": _summary(
                [item["executions"]["paper_long_only"] for item in selected_rows],
                strategy_capital_usd,
                signal_triggered=long_triggers,
            ),
            "long_plus_short": _summary(
                [item["executions"]["paper_long_short"] for item in selected_rows],
                strategy_capital_usd,
                signal_triggered=all_triggers,
            ),
        },
        "cash": {
            "long_only": _summary(
                [item["executions"]["cash_long_only"] for item in selected_rows],
                strategy_capital_usd,
                signal_triggered=long_triggers,
            ),
            "long_plus_short": _summary(
                [item["executions"]["cash_long_only"] for item in selected_rows],
                strategy_capital_usd,
                signal_triggered=long_triggers,
            ),
            "shorts_enabled": False,
        },
    }
    result = {
        "orb_version": orb_version,
        "trading_date": trading_date,
        "orb_universe_count": len(candidates),
        "universe_exclusions": universe_exclusions,
        "unresolved_bar_rows": dict(sorted(unresolved.items())),
        "ranked_candidates": rows,
        "not_rankable": [
            {
                key: item[key]
                for key in (
                    "orb_version", "trading_date", "ticker",
                    "stable_security_id", "baseline_sessions_used",
                    "selection_status", "outside_scout_band",
                    "scout_scorable", "scout_selected", "top_10_mover",
                )
            }
            for item in sorted(
                not_rankable,
                key=lambda value: (value["ticker"], value["stable_security_id"]),
            )
        ],
        "summary": summaries,
        "overlap": {
            "scout_scorable": sum(item["scout_scorable"] for item in selected_rows),
            "scout_selected": sum(item["scout_selected"] for item in selected_rows),
            "top_10_mover": sum(item["top_10_mover"] for item in selected_rows),
        },
        "cost_model_id": costs["cost_model_id"],
    }
    _write_outcomes_csv(output_path, result)
    return result


def write_orb_daily_csv(output_root: Path, completed_dates: list[str]) -> Path:
    path = Path(output_root) / "orb_daily.csv"
    fields = [
        "orb_version", "trading_date", "variant", "cohort",
        "net_realized_return_pct", "gross_realized_return_pct",
        "equity_index",
    ]
    rows: list[dict[str, Any]] = []
    equity: dict[tuple[str, str], float] = defaultdict(lambda: 1.0)
    for trading_date in completed_dates:
        postmortem = Path(output_root) / "days" / trading_date / f"postmortem_{trading_date}.json"
        if not postmortem.is_file():
            continue
        orb = json.loads(postmortem.read_text(encoding="utf-8")).get("orb")
        if not orb:
            continue
        for variant, cohorts in orb["summary"].items():
            if not isinstance(cohorts, dict):
                continue
            for cohort in ("long_only", "long_plus_short"):
                summary = cohorts.get(cohort)
                if summary:
                    key = (variant, cohort)
                    equity[key] *= 1 + float(summary["net_realized_return_pct"]) / 100
                    rows.append({
                        "orb_version": orb["orb_version"],
                        "trading_date": trading_date,
                        "variant": variant,
                        "cohort": cohort,
                        "net_realized_return_pct": summary["net_realized_return_pct"],
                        "gross_realized_return_pct": summary["gross_realized_return_pct"],
                        "equity_index": equity[key],
                    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path
