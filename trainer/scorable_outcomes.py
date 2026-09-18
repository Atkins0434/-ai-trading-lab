from __future__ import annotations

from trainer.output_paths import daily_path, day_file, legacy_name

import csv
import json
from datetime import datetime, time
from pathlib import Path
from typing import Any, Iterable

from trainer.scorable_outcomes_schema import (
    ELIGIBLE_OUTCOMES_COLUMNS,
    METRIC_IDS,
    scorable_outcomes_columns,
)


RAW_VALUE_PATHS = {
    "premarket_range_expansion": "expansion_ratio",
    "price_vs_premarket_vwap": "price_vs_vwap_pct",
    "price_volume_confirmation": "aligned_positive_windows",
}


def _observation(security: dict[str, Any], name: str) -> Any:
    return security.get("market_data", {}).get(name, {}).get("value")


def _raw(component: dict[str, Any], metric_id: str) -> Any:
    value = component.get("raw_value")
    path = RAW_VALUE_PATHS.get(metric_id)
    return value.get(path) if path and isinstance(value, dict) else value


def _path_fields(bars: list[dict[str, Any]]) -> dict[str, Any]:
    if not bars:
        return {
            "open_0930": None, "day_high": None, "day_low": None,
            "close": None, "day_mfe_pct": None, "day_mae_pct": None,
            "close_vs_open_pct": None, "time_of_day_high": None,
            "high_0930_1000_pct": None, "return_0930_1000_pct": None,
            "no_regular_session_path": True,
        }
    ordered = sorted(bars, key=lambda bar: bar["timestamp"])
    opening = float(ordered[0]["open"])
    high = max(float(bar["high"]) for bar in ordered)
    low = min(float(bar["low"]) for bar in ordered)
    close = float(ordered[-1]["close"])
    high_bar = next(bar for bar in ordered if float(bar["high"]) == high)
    through_1000 = [
        bar for bar in ordered
        if datetime.fromisoformat(bar["timestamp"].replace("Z", "+00:00")).time()
        < time(10, 0)
    ]
    return_1000 = float(through_1000[-1]["close"]) if through_1000 else None
    high_1000 = max(float(bar["high"]) for bar in through_1000) if through_1000 else None
    return {
        "open_0930": opening,
        "day_high": high,
        "day_low": low,
        "close": close,
        "day_mfe_pct": (high / opening - 1) * 100,
        "day_mae_pct": (low / opening - 1) * 100,
        "close_vs_open_pct": (close / opening - 1) * 100,
        "time_of_day_high": high_bar["timestamp"],
        "high_0930_1000_pct": (
            (high_1000 / opening - 1) * 100 if high_1000 is not None else None
        ),
        "return_0930_1000_pct": (
            (return_1000 / opening - 1) * 100
            if return_1000 is not None else None
        ),
        "no_regular_session_path": False,
    }


def _policy_ids(outcome: dict[str, Any]) -> list[str]:
    primary = next(
        (
            item.get("execution_result", {}).get("policy_id")
            for item in outcome.get("outcomes", [])
            if item.get("execution_result", {}).get("policy_id")
        ),
        "execution_policy_v1.0",
    )
    return [primary] + [
        item["policy_id"] for item in outcome.get("policy_comparisons", [])
    ]


def _comparison_by_ticker(
    outcome: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for comparison in outcome.get("policy_comparisons", []):
        policy = result.setdefault(comparison["policy_id"], {})
        for item in comparison.get("executions", []):
            if item.get("cohort") == "SCOUT_SELECTION":
                policy[item["ticker"]] = item["execution_result"]
    return result


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, columns: Iterable[str], rows: list[dict[str, Any]]) -> Path:
    fields = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fields})
    return path


def export_daily_outcomes(
    day_dir: Path,
    snapshot: dict[str, Any],
    scout: dict[str, Any],
    outcome: dict[str, Any],
    benchmark: dict[str, Any],
) -> tuple[Path, Path]:
    trading_date = snapshot.get("trading_date", day_dir.name)
    securities = {
        item["ticker"]: item for item in snapshot.get("securities", [])
    }
    outcomes = {item["ticker"]: item for item in outcome["outcomes"]}
    candidates = {
        item["ticker"]: item
        for item in scout.get("candidates", [])
        if item.get("ticker")
    }
    ranks = {
        item["ticker"]: item["benchmark_rank"]
        for item in benchmark.get("benchmark_candidates", [])
    }
    policy_ids = _policy_ids(outcome)
    comparisons = _comparison_by_ticker(outcome)
    scorable_rows: list[dict[str, Any]] = []
    for ticker, candidate in candidates.items():
        if candidate.get("status") not in {"SCORED", "SELECTED", "SHADOW_SCORED"}:
            continue
        security = securities[ticker]
        observed = outcomes.get(ticker, {})
        row = {
            "trading_date": trading_date,
            "ticker": ticker,
            "stable_security_id": security.get("stable_security_id"),
            "shadow": bool(candidate.get("shadow", False)),
            "premarket_real_bars_60m": _observation(security, "real_bar_count_60m"),
            "premarket_real_bars_total": _observation(security, "real_bar_count"),
            "prior_close": _observation(security, "previous_close"),
            "price_at_freeze": _observation(security, "last_price"),
            "premarket_dollar_volume_usd": _observation(security, "premarket_dollar_volume"),
            "average_daily_dollar_volume_usd": _observation(security, "average_daily_dollar_volume"),
            "market_cap_usd": security.get("market_cap_usd", {}).get("value"),
            "total_score": candidate.get("total_score"),
            "score_pct": candidate.get("score_pct"),
            "reversal_score_pct": candidate.get("reversal_score_pct"),
            "selected": bool(candidate.get("research_selected", False)),
            "selection_basis": candidate.get("selection_basis"),
            "rejection_reasons": "|".join(candidate.get("rejection_reasons", [])),
            "liquidity_action": candidate.get("guardrails", {}).get("aggregate_liquidity", {}).get("action"),
            "top_10_mover": ticker in ranks,
            "benchmark_rank": ranks.get(ticker),
            **_path_fields(observed.get("intraday_path", [])),
        }
        for metric_id in METRIC_IDS:
            component = candidate.get("component_scores", {}).get(metric_id, {})
            row[f"{metric_id}_raw"] = _raw(component, metric_id)
            row[f"{metric_id}_score"] = component.get("score")
        primary = observed.get("execution_result", {})
        executions = {policy_ids[0]: primary}
        executions.update({
            policy_id: values.get(ticker, {})
            for policy_id, values in comparisons.items()
        })
        for policy_id in policy_ids:
            execution = executions.get(policy_id, {})
            row[f"{policy_id}_exit_reason"] = execution.get("exit_reason")
            row[f"{policy_id}_realized_return_pct"] = execution.get("realized_return_pct")
            row[f"{policy_id}_capture_ratio"] = execution.get("capture_ratio")
            row[f"{policy_id}_stop_distance_pct"] = execution.get("stop_distance_pct")
        scorable_rows.append(row)

    eligible_rows = []
    for ticker, security in securities.items():
        observed = outcomes.get(ticker, {})
        path = _path_fields(observed.get("intraday_path", []))
        candidate = candidates.get(ticker, {})
        eligible_rows.append({
            "trading_date": trading_date,
            "ticker": ticker,
            "stable_security_id": security.get("stable_security_id"),
            "premarket_real_bars_60m": _observation(security, "real_bar_count_60m"),
            "scorable": candidate.get("status") in {"SCORED", "SELECTED", "SHADOW_SCORED"},
            "open_0930": path["open_0930"],
            "close": path["close"],
            "day_mfe_pct": path["day_mfe_pct"],
            "close_vs_open_pct": path["close_vs_open_pct"],
            "top_10_mover": ticker in ranks,
            "realized_return_pct": observed.get("execution_result", {}).get("realized_return_pct"),
        })

    key = lambda row: (row["trading_date"], row["ticker"])
    scorable_rows.sort(key=key)
    eligible_rows.sort(key=key)
    return (
        _write_csv(
            daily_path(day_dir, trading_date, "scorable_outcomes"),
            scorable_outcomes_columns(policy_ids),
            scorable_rows,
        ),
        _write_csv(
            daily_path(day_dir, trading_date, "eligible_outcomes"),
            ELIGIBLE_OUTCOMES_COLUMNS,
            eligible_rows,
        ),
    )


def concatenate_completed_outcomes(output_root: Path, completed_dates: list[str]) -> tuple[Path, Path]:
    outputs = []
    for kind in ("scorable_outcomes", "eligible_outcomes"):
        filename = legacy_name(kind)
        header: list[str] | None = None
        rows: list[dict[str, str]] = []
        for trading_date in sorted(completed_dates):
            path = day_file(output_root, trading_date, kind)
            if not path.is_file():
                day_dir = path.parent
                required = (
                    "historical_snapshot", "research_alpha_output",
                    "end_of_day_outcome", "benchmark_result",
                )
                if not all((daily_path(day_dir, trading_date, name)).is_file() for name in required):
                    raise FileNotFoundError(
                        f"Required daily outcome export is missing: {path}"
                    )
                artifacts = []
                for name in required:
                    with (daily_path(day_dir, trading_date, name)).open(encoding="utf-8") as handle:
                        artifacts.append(json.load(handle))
                export_daily_outcomes(day_dir, *artifacts)
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if header is None:
                    header = list(reader.fieldnames or [])
                elif list(reader.fieldnames or []) != header:
                    raise ValueError(f"Outcome export schema changed across days: {path}")
                rows.extend(reader)
        destination = output_root / filename
        _write_csv(destination, header or (), rows)
        outputs.append(destination)
    return outputs[0], outputs[1]
