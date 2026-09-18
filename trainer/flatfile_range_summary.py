from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any


VERDICT_LABELS = {
    "SCOUT_OUTPERFORMED": "WIN",
    "SCOUT_TIED": "TIE",
    "SCOUT_UNDERPERFORMED": "MISS",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _selected(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("qualification_selected")
        or candidate.get("research_selected")
    )


def _day_wall_seconds(day: dict[str, Any]) -> float:
    return sum(
        float(value)
        for value in day.get("phase_wall_time_seconds", {}).values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _fmt(value: float | int | None, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.4f}{suffix}"


def _day_evidence(
    output_root: Path,
    trading_date: str,
) -> dict[str, Any]:
    day_root = output_root / "days" / trading_date
    scout = _load_json(day_root / "research_alpha_output.json") or {}
    outcome = _load_json(day_root / "end_of_day_outcome.json") or {}
    benchmark = _load_json(day_root / "benchmark_result.json") or {}
    selected_tickers = {
        item["ticker"]
        for item in scout.get("candidates", [])
        if _selected(item)
    }
    selected_count = len(selected_tickers)
    if not scout and outcome:
        selected_count = sum(
            bool(item.get("selected")) for item in outcome.get("outcomes", [])
        )

    primary_summary = benchmark.get(
        "combined_summary", benchmark.get("scout_summary", {})
    )
    primary_return = primary_summary.get("realized_return_pct")
    primary_captures = []
    primary_id = "execution_policy_v1.0"
    for item in outcome.get("outcomes", []):
        if selected_tickers and item.get("ticker") not in selected_tickers:
            continue
        if not selected_tickers and not item.get("selected"):
            continue
        execution = item.get("execution_result", {})
        primary_id = execution.get("policy_id", primary_id)
        capture = execution.get("capture_ratio")
        if capture is not None:
            primary_captures.append(float(capture))

    policies = {
        primary_id: {
            "return_pct": (
                float(primary_return) if primary_return is not None else None
            ),
            "captures": primary_captures,
        }
    }
    atr_id = None
    for comparison in outcome.get("policy_comparisons", []):
        policy_id = comparison["policy_id"]
        captures = [
            float(item["execution_result"]["capture_ratio"])
            for item in comparison.get("executions", [])
            if item.get("cohort") == "SCOUT_SELECTION"
            and item.get("execution_result", {}).get("capture_ratio") is not None
        ]
        value = comparison.get("summary", {}).get("realized_return_pct")
        policies[policy_id] = {
            "return_pct": float(value) if value is not None else None,
            "captures": captures,
        }
        if comparison.get("exit_mode") == "ATR" or "atr" in policy_id.lower():
            atr_id = policy_id

    result_code = benchmark.get("comparison", {}).get("result_code")
    return {
        "selected_count": selected_count if scout or outcome else None,
        "primary_policy_id": primary_id,
        "atr_policy_id": atr_id,
        "policies": policies,
        "verdict": VERDICT_LABELS.get(result_code),
    }


def render_range_summary(
    manifest: dict[str, Any],
    output_root: Path,
) -> str:
    evidence_by_date: dict[str, dict[str, Any]] = {}
    lines = ["Flat-File Replay Range summary"]
    for day in manifest.get("days", []):
        trading_date = day["trading_date"]
        evidence = _day_evidence(output_root, trading_date)
        evidence_by_date[trading_date] = evidence
        primary = evidence["policies"].get(
            evidence["primary_policy_id"], {}
        )
        atr = evidence["policies"].get(evidence["atr_policy_id"], {})
        lines.append(
            f"{trading_date} "
            f"status={day.get('status', 'PENDING')} "
            f"eligible={day.get('universe_size', 0)} "
            f"scorable={day.get('scored_ticker_count', 0)} "
            f"selected={_fmt(evidence['selected_count'])} "
            f"primary_return={_fmt(primary.get('return_pct'), suffix='%')} "
            f"atr_return={_fmt(atr.get('return_pct'), suffix='%')} "
            f"excluded_tickers={len(day.get('excluded_tickers', []))} "
            f"wall_seconds={_fmt(_day_wall_seconds(day))}"
        )

    requested = manifest.get("requested_dates", [])
    completed = manifest.get("completed_dates", [])
    remaining = manifest.get("remaining_dates")
    if remaining is None:
        remaining = [value for value in requested if value not in completed]
    verdicts = Counter(
        evidence["verdict"]
        for date_value, evidence in evidence_by_date.items()
        if date_value in completed and evidence["verdict"] is not None
    )
    policy_returns: dict[str, float] = Counter()
    policy_captures: dict[str, list[float]] = {}
    for date_value in completed:
        evidence = evidence_by_date.get(date_value)
        if evidence is None:
            evidence = _day_evidence(output_root, date_value)
        for policy_id, values in evidence["policies"].items():
            if values["return_pct"] is not None:
                policy_returns[policy_id] += float(values["return_pct"])
            policy_captures.setdefault(policy_id, []).extend(values["captures"])

    primary_ids = [
        evidence["primary_policy_id"]
        for evidence in evidence_by_date.values()
        if evidence["primary_policy_id"]
    ]
    atr_ids = [
        evidence["atr_policy_id"]
        for evidence in evidence_by_date.values()
        if evidence["atr_policy_id"]
    ]
    primary_id = primary_ids[0] if primary_ids else "execution_policy_v1.0"
    atr_id = atr_ids[0] if atr_ids else "execution_policy_atr_v1.0"
    total_wall = sum(_day_wall_seconds(day) for day in manifest.get("days", []))
    lines.extend([
        "Cumulative",
        f"Days completed/requested: {len(completed)}/{len(requested)}",
        "Days remaining: " + (", ".join(remaining) if remaining else "none"),
        (
            "WIN/TIE/MISS: "
            f"{verdicts['WIN']}/{verdicts['TIE']}/{verdicts['MISS']}"
        ),
        f"{primary_id} cumulative return: {_fmt(policy_returns.get(primary_id, 0.0), suffix='%')}",
        f"{atr_id} cumulative return: {_fmt(policy_returns.get(atr_id, 0.0), suffix='%')}",
        (
            f"{primary_id} cumulative capture ratio: "
            f"{_fmt(mean(policy_captures.get(primary_id, [])) if policy_captures.get(primary_id) else None)}"
        ),
        (
            f"{atr_id} cumulative capture ratio: "
            f"{_fmt(mean(policy_captures.get(atr_id, [])) if policy_captures.get(atr_id) else None)}"
        ),
        f"Total wall time: {_fmt(total_wall)} seconds",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a multi-day flat-file replay summary."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.output_root / "flatfile_replay_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(render_range_summary(manifest, args.output_root), end="")


if __name__ == "__main__":
    main()
