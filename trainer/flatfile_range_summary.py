from __future__ import annotations

from trainer.output_paths import daily_path

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
    scout = _load_json(daily_path(day_root, day_root.name, "research_alpha_output")) or {}
    outcome = _load_json(daily_path(day_root, day_root.name, "end_of_day_outcome")) or {}
    benchmark = _load_json(daily_path(day_root, day_root.name, "benchmark_result")) or {}
    postmortem = _load_json(daily_path(day_root, day_root.name, "postmortem")) or {}
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
            "net_return_pct": primary_summary.get("net_realized_return_pct"),
            "net_captures": [item["execution_result"]["net_capture_ratio"] for item in outcome.get("outcomes", []) if item.get("selected") and item.get("execution_result", {}).get("net_capture_ratio") is not None],
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
            "net_return_pct": comparison.get("summary", {}).get("net_realized_return_pct"),
            "net_captures": [item["execution_result"]["net_capture_ratio"] for item in comparison.get("executions", []) if item.get("cohort") == "SCOUT_SELECTION" and item.get("execution_result", {}).get("net_capture_ratio") is not None],
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
        "net_verdict": VERDICT_LABELS.get(benchmark.get("comparison", {}).get("net_result_code")),
        "reachability": postmortem.get("reachability", {}),
        "reversal_cohort": postmortem.get("reversal_cohort", {}),
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
            f"reversal_candidates={int(evidence['reversal_cohort'].get('candidate_count', 0))} "
            f"primary_return={_fmt(primary.get('return_pct'), suffix='%')} "
            f"atr_return={_fmt(atr.get('return_pct'), suffix='%')} "
            f"primary_net_return={_fmt(primary.get('net_return_pct'), suffix='%')} "
            f"atr_net_return={_fmt(atr.get('net_return_pct'), suffix='%')} "
            f"gross_verdict={evidence['verdict']} net_verdict={evidence['net_verdict']} "
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
    net_policy_returns: dict[str, list[float]] = {}
    net_policy_captures: dict[str, list[float]] = {}
    net_verdicts = Counter()
    flips = 0
    policy_captures: dict[str, list[float]] = {}
    reachability: Counter[str] = Counter()
    reversal_candidates_by_day: list[str] = []
    reversal_closed_above: list[bool] = []
    reversal_day_mfe: list[float] = []
    reversal_policy_returns: Counter[str] = Counter()
    net_reversal_returns: dict[str, list[float]] = {}
    for date_value in completed:
        evidence = evidence_by_date.get(date_value)
        if evidence is None:
            evidence = _day_evidence(output_root, date_value)
        net_verdicts[evidence.get("net_verdict")] += 1
        flips += evidence["verdict"] == "WIN" and evidence.get("net_verdict") in {"TIE", "MISS"}
        for policy_id, values in evidence["policies"].items():
            if values.get("net_return_pct") is not None:
                net_policy_returns.setdefault(policy_id, []).append(values["net_return_pct"])
            net_policy_captures.setdefault(policy_id, []).extend(values.get("net_captures", []))
            if values["return_pct"] is not None:
                policy_returns[policy_id] += float(values["return_pct"])
            policy_captures.setdefault(policy_id, []).extend(values["captures"])
        reachability.update({
            key: int(value)
            for key, value in evidence.get("reachability", {}).items()
        })
        cohort = evidence.get("reversal_cohort", {})
        for policy_id, value in cohort.get("net_policy_returns", {}).items():
            if value is not None:
                net_reversal_returns.setdefault(policy_id, []).append(value)
        reversal_candidates_by_day.append(
            f"{date_value}={int(cohort.get('candidate_count', 0))}"
        )
        for candidate in cohort.get("candidates", []):
            if candidate.get("closed_above_open") is not None:
                reversal_closed_above.append(candidate["closed_above_open"])
            if candidate.get("day_mfe_pct") is not None:
                reversal_day_mfe.append(float(candidate["day_mfe_pct"]))
        reversal_policy_returns.update({
            policy_id: float(value)
            for policy_id, value in cohort.get("policy_returns", {}).items()
        })

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
        "Net reversal exploration return sums (costed days): " + (
            "; ".join(f"{key}={_fmt(sum(values), suffix='%')} ({len(values)} days)" for key, values in sorted(net_reversal_returns.items())) or "n/a"
        ),
        f"Net WIN/TIE/MISS: {net_verdicts['WIN']}/{net_verdicts['TIE']}/{net_verdicts['MISS']}",
        f"Gross WIN to net TIE/MISS: {flips}",
        *[f"{policy_id} net cumulative return: {_fmt(sum(values), suffix='%')} (costed days={len(values)}); net capture ratio: {_fmt(mean(net_policy_captures[policy_id]) if net_policy_captures[policy_id] else None)}" for policy_id, values in sorted(net_policy_returns.items())],
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
        (
            "Top-10 reachability: "
            f"bars_0={reachability['bars_0']} "
            f"bars_1_9={reachability['bars_1_9']} "
            f"bars_10_29={reachability['bars_10_29']} "
            f"visible_scored_low={reachability['visible_scored_low']} "
            f"visible_reversal_candidate={reachability['visible_reversal_candidate']} "
            f"visible_guardrail_rejected={reachability['visible_guardrail_rejected']} "
            f"picked={reachability['picked']} "
            f"opening_range_reachable={reachability['opening_range_reachable_count']}"
        ),
        "Reversal candidates per day: " + (
            ", ".join(reversal_candidates_by_day)
            if reversal_candidates_by_day else "none"
        ),
        (
            "Reversal share closing above open: "
            f"{_fmt(sum(reversal_closed_above) / len(reversal_closed_above) if reversal_closed_above else None)}"
        ),
        (
            "Reversal mean day MFE: "
            f"{_fmt(mean(reversal_day_mfe) if reversal_day_mfe else None, suffix='%')}"
        ),
        "Reversal exploration returns by policy: " + (
            ", ".join(
                f"{policy_id}={_fmt(value, suffix='%')}"
                for policy_id, value in sorted(reversal_policy_returns.items())
            )
            if reversal_policy_returns else "none"
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
