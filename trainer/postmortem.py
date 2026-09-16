from __future__ import annotations

import json
from typing import Any

from trainer.validate_contracts import ContractError, validate_contract
from trainer.evidence_eligibility import (
    EvidenceEligibilityError,
    assert_matching_universe,
    require_research_evidence,
)


class PostmortemError(Exception):
    """Raised when a deterministic postmortem cannot be produced."""


def _failure_stage(candidate: dict[str, Any] | None) -> str:
    if candidate is None:
        return "UNIVERSE_EXCLUSION"
    reasons = candidate.get("rejection_reasons", [])
    if any("MISSING" in reason for reason in reasons):
        return "MISSING_DATA"
    if not candidate.get("research_eligible", candidate.get("eligible", False)):
        return "HARD_GUARDRAIL"
    if not candidate.get("research_selected", candidate.get("selected", False)):
        return "BELOW_SELECTION_THRESHOLD"
    return "EXECUTION_REJECTION"


def _differentiators(candidate: dict[str, Any] | None) -> list[dict[str, Any]]:
    if candidate is None:
        return []
    observed = [
        (metric_id, component)
        for metric_id, component in candidate["component_scores"].items()
        if component["status"] == "OBSERVED"
    ]
    observed.sort(key=lambda item: (-(item[1]["score"] or 0), item[0]))
    result = []
    for metric_id, component in observed[:3]:
        raw = component["raw_value"]
        if isinstance(raw, (dict, list)):
            raw = json.dumps(raw, sort_keys=True)
        result.append({
            "name": metric_id,
            "available_before_freeze": True,
            "observed_value": raw,
            "comparison_to_selected_candidates": "Not represented in the selected basket",
            "potential_predictive_value": "Requires at least 30 independent occurrences before proposal eligibility",
        })
    return result


def build_postmortem(
    snapshot: dict[str, Any],
    scout_result: dict[str, Any],
    benchmark_result: dict[str, Any],
) -> dict[str, Any]:
    try:
        universe_metadata = assert_matching_universe(
            snapshot, scout_result, benchmark_result
        )
        require_research_evidence(benchmark_result, consumer="Trainer")
    except EvidenceEligibilityError as exc:
        raise PostmortemError(str(exc)) from exc
    candidates = {item["ticker"]: item for item in scout_result["candidates"]}
    missed = []
    failures: dict[tuple[str, str], dict[str, Any]] = {}
    for item in benchmark_result["benchmark_candidates"]:
        candidate = candidates.get(item["ticker"])
        selected = bool(candidate and candidate.get("research_selected", candidate.get("selected", False)))
        if selected:
            continue
        stage = _failure_stage(candidate)
        reasons = candidate.get("rejection_reasons", []) if candidate else ["NOT_IN_SCOUT_OUTPUT"]
        missed.append({
            "ticker": item["ticker"],
            "benchmark_rank": item["benchmark_rank"],
            "was_in_scout_output": candidate is not None,
            "scout_score_pct": candidate.get("score_pct") if candidate else None,
            "failure_stage": stage,
            "failure_reason_codes": reasons,
            "observable_differentiators": _differentiators(candidate),
            "realized_benchmark_return_pct": item["realized_return_pct"],
            "realized_benchmark_pnl_usd": item["realized_pnl_usd"],
        })
        if stage == "HARD_GUARDRAIL":
            key = ("aggregate_liquidity", "OTHER")
            failure = failures.setdefault(key, {
                "feature_id": key[0],
                "failure_type": key[1],
                "description": "A benchmark mover was excluded by the Alpha liquidity guardrail.",
                "evidence": [],
            })
            failure["evidence"] = sorted(set(failure["evidence"] + reasons))
        elif stage == "BELOW_SELECTION_THRESHOLD":
            key = ("selection_threshold", "THRESHOLD")
            failure = failures.setdefault(key, {
                "feature_id": key[0],
                "failure_type": key[1],
                "description": "A benchmark mover was eligible but remained below the research selection threshold.",
                "evidence": [],
            })
            failure["evidence"].append(
                f"{item['ticker']} score={candidate['score_pct']:.2f}%"
            )

    code = benchmark_result["comparison"]["result_code"]
    result = {
        "replay_id": snapshot["replay_id"],
        "trading_date": snapshot["trading_date"],
        "scout_version": scout_result["scout_version"],
        "trainer_version": "scout_trainer_v1.0",
        "execution_policy_version": benchmark_result["execution_policy_version"],
        **universe_metadata,
        "result": {"SCOUT_OUTPERFORMED": "WIN", "SCOUT_TIED": "TIE", "SCOUT_UNDERPERFORMED": "MISS"}[code],
        "scout_performance": {key: benchmark_result["scout_summary"].get(key) for key in ("realized_return_pct", "realized_pnl_usd", "max_drawdown_pct", "win_rate_pct", "average_capture_ratio")},
        "benchmark_performance": {key: benchmark_result["benchmark_summary"].get(key) for key in ("realized_return_pct", "realized_pnl_usd", "max_drawdown_pct", "win_rate_pct", "average_capture_ratio")},
        "missed_opportunities": missed,
        "feature_failures": list(failures.values()),
        "feature_proposals": [],
        "notes": [
            "Research-only postmortem; no Production Scout configuration was modified.",
            "No feature proposal is eligible from one day; at least 30 independent occurrences are required.",
        ],
    }
    try:
        validate_contract("postmortem", result)
    except ContractError as exc:
        raise PostmortemError(f"Postmortem contract validation failed: {exc}") from exc
    return result
