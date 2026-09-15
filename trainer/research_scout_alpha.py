from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

from trainer.price_volume import PriceVolumeError, calculate_price_volume_metrics
from trainer.replay_engine import ReplayError, validate_freeze_timestamp, validate_point_in_time_inputs
from trainer.scout_engine import evaluate_liquidity_guardrail, observed_component, score_relative_volume
from trainer.validate_contracts import ContractError, load_json, validate_contract


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "scout_alpha_v1.json"
REGISTRY_PATH = ROOT / "config" / "feature_registry_alpha_v1.json"
MAXIMUM_POINTS = 48


class ResearchScoutError(Exception):
    """Raised when Research Scout Alpha cannot score a frozen snapshot."""


def _load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG_PATH)
    registry = load_json(REGISTRY_PATH)
    metrics = registry.get("metrics", [])
    ids = [metric.get("id") for metric in metrics]
    if (
        config.get("execution_allowed") is not False
        or config.get("scoring", {}).get("maximum_points") != MAXIMUM_POINTS
        or registry.get("maximum_points") != MAXIMUM_POINTS
        or len(metrics) != 12
        or len(set(ids)) != 12
    ):
        raise ResearchScoutError("Alpha must remain a 12-metric, 48-point research contract.")
    return config, registry


def _missing_component() -> dict[str, Any]:
    return {
        "status": "MISSING",
        "score": None,
        "maximum_score": 4,
        "raw_value": None,
        "as_of_timestamp": None,
        "reason_code": "REQUIRED_ALPHA_INPUT_MISSING",
        "calculation_version": "research_alpha_v1.0",
    }


def _not_evaluated(reason_code: str) -> dict[str, Any]:
    return {
        "passed": False,
        "action": "NOT_EVALUATED",
        "observed_value": None,
        "threshold": None,
        "reason_code": reason_code,
    }


def _score_security(
    security: dict[str, Any],
    config: dict[str, Any],
    registry: dict[str, Any],
    threshold_pct: float,
    timestamp: str,
) -> dict[str, Any]:
    component_scores = {
        metric["id"]: _missing_component() for metric in registry["metrics"]
    }
    market_data = security["market_data"]
    relative_observation = market_data.get("relative_volume")
    if relative_observation is not None and relative_observation.get("value") is not None:
        relative_volume = float(relative_observation["value"])
        component_scores["relative_volume"] = observed_component(
            score_relative_volume(relative_volume),
            relative_volume,
            relative_observation["as_of_timestamp"],
            "RELATIVE_VOLUME_SCORE",
            "research_alpha_relative_volume_v1.0",
        )

    try:
        calculated = calculate_price_volume_metrics(security, config)
    except PriceVolumeError as exc:
        raise ResearchScoutError(f"Unable to score {security['ticker']}: {exc}") from exc
    for metric_id, metric in calculated.items():
        component_scores[metric_id] = observed_component(
            metric.score,
            metric.raw_value,
            metric.as_of_timestamp,
            metric.reason_code,
            "research_alpha_price_volume_v1.0",
        )

    liquidity = evaluate_liquidity_guardrail(market_data, config)
    guardrails = {
        "aggregate_liquidity": liquidity,
        "historical_spread": _not_evaluated("MASSIVE_FREE_HAS_NO_HISTORICAL_QUOTES"),
        "order_book_depth": _not_evaluated("MASSIVE_FREE_HAS_NO_ORDER_BOOK_DEPTH"),
    }
    rejection_reasons: list[str] = []
    if liquidity["action"] == "REJECT":
        rejection_reasons.append(liquidity["reason_code"])
    if not security["eligible"]:
        rejection_reasons.extend(security.get("eligibility_reasons", ["UNIVERSE_INELIGIBLE"]))

    total_score = sum(item["score"] or 0 for item in component_scores.values())
    threshold_points = ceil(threshold_pct / 100 * MAXIMUM_POINTS)
    research_eligible = security["eligible"] and not rejection_reasons
    research_selected = research_eligible and total_score >= threshold_points
    reason_codes = [
        "RESEARCH_ALPHA_SELECTED" if research_selected else "RESEARCH_ALPHA_NOT_SELECTED",
        "EXECUTION_DISABLED_RESEARCH_ONLY",
    ]
    if research_eligible and not research_selected:
        rejection_reasons.append("BELOW_RESEARCH_THRESHOLD")

    return {
        "ticker": security["ticker"],
        "timestamp": timestamp,
        "research_eligible": research_eligible,
        "qualification_selected": research_selected,
        "research_selected": research_selected,
        "selection_basis": "QUALIFYING_THRESHOLD" if research_selected else "NOT_SELECTED",
        "execution_eligible": False,
        "rank": None,
        "total_score": total_score,
        "maximum_possible_score": MAXIMUM_POINTS,
        "score_pct": total_score / MAXIMUM_POINTS * 100,
        "threshold_points": threshold_points,
        "component_scores": component_scores,
        "guardrails": guardrails,
        "reason_codes": reason_codes,
        "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
        "unavailable_execution_checks": config["unavailable_execution_checks"],
    }


def run_research_scout_alpha(
    snapshot: dict[str, Any],
    threshold_pct: float | None = None,
    exploration_top_k: int = 0,
) -> dict[str, Any]:
    """Run the isolated 48-point Alpha; it can never authorize execution."""
    try:
        validate_contract("historical_snapshot", snapshot)
        validate_freeze_timestamp(snapshot)
        validate_point_in_time_inputs(snapshot)
    except (ContractError, ReplayError) as exc:
        raise ResearchScoutError(f"Invalid Alpha snapshot: {exc}") from exc

    config, registry = _load_contract()
    threshold = float(
        config["selection_threshold_pct"] if threshold_pct is None else threshold_pct
    )
    if not 0 <= threshold <= 100:
        raise ResearchScoutError("Alpha threshold must be between 0 and 100.")
    if exploration_top_k < 0 or exploration_top_k > 5:
        raise ResearchScoutError("Alpha exploration_top_k must be between 0 and 5.")
    candidates = [
        _score_security(security, config, registry, threshold, snapshot["freeze_timestamp"])
        for security in snapshot["securities"]
    ]
    exploration_pool = sorted(
        (candidate for candidate in candidates if candidate["research_eligible"]),
        key=lambda candidate: (-candidate["score_pct"], candidate["ticker"]),
    )[:exploration_top_k]
    for candidate in exploration_pool:
        if candidate["qualification_selected"]:
            continue
        candidate["research_selected"] = True
        candidate["selection_basis"] = "EXPLORATION_TOP_K"
        candidate["reason_codes"] = [
            "RESEARCH_ALPHA_EXPLORATION_SELECTED",
            "EXECUTION_DISABLED_RESEARCH_ONLY",
        ]
        candidate["rejection_reasons"] = [
            reason for reason in candidate["rejection_reasons"]
            if reason != "BELOW_RESEARCH_THRESHOLD"
        ]

    selected = sorted(
        (candidate for candidate in candidates if candidate["research_selected"]),
        key=lambda candidate: (-candidate["score_pct"], candidate["ticker"]),
    )
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank

    result = {
        "replay_id": snapshot["replay_id"],
        "scout_version": config["scout_id"],
        "mode": "RESEARCH_ONLY",
        "snapshot_timestamp": snapshot["freeze_timestamp"],
        "scoring_threshold_pct": threshold,
        "eligible_universe_count": sum(c["research_eligible"] for c in candidates),
        "qualifying_candidate_count": sum(c["qualification_selected"] for c in candidates),
        "exploration_top_k": exploration_top_k,
        "exploration_candidate_count": sum(c["selection_basis"] == "EXPLORATION_TOP_K" for c in candidates),
        "selected_candidate_count": len(selected),
        "candidates": candidates,
    }
    try:
        validate_contract("research_scout_output", result)
    except ContractError as exc:
        raise ResearchScoutError(f"Alpha produced invalid output: {exc}") from exc
    return result
