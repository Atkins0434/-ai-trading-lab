from __future__ import annotations

from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any

from trainer.price_volume import PriceVolumeError, calculate_price_volume_metrics
from trainer.extended_alpha_metrics import EXTENDED_METRIC_IDS, extended_components
from trainer.sector_metrics import (
    SECTOR_METRIC_IDS, availability_metadata, score_percentages,
    benchmark_symbol, build_sector_context, sector_components,
)
from trainer.rate_control import load_massive_plan
from trainer.replay_engine import ReplayError, validate_freeze_timestamp, validate_point_in_time_inputs
from trainer.scout_engine import (
    evaluate_liquidity_guardrail,
    evaluate_order_book_depth_guardrail,
    evaluate_spread_guardrail,
    observed_component,
    score_relative_volume,
)
from trainer.validate_contracts import ContractError, load_json, validate_contract


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "scout_alpha_v1.json"
REVERSAL_CONFIG_PATH = ROOT / "config" / "scout_reversal_v1.json"
REGISTRY_PATH = ROOT / "config" / "feature_registry_alpha_v1.json"
MAXIMUM_POINTS = 88
ALPHA12_MAXIMUM_POINTS = 48
RUBRIC_VERSION = "alpha_v1.2_22m"


class ResearchScoutError(Exception):
    """Raised when Research Scout Alpha cannot score a frozen snapshot."""


def _load_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG_PATH)
    reversal = load_json(REVERSAL_CONFIG_PATH)
    registry = load_json(REGISTRY_PATH)
    metrics = [m for m in registry.get("metrics", []) if m["availability"] != "UNAVAILABLE"]
    ids = [metric.get("id") for metric in metrics]
    if (
        config.get("execution_allowed") is not False
        or config.get("scoring", {}).get("maximum_points") != MAXIMUM_POINTS
        or registry.get("maximum_points") != MAXIMUM_POINTS
        or registry.get("metric_count") != 22
        or [m.get("number") for m in registry["metrics"]] != list(range(1, 31))
        or len({m.get("id") for m in registry["metrics"]}) != 30
        or any(m.get("availability") not in {"AVAILABLE", "DATA_DEPENDENT", "UNAVAILABLE"} for m in registry["metrics"])
        or len(metrics) != 22
        or len(set(ids)) != 22
        or config["scoring"].get("denominator_policy") != "DATA_AVAILABILITY"
        or registry.get("denominator_policy") != "DATA_AVAILABILITY"
        or not 0 <= int(config.get("shadow_min_premarket_bars", -1))
        < int(config["minimum_real_bars_60m"])
        or reversal.get("pattern_id") != "research_reversal_v1.0"
        or reversal.get("execution_allowed") is not False
        or reversal.get("morning_freeze_time") != config["morning_freeze_time"]
        or reversal.get("minimum_real_bars_60m")
        != config["minimum_real_bars_60m"]
        or reversal.get("liquidity") != config["liquidity"]
        or reversal.get("spread") != config["spread"]
        or reversal.get("order_book_depth") != config["order_book_depth"]
        or set(reversal.get("scoring", {}).get("metrics", {})) != set(ids[:12])
        or set(ids[12:]) != set(EXTENDED_METRIC_IDS) | set(SECTOR_METRIC_IDS)
    ):
        raise ResearchScoutError("Alpha requires 22 reachable metrics/88 points and reversal requires the original 12.")
    return config, registry, reversal


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


def _numeric_raw(component: dict[str, Any], path: str | None = None) -> float | None:
    if component.get("status") != "OBSERVED":
        return None
    value = component.get("raw_value")
    if path is not None:
        value = value.get(path) if isinstance(value, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def _threshold_score(value: float, thresholds: dict[str, Any]) -> int:
    score = 0
    for points in range(1, 5):
        if value >= float(thresholds[str(points)]):
            score = points
    return score


def _reversal_component_scores(
    component_scores: dict[str, dict[str, Any]],
    reversal_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    scored: dict[str, dict[str, Any]] = {}
    rules = reversal_config["scoring"]["metrics"]
    for metric_id, primary in component_scores.items():
        component = deepcopy(primary)
        rule = rules[metric_id]
        mode = rule["mode"]
        if primary.get("status") != "OBSERVED":
            component["score"] = None
        elif mode == "PRIMARY_SCORE":
            component["score"] = primary.get("score")
        else:
            raw = _numeric_raw(primary, rule.get("raw_value_path"))
            if raw is None:
                component["score"] = None
            elif mode == "NEGATIVE_MAGNITUDE":
                component["score"] = (
                    _threshold_score(abs(raw), rule["thresholds"])
                    if raw < 0
                    else 0
                )
            elif mode == "MAXIMUM":
                component["score"] = next(
                    (
                        points
                        for points in range(4, 0, -1)
                        if raw <= float(rule["thresholds"][str(points)])
                    ),
                    0,
                )
            else:
                raise ResearchScoutError(
                    f"Unknown reversal scoring mode for {metric_id}: {mode}"
                )
        component["reason_code"] = f"REVERSAL_{metric_id.upper()}_SCORE"
        component["calculation_version"] = reversal_config["pattern_id"]
        scored[metric_id] = component
    return scored


def _reversal_score(
    component_scores: dict[str, dict[str, Any]],
    reversal_config: dict[str, Any],
) -> tuple[str, int | None, float | None, dict[str, dict[str, Any]]]:
    reversal_components = _reversal_component_scores(
        component_scores, reversal_config
    )
    for metric_id, condition in reversal_config["scoring"][
        "required_conditions"
    ].items():
        raw = _numeric_raw(component_scores[metric_id])
        if condition["operator"] != ">":
            raise ResearchScoutError(
                f"Unsupported reversal required-condition operator: {condition['operator']}"
            )
        if raw is None or raw <= float(condition["value"]):
            return (
                "NOT_A_REVERSAL_CANDIDATE",
                None,
                None,
                reversal_components,
            )
    total = sum(item.get("score") or 0 for item in reversal_components.values())
    maximum = int(reversal_config["scoring"]["maximum_points"])
    return "SCORED", total, total / maximum * 100.0, reversal_components


def _select_reversal_exploration(
    candidates: list[dict[str, Any]],
    reversal_config: dict[str, Any],
) -> list[dict[str, Any]]:
    threshold = float(reversal_config["selection_threshold_pct"])
    selected = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["research_eligible"]
            and candidate["status"] == "SCORED"
            and not candidate["qualification_selected"]
            and candidate["selection_basis"] == "NOT_SELECTED"
            and candidate["reversal_status"] == "SCORED"
            and candidate["reversal_score_pct"] >= threshold
        ),
        key=lambda candidate: (
            -candidate["reversal_score_pct"], candidate["ticker"]
        ),
    )[: int(reversal_config["exploration_top_k"])]
    for candidate in selected:
        candidate["research_selected"] = True
        candidate["selection_basis"] = "REVERSAL_EXPLORATION"
        candidate["reason_codes"] = list(dict.fromkeys(
            [
                reason for reason in candidate["reason_codes"]
                if reason != "RESEARCH_ALPHA_NOT_SELECTED"
            ]
            + ["RESEARCH_REVERSAL_EXPLORATION_SELECTED"]
        ))
        candidate["rejection_reasons"] = [
            reason for reason in candidate["rejection_reasons"]
            if reason != "BELOW_RESEARCH_THRESHOLD"
        ]
    return selected


def _score_security(
    security: dict[str, Any],
    config: dict[str, Any],
    registry: dict[str, Any],
    reversal_config: dict[str, Any],
    threshold_pct: float,
    timestamp: str,
    sector_context: dict[str, Any],
) -> dict[str, Any]:
    availability = availability_metadata(registry)
    maximum_points = 4 * availability["reachable_metric_count"]
    component_scores = {
        metric["id"]: _missing_component() for metric in registry["metrics"]
        if metric["availability"] != "UNAVAILABLE"
    }
    for metric_id in EXTENDED_METRIC_IDS:
        component_scores[metric_id]["calculation_version"] = (
            "rsi_cutler_window_v1.0" if metric_id.startswith("relative_strength_index_")
            else "research_alpha_extended_v1.0"
        )
    for metric_id in SECTOR_METRIC_IDS:
        component_scores[metric_id]["calculation_version"] = "sector_context_v1.0"
    market_data = security["market_data"]
    real_bar_observation = market_data.get("real_bar_count_60m", {})
    real_bar_count_60m = real_bar_observation.get("value")
    if real_bar_count_60m is None:
        from datetime import datetime, timedelta

        freeze = datetime.fromisoformat(timestamp)
        start = freeze - timedelta(minutes=60)
        real_bar_count_60m = sum(
            start
            <= datetime.fromisoformat(bar["timestamp"].replace("Z", "+00:00"))
            < freeze
            for bar in security.get("premarket_bars", [])
        )
    real_bar_count_60m = int(real_bar_count_60m)
    selection_minimum = int(config["minimum_real_bars_60m"])
    shadow_minimum = int(config["shadow_min_premarket_bars"])
    if real_bar_count_60m < shadow_minimum:
        threshold_points = ceil(threshold_pct / 100 * maximum_points)
        return {
            "ticker": security["ticker"],
            "timestamp": timestamp,
            "status": "NOT_SCORABLE",
            "shadow": False,
            "research_eligible": bool(security["eligible"]),
            "qualification_selected": False,
            "research_selected": False,
            "selection_basis": "NOT_SELECTED",
            "execution_eligible": False,
            "rank": None,
            "total_score": 0,
            "maximum_possible_score": maximum_points,
            "score_pct": 0.0,
            "score_pct_reachable": 0.0,
            "score_pct_fixed120": 0.0,
            **availability,
            "benchmark_symbol": benchmark_symbol(security, config),
            "threshold_points": threshold_points,
            "component_scores": component_scores,
            "reversal_status": "NOT_SCORABLE",
            "reversal_total_score": None,
            "reversal_score_pct": None,
            "alpha12_total_score": 0,
            "alpha12_score_pct": 0.0,
            "rubric_version": RUBRIC_VERSION,
            "reversal_component_scores": {
                key: deepcopy(value) for key, value in component_scores.items()
                if key in reversal_config["scoring"]["metrics"]
            },
            "guardrails": {},
            "reason_codes": [
                "INSUFFICIENT_PREMARKET_BARS",
                "EXECUTION_DISABLED_RESEARCH_ONLY",
            ],
            "rejection_reasons": ["INSUFFICIENT_PREMARKET_BARS"],
            "unavailable_execution_checks": config[
                "unavailable_execution_checks"
            ],
        }
    shadow = real_bar_count_60m < selection_minimum
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
        calculated = calculate_price_volume_metrics(
            security, config, timestamp
        )
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
        "historical_spread": evaluate_spread_guardrail(market_data, config),
        "order_book_depth": evaluate_order_book_depth_guardrail(
            market_data, config
        ),
    }
    rejection_reasons: list[str] = []
    guardrail_reason_codes: list[str] = []
    for guardrail in guardrails.values():
        guardrail_reason_codes.append(guardrail["reason_code"])
        if guardrail["action"] == "REJECT":
            rejection_reasons.append(guardrail["reason_code"])
    if not security["eligible"]:
        rejection_reasons.extend(security.get("eligibility_reasons", ["UNIVERSE_INELIGIBLE"]))

    alpha12_components = {
        key: value for key, value in component_scores.items()
        if key in reversal_config["scoring"]["metrics"]
    }
    alpha12_total_score = sum(item["score"] or 0 for item in alpha12_components.values())
    component_scores.update(extended_components(security, config, timestamp))
    component_scores.update(sector_components(security, config, timestamp, sector_context))
    total_score = sum(item["score"] or 0 for item in component_scores.values())
    (
        reversal_status,
        reversal_total_score,
        reversal_score_pct,
        reversal_component_scores,
    ) = _reversal_score(alpha12_components, reversal_config)
    threshold_points = ceil(threshold_pct / 100 * maximum_points)
    research_eligible = security["eligible"] and not rejection_reasons
    research_selected = (
        research_eligible and not shadow and total_score >= threshold_points
    )
    reason_codes = guardrail_reason_codes + (
        ["SHADOW_SCORED_DIAGNOSTIC_ONLY", "INSUFFICIENT_PREMARKET_BARS"]
        if shadow
        else []
    ) + [
        "RESEARCH_ALPHA_SELECTED"
        if research_selected
        else "RESEARCH_ALPHA_NOT_SELECTED",
        "EXECUTION_DISABLED_RESEARCH_ONLY",
    ]
    if shadow:
        rejection_reasons.append("INSUFFICIENT_PREMARKET_BARS")
    elif research_eligible and not research_selected:
        rejection_reasons.append("BELOW_RESEARCH_THRESHOLD")

    return {
        "ticker": security["ticker"],
        "timestamp": timestamp,
        "status": "SHADOW_SCORED" if shadow else "SCORED",
        "shadow": shadow,
        "research_eligible": research_eligible,
        "qualification_selected": research_selected,
        "research_selected": research_selected,
        "selection_basis": "QUALIFYING_THRESHOLD" if research_selected else "NOT_SELECTED",
        "execution_eligible": False,
        "rank": None,
        "total_score": total_score,
        "alpha12_total_score": alpha12_total_score,
        "alpha12_score_pct": alpha12_total_score / ALPHA12_MAXIMUM_POINTS * 100,
        "rubric_version": RUBRIC_VERSION,
        "maximum_possible_score": maximum_points,
        **score_percentages(total_score, availability["reachable_metric_count"]),
        **availability,
        "benchmark_symbol": benchmark_symbol(security, config),
        "threshold_points": threshold_points,
        "component_scores": component_scores,
        "reversal_status": reversal_status,
        "reversal_total_score": reversal_total_score,
        "reversal_score_pct": reversal_score_pct,
        "reversal_component_scores": reversal_component_scores,
        "guardrails": guardrails,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
        "unavailable_execution_checks": config["unavailable_execution_checks"],
    }


def run_research_scout_alpha(
    snapshot: dict[str, Any],
    threshold_pct: float | None = None,
    exploration_top_k: int = 0,
) -> dict[str, Any]:
    """Run the isolated availability-denominator Alpha; it can never authorize execution."""
    try:
        validate_contract("historical_snapshot", snapshot)
        validate_point_in_time_inputs(snapshot)
    except (ContractError, ReplayError) as exc:
        raise ResearchScoutError(f"Invalid Alpha snapshot: {exc}") from exc

    config, registry, reversal_config = _load_contract()
    try:
        validate_freeze_timestamp(snapshot, config=config)
    except ReplayError as exc:
        raise ResearchScoutError(f"Invalid Alpha snapshot: {exc}") from exc
    threshold = float(
        config["selection_threshold_pct"] if threshold_pct is None else threshold_pct
    )
    if not 0 <= threshold <= 100:
        raise ResearchScoutError("Alpha threshold must be between 0 and 100.")
    if exploration_top_k < 0 or exploration_top_k > 5:
        raise ResearchScoutError("Alpha exploration_top_k must be between 0 and 5.")
    sector_context = build_sector_context(snapshot, config)
    candidates = [
        _score_security(
            security,
            config,
            registry,
            reversal_config,
            threshold,
            snapshot["freeze_timestamp"],
            sector_context,
        )
        for security in snapshot["securities"]
    ]
    reversal_threshold = float(reversal_config["selection_threshold_pct"])
    reversal_pool = _select_reversal_exploration(
        candidates, reversal_config
    )
    exploration_pool = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["research_eligible"]
            and candidate["status"] == "SCORED"
            and not candidate["qualification_selected"]
            and candidate["selection_basis"] == "NOT_SELECTED"
        ),
        key=lambda candidate: (-candidate["score_pct"], candidate["ticker"]),
    )[:exploration_top_k]
    for candidate in exploration_pool:
        candidate["research_selected"] = True
        candidate["selection_basis"] = "EXPLORATION_TOP_K"
        candidate["reason_codes"] = list(dict.fromkeys(
            [
                reason for reason in candidate["reason_codes"]
                if reason != "RESEARCH_ALPHA_NOT_SELECTED"
            ]
            + ["RESEARCH_ALPHA_EXPLORATION_SELECTED"]
        ))
        candidate["rejection_reasons"] = [
            reason for reason in candidate["rejection_reasons"]
            if reason != "BELOW_RESEARCH_THRESHOLD"
        ]

    qualifying = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["selection_basis"] == "QUALIFYING_THRESHOLD"
        ),
        key=lambda candidate: (-candidate["score_pct"], candidate["ticker"]),
    )
    exploration = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["selection_basis"] == "EXPLORATION_TOP_K"
        ),
        key=lambda candidate: (-candidate["score_pct"], candidate["ticker"]),
    )
    reversal_exploration = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["selection_basis"] == "REVERSAL_EXPLORATION"
        ),
        key=lambda candidate: (
            -candidate["reversal_score_pct"], candidate["ticker"]
        ),
    )
    # Rank the production-like qualifying basket first. Exploration names are
    # explicitly appended so they can never take an execution slot from a
    # qualifying candidate, even if a future scoring change makes their raw
    # score ordering overlap.
    selected = qualifying + exploration + reversal_exploration
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank

    result = {
        "replay_id": snapshot["replay_id"],
        "scout_version": config["scout_id"],
        "rubric_version": RUBRIC_VERSION,
        "mode": "RESEARCH_ONLY",
        **availability_metadata(registry),
        "spy_premarket_return_pct": sector_context["benchmark_returns"]["SPY"],
        "snapshot_timestamp": snapshot["freeze_timestamp"],
        "massive_plan": snapshot.get("massive_plan", load_massive_plan()),
        "scoring_threshold_pct": threshold,
        "reversal_pattern_id": reversal_config["pattern_id"],
        "reversal_scoring_threshold_pct": reversal_threshold,
        "reversal_exploration_top_k": int(reversal_config["exploration_top_k"]),
        "universe_mode": snapshot["universe_mode"],
        "universe_manifest_hash": snapshot["universe_manifest_hash"],
        "universe_coverage": snapshot["universe_coverage"],
        "research_evidence": snapshot["research_evidence"],
        "promotion_eligible": snapshot["promotion_eligible"],
        "eligible_universe_count": sum(c["research_eligible"] for c in candidates),
        "scorable_candidate_count": sum(
            c["status"] == "SCORED" for c in candidates
        ),
        "shadow_scored_candidate_count": sum(
            c["status"] == "SHADOW_SCORED" for c in candidates
        ),
        "not_scorable_candidate_count": sum(
            c["status"] != "SCORED" for c in candidates
        ),
        "qualifying_candidate_count": sum(c["qualification_selected"] for c in candidates),
        "exploration_top_k": exploration_top_k,
        "exploration_candidate_count": sum(
            c["selection_basis"] in {
                "EXPLORATION_TOP_K", "REVERSAL_EXPLORATION"
            }
            for c in candidates
        ),
        "reversal_candidate_count": sum(
            c["reversal_score_pct"] is not None
            and c["reversal_score_pct"] >= reversal_threshold
            for c in candidates
        ),
        "reversal_selected_count": len(reversal_exploration),
        "selected_candidate_count": len(selected),
        "candidates": candidates,
    }
    try:
        validate_contract("research_scout_output", result)
    except ContractError as exc:
        raise ResearchScoutError(f"Alpha produced invalid output: {exc}") from exc
    return result
