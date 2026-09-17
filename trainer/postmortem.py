from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from trainer.evidence_eligibility import (
    EvidenceEligibilityError,
    assert_matching_universe,
    require_research_evidence,
)
from trainer.rate_control import load_massive_plan
from trainer.validate_contracts import ContractError, load_json, validate_contract


MISS_CLASSIFICATIONS = {
    "NOT_IN_UNIVERSE",
    "INVISIBLE_AT_FREEZE",
    "VISIBLE_SCORED_LOW",
    "VISIBLE_GUARDRAIL_REJECT",
    "PICKED_EXECUTION_LOSS",
    "UNCLASSIFIED",
}
ALPHA_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "scout_alpha_v1.json"
)


class PostmortemError(Exception):
    """Raised when a deterministic postmortem cannot be produced."""


def _timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PostmortemError("Execution timestamps require a timezone.")
    return parsed


def _component_scores(candidate: dict[str, Any] | None) -> list[dict[str, Any]]:
    if candidate is None:
        return []
    return [
        {
            "metric_id": metric_id,
            "score": component.get("score"),
            "raw_value": component.get("raw_value"),
        }
        for metric_id, component in sorted(candidate["component_scores"].items())
    ]


def _raw_metric(candidate: dict[str, Any], metric_id: str) -> Any:
    component = candidate.get("component_scores", {}).get(metric_id, {})
    if component.get("status") != "OBSERVED":
        return None
    return component.get("raw_value")


def _is_invisible_at_freeze(
    security: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> tuple[bool, list[str]]:
    real_bar_observation = (
        (security or {}).get("market_data", {}).get("real_bar_count_60m", {})
    )
    real_bar_count = real_bar_observation.get("value")
    minimum = int(load_json(ALPHA_CONFIG_PATH)["minimum_real_bars_60m"])
    if (
        isinstance(real_bar_count, (int, float))
        and not isinstance(real_bar_count, bool)
        and float(real_bar_count) < minimum
    ):
        return True, ["INSUFFICIENT_PREMARKET_BARS"]
    relative_volume = _raw_metric(candidate, "relative_volume")
    gap_pct = _raw_metric(candidate, "premarket_gap_strength")
    if (
        isinstance(relative_volume, (int, float))
        and isinstance(gap_pct, (int, float))
        and float(relative_volume) < 1.0
        and abs(float(gap_pct)) < 1.0
    ):
        return True, ["LOW_RELATIVE_VOLUME_AND_SUB_1PCT_GAP"]
    return False, []


def _guardrail_reasons(candidate: dict[str, Any]) -> list[str]:
    reasons = []
    for guardrail_id in (
        "liquidity",
        "aggregate_liquidity",
        "spread",
        "historical_spread",
        "order_book_depth",
    ):
        guardrail = candidate.get("guardrails", {}).get(guardrail_id)
        # NOT_EVALUATED is an explicit audit state, never a rejection.
        if guardrail and guardrail.get("action") == "REJECT":
            reasons.append(guardrail.get("reason_code") or f"{guardrail_id.upper()}_REJECT")
    return sorted(set(reasons))


def _execution_loss_reasons(item: dict[str, Any]) -> list[str]:
    if not item.get("scout_selected"):
        return []
    reasons = []
    maximum = float(item["maximum_capturable_move_pct"])
    realized = float(item.get("scout_realized_return_pct") or 0.0)
    if maximum > 0 and (maximum - realized) / maximum > 0.5:
        reasons.append("EXECUTION_CAPTURE_BELOW_50_PERCENT")
    exit_timestamp = _timestamp(item.get("scout_exit_timestamp"))
    high_timestamp = _timestamp(item.get("day_high_timestamp"))
    if (
        item.get("scout_exit_reason") == "TRAILING_STOP"
        and exit_timestamp is not None
        and high_timestamp is not None
        and exit_timestamp < high_timestamp
    ):
        reasons.append("TRAILING_STOP_BEFORE_DAY_HIGH")
    return reasons


def _classify_miss(
    item: dict[str, Any],
    candidate: dict[str, Any] | None,
    security: dict[str, Any] | None,
) -> tuple[str | None, list[str]]:
    execution_reasons = _execution_loss_reasons(item)
    if item.get("scout_selected"):
        return (
            ("PICKED_EXECUTION_LOSS", execution_reasons)
            if execution_reasons
            else (None, [])
        )
    if candidate is None or not item.get("universe_eligible", False):
        reasons = (security or {}).get("eligibility_reasons", [])
        return "NOT_IN_UNIVERSE", reasons or ["NOT_IN_ELIGIBLE_UNIVERSE"]

    invisible, reasons = _is_invisible_at_freeze(security, candidate)
    if invisible:
        return "INVISIBLE_AT_FREEZE", reasons

    total_score = int(candidate.get("total_score", 0))
    threshold_points = int(candidate.get("threshold_points", 0))
    if total_score < threshold_points:
        return "VISIBLE_SCORED_LOW", ["BELOW_RESEARCH_THRESHOLD"]

    guardrail_reasons = _guardrail_reasons(candidate)
    if guardrail_reasons:
        return "VISIBLE_GUARDRAIL_REJECT", guardrail_reasons
    return "UNCLASSIFIED", sorted(
        set(
            candidate.get("rejection_reasons", [])
            + ["NO_KNOWN_REJECTION_PATH"]
        )
    )


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

    candidates = {
        item["ticker"]: item for item in scout_result["candidates"]
    }
    securities = {
        item["ticker"]: item for item in snapshot.get("securities", [])
    }
    missed: list[dict[str, Any]] = []
    failures: dict[tuple[str, str], dict[str, Any]] = {}
    for item in benchmark_result["benchmark_candidates"]:
        candidate = candidates.get(item["ticker"])
        classification, classification_reasons = _classify_miss(
            item, candidate, securities.get(item["ticker"])
        )
        if classification is None:
            continue
        if classification not in MISS_CLASSIFICATIONS:
            raise PostmortemError(
                f"Unknown miss classification: {classification}"
            )
        reasons = sorted(
            set(
                classification_reasons
                + (candidate.get("rejection_reasons", []) if candidate else [])
            )
        )
        miss = {
            "ticker": item["ticker"],
            "benchmark_rank": item["benchmark_rank"],
            "was_in_scout_output": candidate is not None,
            "scout_selected": bool(item.get("scout_selected", False)),
            "scout_score_pct": candidate.get("score_pct") if candidate else None,
            "miss_classification": classification,
            "failure_reason_codes": reasons,
            "component_scores": _component_scores(candidate),
            "realized_benchmark_return_pct": item["realized_return_pct"],
            "realized_benchmark_pnl_usd": item["realized_pnl_usd"],
            "scout_realized_return_pct": item.get("scout_realized_return_pct"),
            "maximum_capturable_move_pct": item[
                "maximum_capturable_move_pct"
            ],
            "scout_exit_reason": item.get("scout_exit_reason"),
        }
        missed.append(miss)

        if classification == "VISIBLE_GUARDRAIL_REJECT":
            key = ("aggregate_liquidity", "OTHER")
            failure = failures.setdefault(
                key,
                {
                    "feature_id": key[0],
                    "failure_type": key[1],
                    "description": (
                        "A visible benchmark mover met the score threshold but "
                        "was rejected by a liquidity, spread, or depth guardrail."
                    ),
                    "evidence": [],
                },
            )
            failure["evidence"] = sorted(
                set(failure["evidence"] + reasons)
            )
        elif classification == "VISIBLE_SCORED_LOW":
            key = ("selection_threshold", "THRESHOLD")
            failure = failures.setdefault(
                key,
                {
                    "feature_id": key[0],
                    "failure_type": key[1],
                    "description": (
                        "A visible benchmark mover remained below the research "
                        "selection threshold."
                    ),
                    "evidence": [],
                },
            )
            failure["evidence"].append(
                f"{item['ticker']} score={candidate['score_pct']:.2f}%"
            )

    unreachable = [
        item
        for item in missed
        if item["miss_classification"] == "INVISIBLE_AT_FREEZE"
    ]
    execution_review = [
        {
            "ticker": item["ticker"],
            "benchmark_rank": item["benchmark_rank"],
            "realized_return_pct": item["scout_realized_return_pct"],
            "maximum_capturable_move_pct": item[
                "maximum_capturable_move_pct"
            ],
            "exit_reason": item["scout_exit_reason"],
            "reason_codes": item["failure_reason_codes"],
        }
        for item in missed
        if item["miss_classification"] == "PICKED_EXECUTION_LOSS"
    ]
    top_10_count = len(benchmark_result["benchmark_candidates"])
    baselines = benchmark_result["return_baselines"]
    code = benchmark_result["comparison"]["result_code"]
    result = {
        "replay_id": snapshot["replay_id"],
        "trading_date": snapshot["trading_date"],
        "scout_version": scout_result["scout_version"],
        "trainer_version": "scout_trainer_v1.1",
        "execution_policy_version": benchmark_result["execution_policy_version"],
        "massive_plan": snapshot.get("massive_plan", load_massive_plan()),
        **universe_metadata,
        "result": {
            "SCOUT_OUTPERFORMED": "WIN",
            "SCOUT_TIED": "TIE",
            "SCOUT_UNDERPERFORMED": "MISS",
        }[code],
        "scout_performance": {
            key: benchmark_result["scout_summary"].get(key)
            for key in (
                "realized_return_pct",
                "realized_pnl_usd",
                "max_drawdown_pct",
                "win_rate_pct",
                "average_capture_ratio",
            )
        },
        "benchmark_performance": {
            "realized_return_pct": baselines[
                "random_draw_mean_realized_return_pct"
            ],
            "realized_pnl_usd": baselines[
                "random_draw_mean_realized_pnl_usd"
            ],
            "max_drawdown_pct": None,
            "win_rate_pct": None,
            "average_capture_ratio": None,
        },
        "unreachable_mover_count": len(unreachable),
        "unreachable_pct": (
            len(unreachable) / top_10_count * 100 if top_10_count else 0.0
        ),
        "missed_opportunities": missed,
        "execution_policy_review": execution_review,
        "feature_failures": list(failures.values()),
        "feature_proposals": [],
        "notes": [
            "WIN/TIE/MISS uses the deterministic random-draw return baseline, not top-10 capture.",
            "Top-10 capture is diagnostic only.",
            "No feature proposal is eligible from one day; at least 30 independent occurrences are required.",
        ],
    }
    try:
        validate_contract("postmortem", result)
    except ContractError as exc:
        raise PostmortemError(
            f"Postmortem contract validation failed: {exc}"
        ) from exc
    return result
