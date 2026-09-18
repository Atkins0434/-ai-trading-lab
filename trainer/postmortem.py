from __future__ import annotations

from datetime import datetime
from collections import Counter
from pathlib import Path
from statistics import mean
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
    *,
    outcome_result: dict[str, Any] | None = None,
    bar_statistics: dict[str, Any] | None = None,
    scorability_statistics: dict[str, Any] | None = None,
    strategy_capital_usd: float = 2500.0,
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
    if outcome_result is None:
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
        _, execution_verdict = _execution_policy_review(
            None, benchmark_result, strategy_capital_usd
        )
    else:
        execution_review, execution_verdict = _execution_policy_review(
            outcome_result,
            benchmark_result,
            strategy_capital_usd,
        )
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
        "universe_scorability": _universe_scorability(
            bar_statistics,
            scorability_statistics,
            len(unreachable),
        ),
        "missed_opportunities": missed,
        "execution_policy_review": execution_review,
        "execution_policy_verdict": execution_verdict,
        "feature_failures": list(failures.values()),
        "feature_proposals": [],
        "notes": [
            "WIN/TIE/MISS uses the deterministic random-draw return baseline, not top-10 capture.",
            "Top-10 capture is diagnostic only.",
            "No feature proposal is eligible from one day; at least 30 independent occurrences are required.",
            "Execution policy review is descriptive. Promotion requires the gates defined in the execution learner, not a single-day verdict.",
        ],
    }
    try:
        validate_contract("postmortem", result)
    except ContractError as exc:
        raise PostmortemError(
            f"Postmortem contract validation failed: {exc}"
        ) from exc
    return result


def _execution_summary(
    executions: list[dict[str, Any]],
    strategy_capital_usd: float,
) -> dict[str, Any]:
    completed = [item for item in executions if item.get("trade_executed")]
    returns = [float(item["realized_return_pct"]) for item in completed]
    captures = [
        float(item["capture_ratio"])
        for item in completed
        if item.get("capture_ratio") is not None
    ]
    drawdowns = [
        float(item["maximum_position_drawdown_pct"])
        for item in completed
        if item.get("maximum_position_drawdown_pct") is not None
    ]
    reasons = Counter(
        item.get("exit_reason") for item in executions
        if item.get("exit_reason") in {
            "TRAILING_STOP", "PROFIT_TARGET", "SESSION_END"
        }
    )
    rejected = Counter(
        item.get("entry_rejection_reason") or "UNSPECIFIED"
        for item in executions
        if item.get("exit_reason") == "ENTRY_REJECTED"
    )
    net_pnl = sum(float(item.get("realized_pnl_usd") or 0.0) for item in completed)
    return {
        "trades_executed": len(completed),
        "realized_return_pct": mean(returns) if returns else 0.0,
        "net_realized_pnl_usd": net_pnl,
        "average_capture_ratio": mean(captures) if captures else None,
        "win_rate_pct": (
            sum(value > 0 for value in returns) / len(returns) * 100
            if returns else None
        ),
        "max_drawdown_pct": min(drawdowns) if drawdowns else None,
        "exit_reason_counts": {
            "TRAILING_STOP": reasons["TRAILING_STOP"],
            "PROFIT_TARGET": reasons["PROFIT_TARGET"],
            "SESSION_END": reasons["SESSION_END"],
            "ENTRY_REJECTED": dict(sorted(rejected.items())),
        },
    }


def _execution_policy_review(
    outcome_result: dict[str, Any] | None,
    benchmark_result: dict[str, Any],
    strategy_capital_usd: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if outcome_result is None:
        return [], {
            "champion_policy_id": "execution_policy_v1.0",
            "best_challenger_policy_id": None,
            "challenger_won_on_selection": False,
            "challenger_won_on_movers": False,
            "sample_size": {"SCOUT_SELECTION": 0, "TOP_10_MOVER": 0},
        }

    outcomes = outcome_result.get("outcomes", [])
    selected = [
        item["execution_result"] for item in outcomes if item.get("selected")
    ]
    mover_candidates = benchmark_result.get("benchmark_candidates", [])
    mover_tickers = [item["ticker"] for item in mover_candidates]
    champion_movers = [
        {
            "trade_executed": bool(item.get("trade_executed")),
            "realized_return_pct": item.get("realized_return_pct") or 0.0,
            "realized_pnl_usd": item.get("realized_pnl_usd") or 0.0,
            "capture_ratio": item.get("capture_ratio"),
            "maximum_position_drawdown_pct": item.get(
                "maximum_position_drawdown_pct"
            ),
            "exit_reason": item.get("exit_reason"),
        }
        for item in mover_candidates
    ]
    sample = {
        "SCOUT_SELECTION": len(selected),
        "TOP_10_MOVER": len(mover_tickers),
    }
    if selected:
        champion_policy_id = selected[0]["policy_id"]
        exit_mode = selected[0]["exit_mode"]
        sizing_mode = selected[0]["sizing_mode"]
    else:
        from trainer.trade_engine import load_execution_policy
        policy = load_execution_policy()
        champion_policy_id = policy.policy_id
        exit_mode = policy.exit_mode
        sizing_mode = policy.sizing_mode
    champion_summaries = {
        "SCOUT_SELECTION": _execution_summary(selected, strategy_capital_usd),
        "TOP_10_MOVER": _execution_summary(
            champion_movers, strategy_capital_usd
        ),
    }
    champion = {
        "policy_id": champion_policy_id,
        "exit_mode": exit_mode,
        "sizing_mode": sizing_mode,
        "cohort_summaries": champion_summaries,
    }
    for summary in champion_summaries.values():
        summary["delta_vs_champion"] = {
            "return_pct": None,
            "capture_ratio": None,
        }
    reviews = [champion]

    for comparison in outcome_result.get("policy_comparisons", []):
        cohorts: dict[str, list[dict[str, Any]]] = {
            "SCOUT_SELECTION": [],
            "TOP_10_MOVER": [],
        }
        for item in comparison.get("executions", []):
            cohorts[item["cohort"]].append(item["execution_result"])
        summaries = {
            name: _execution_summary(values, strategy_capital_usd)
            for name, values in cohorts.items()
        }
        for name, summary in summaries.items():
            champion_summary = champion_summaries[name]
            champion_capture = champion_summary["average_capture_ratio"]
            challenger_capture = summary["average_capture_ratio"]
            summary["delta_vs_champion"] = {
                "return_pct": (
                    summary["realized_return_pct"]
                    - champion_summary["realized_return_pct"]
                ),
                "capture_ratio": (
                    challenger_capture - champion_capture
                    if challenger_capture is not None
                    and champion_capture is not None
                    else None
                ),
            }
        reviews.append({
            "policy_id": comparison["policy_id"],
            "exit_mode": comparison["exit_mode"],
            "sizing_mode": comparison["sizing_mode"],
            "cohort_summaries": summaries,
        })

    challengers = reviews[1:]
    best = max(
        challengers,
        key=lambda item: item["cohort_summaries"]["SCOUT_SELECTION"][
            "realized_return_pct"
        ],
        default=None,
    )
    return reviews, {
        "champion_policy_id": champion_policy_id,
        "best_challenger_policy_id": best["policy_id"] if best else None,
        "challenger_won_on_selection": bool(
            best
            and best["cohort_summaries"]["SCOUT_SELECTION"]["realized_return_pct"]
            > champion_summaries["SCOUT_SELECTION"]["realized_return_pct"]
        ),
        "challenger_won_on_movers": bool(
            best
            and best["cohort_summaries"]["TOP_10_MOVER"]["realized_return_pct"]
            > champion_summaries["TOP_10_MOVER"]["realized_return_pct"]
        ),
        "sample_size": sample,
    }


def _universe_scorability(
    bar_statistics: dict[str, Any] | None,
    scorability_statistics: dict[str, Any] | None,
    top_10_invisible_count: int,
) -> dict[str, Any]:
    bars = bar_statistics or {}
    scorability = scorability_statistics or {}
    by_ticker = bars.get("by_ticker", {})
    eligible = int(scorability.get("universe_ticker_count", len(by_ticker)))
    scorable = int(scorability.get("scorable_ticker_count", eligible))
    return {
        "eligible_count": eligible,
        "scorable_count": scorable,
        "not_scorable_share": float(
            scorability.get(
                "not_scorable_share",
                (eligible - scorable) / eligible if eligible else 0.0,
            )
        ),
        "zero_premarket_bar_count": sum(
            int(value.get("real_premarket", 0)) == 0
            for value in by_ticker.values()
        ),
        "zero_premarket_60m_bar_count": sum(
            int(value.get("real_premarket_60m", 0)) == 0
            for value in by_ticker.values()
        ),
        "top_10_invisible_count": top_10_invisible_count,
    }
