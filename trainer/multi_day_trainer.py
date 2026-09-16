from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import json
import math
from pathlib import Path
from statistics import pstdev
from typing import Any, Callable

from trainer.dataset_splits import (
    build_chronological_splits,
    enforce_partition_access,
)
from trainer.replay_queue import ReplayQueue
from trainer.research_alpha_batch import run_massive_alpha_batch
from trainer.quarantine_legacy_results import quarantine_legacy_results
from trainer.trainer_summary_report import generate_trainer_summary_pdf
from trainer.validate_contracts import validate_contract
from trainer.evidence_eligibility import (
    EvidenceEligibilityError,
    assert_matching_universe,
    require_research_evidence,
)
from trainer.universe_manifest import CI_FIXTURE, HISTORICAL_RESEARCH
from trainer.rate_control import load_massive_plan


MINIMUM_OCCURRENCES = 30
STATE_VERSION = "multi_day_trainer_run_v2.2"
HYPOTHESIS_ELIGIBLE_MISSES = {
    "VISIBLE_SCORED_LOW",
    "VISIBLE_GUARDRAIL_REJECT",
}
DayRunner = Callable[..., dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _preserve_superseded(path: Path) -> None:
    if not path.exists():
        return
    suffix = 0
    while True:
        marker = "" if suffix == 0 else f".{suffix}"
        target = path.with_name(f"{path.stem}.superseded{marker}{path.suffix}")
        if not target.exists():
            path.replace(target)
            return
        suffix += 1


def _validate_dates(trading_dates: list[str]) -> list[str]:
    if not trading_dates:
        raise ValueError("At least one historical trading date is required.")
    parsed = []
    for value in sorted(set(trading_dates)):
        parsed_date = date.fromisoformat(value)
        if parsed_date.weekday() >= 5:
            raise ValueError(f"Weekend dates are not valid trading sessions: {value}")
        parsed.append(value)
    return parsed


def _hypothesis_key(classification: str) -> tuple[str, str, str]:
    if classification == "VISIBLE_GUARDRAIL_REJECT":
        return ("aggregate_liquidity", "GUARDRAIL_REVIEW", "Review whether the research liquidity guardrail excludes repeatable tradable movers.")
    if classification == "VISIBLE_SCORED_LOW":
        return ("selection_threshold", "THRESHOLD_REVIEW", "Review whether the Alpha selection threshold suppresses repeatable tradable movers.")
    raise ValueError(
        f"Miss classification cannot generate a Scout hypothesis: {classification}"
    )


def _sortable_raw_value(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    if value is None:
        return (2, "")
    return (1, json.dumps(value, sort_keys=True, separators=(",", ":")))


def _raw_value_range(values: list[Any]) -> tuple[Any, Any]:
    if not values:
        return None, None
    ordered = sorted(values, key=_sortable_raw_value)
    return ordered[0], ordered[-1]


def _compounded_return(returns: list[float]) -> float:
    factor = math.prod(1 + value / 100 for value in returns)
    return round((factor - 1) * 100, 6)


def _maximum_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    maximum = 0.0
    for value in returns:
        equity *= 1 + value / 100
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak * 100)
    return round(maximum, 6)


def _aggregate(
    day_records: list[dict[str, Any]], root: Path
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    scout_returns: list[float] = []
    exploration_returns: list[float] = []
    combined_returns: list[float] = []
    benchmark_returns: list[float] = []
    scout_pnl = 0.0
    exploration_pnl = 0.0
    combined_pnl = 0.0
    benchmark_pnl = 0.0
    captures: list[float] = []
    results = defaultdict(int)
    evidence: dict[str, dict[str, Any]] = {}
    feature_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    low_score_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    execution_policy_review: list[dict[str, Any]] = []
    unreachable_count = 0
    top_mover_count = 0

    for record in sorted(day_records, key=lambda item: item["trading_date"]):
        day_dir = root / record["artifact_directory"]
        partition = record.get("partition", "DEVELOPMENT")
        benchmark = _read_json(day_dir / "benchmark_result.json")
        postmortem = _read_json(day_dir / "postmortem.json")
        try:
            assert_matching_universe(benchmark, postmortem)
            require_research_evidence(postmortem, consumer="Trainer")
        except EvidenceEligibilityError as exc:
            raise ValueError(str(exc)) from exc
        scout_summary = benchmark["scout_summary"]
        exploration_summary = benchmark["exploration_summary"]
        combined_summary = benchmark["combined_summary"]
        baselines = benchmark["return_baselines"]
        scout_returns.append(float(scout_summary["realized_return_pct"]))
        exploration_returns.append(
            float(exploration_summary["realized_return_pct"])
        )
        combined_returns.append(
            float(combined_summary["realized_return_pct"])
        )
        benchmark_returns.append(
            float(baselines["random_draw_mean_realized_return_pct"])
        )
        scout_pnl += float(scout_summary["realized_pnl_usd"])
        exploration_pnl += float(
            exploration_summary["realized_pnl_usd"]
        )
        combined_pnl += float(combined_summary["realized_pnl_usd"])
        benchmark_pnl += float(
            baselines["random_draw_mean_realized_pnl_usd"]
        )
        captures.append(float(benchmark["comparison"]["top_10_capture_rate_pct"]))
        results[postmortem["result"]] += 1
        unreachable_count += int(postmortem["unreachable_mover_count"])
        top_mover_count += len(benchmark["benchmark_candidates"])

        for review in postmortem["execution_policy_review"]:
            execution_policy_review.append({
                "trading_date": postmortem["trading_date"],
                "partition": partition,
                **review,
            })

        # Only development misses may create or strengthen a hypothesis.
        # Validation and holdout dates measure frozen hypotheses; allowing them
        # to train the proposal would leak out-of-sample evidence.
        for miss in (
            postmortem["missed_opportunities"]
            if partition == "DEVELOPMENT"
            else []
        ):
            classification = miss["miss_classification"]
            if classification == "VISIBLE_SCORED_LOW":
                for component in miss["component_scores"]:
                    if component["score"] not in {0, 1}:
                        continue
                    low_score_metrics[component["metric_id"]].append({
                        "trading_date": postmortem["trading_date"],
                        "ticker": miss["ticker"],
                        "score": component["score"],
                        "raw_value": component["raw_value"],
                    })
            if classification not in HYPOTHESIS_ELIGIBLE_MISSES:
                continue
            feature_id, proposal_type, hypothesis = _hypothesis_key(
                classification
            )
            hypothesis_id = f"{feature_id}:{proposal_type}"
            item = evidence.setdefault(hypothesis_id, {
                "hypothesis_id": hypothesis_id,
                "target_feature_id": feature_id,
                "proposal_type": proposal_type,
                "hypothesis": hypothesis,
                "independent_occurrence_count": 0,
                "minimum_required_occurrences": MINIMUM_OCCURRENCES,
                "evidence": [],
            })
            occurrence = {
                "trading_date": postmortem["trading_date"],
                "ticker": miss["ticker"],
                "partition": partition,
                "benchmark_rank": miss["benchmark_rank"],
                "miss_classification": classification,
                "failure_reason_codes": miss.get("failure_reason_codes", []),
            }
            occurrence_key = (occurrence["trading_date"], occurrence["ticker"])
            known = {(entry["trading_date"], entry["ticker"]) for entry in item["evidence"]}
            if occurrence_key not in known:
                item["evidence"].append(occurrence)

        catalyst_path = day_dir / "catalyst_shadow_metrics.json"
        if catalyst_path.exists():
            catalyst_payload = _read_json(catalyst_path)
            for ticker_result in catalyst_payload.get("ticker_metrics", []):
                for component in ticker_result.get("components", []):
                    feature_rows[component["metric"]].append({
                        "trading_date": record["trading_date"],
                        "ticker": ticker_result["ticker"],
                        "partition": partition,
                        "status": component["status"],
                        "points": component["points"],
                    })

    hypotheses = []
    for item in evidence.values():
        item["evidence"].sort(key=lambda entry: (entry["trading_date"], entry["ticker"]))
        item["independent_occurrence_count"] = len(item["evidence"])
        item["evidence_requirement_met"] = item["independent_occurrence_count"] >= MINIMUM_OCCURRENCES
        item["status"] = "READY_FOR_VALIDATION" if item["evidence_requirement_met"] else "COLLECTING_EVIDENCE"
        item["production_mutation_allowed"] = False
        hypotheses.append(item)
    hypotheses.sort(key=lambda item: (-item["independent_occurrence_count"], item["hypothesis_id"]))

    candidate_threshold_reviews = []
    for metric_id, rows in low_score_metrics.items():
        raw_min, raw_max = _raw_value_range(
            [row["raw_value"] for row in rows]
        )
        candidate_threshold_reviews.append({
            "metric_id": metric_id,
            "low_score_frequency": len(rows),
            "missed_ticker_count": len({
                (row["trading_date"], row["ticker"]) for row in rows
            }),
            "observed_raw_value_min": raw_min,
            "observed_raw_value_max": raw_max,
        })
    candidate_threshold_reviews.sort(
        key=lambda item: (-item["low_score_frequency"], item["metric_id"])
    )
    candidate_threshold_reviews = candidate_threshold_reviews[:3]
    execution_policy_review.sort(
        key=lambda item: (
            item["trading_date"], item["benchmark_rank"], item["ticker"]
        )
    )

    feature_evidence = []
    for metric_id, rows in sorted(feature_rows.items()):
        unique = {
            (row["trading_date"], row["ticker"]): row for row in rows
        }
        ordered_rows = [unique[key] for key in sorted(unique)]
        development_rows = [
            row for row in ordered_rows if row["partition"] == "DEVELOPMENT"
        ]
        observed_points = [
            float(row["points"])
            for row in development_rows
            if row["status"] == "OBSERVED" and row["points"] is not None
        ]
        count = len(development_rows)
        feature_evidence.append({
            "metric_id": metric_id,
            "independent_occurrence_count": count,
            "validation_occurrence_count": sum(
                row["partition"] == "VALIDATION" for row in ordered_rows
            ),
            "holdout_occurrence_count": sum(
                row["partition"] == "HOLDOUT" for row in ordered_rows
            ),
            "observed_count": len(observed_points),
            "positive_count": sum(value > 0 for value in observed_points),
            "average_points": (
                round(sum(observed_points) / len(observed_points), 6)
                if observed_points else None
            ),
            "minimum_required_occurrences": MINIMUM_OCCURRENCES,
            "status": (
                "READY_FOR_VALIDATION"
                if count >= MINIMUM_OCCURRENCES
                else "COLLECTING_EVIDENCE"
            ),
            "production_mutation_allowed": False,
            "evidence": ordered_rows,
        })

    count = len(day_records)
    aggregate = {
        "days_processed": count,
        "scout_wins": results["WIN"],
        "ties": results["TIE"],
        "misses": results["MISS"],
        "scout_total_realized_pnl_usd": round(scout_pnl, 4),
        "exploration_total_realized_pnl_usd": round(exploration_pnl, 4),
        "combined_total_realized_pnl_usd": round(combined_pnl, 4),
        "benchmark_total_realized_pnl_usd": round(benchmark_pnl, 4),
        "scout_average_daily_return_pct": round(sum(scout_returns) / count, 6) if count else 0.0,
        "exploration_average_daily_return_pct": round(sum(exploration_returns) / count, 6) if count else 0.0,
        "combined_average_daily_return_pct": round(sum(combined_returns) / count, 6) if count else 0.0,
        "benchmark_average_daily_return_pct": round(sum(benchmark_returns) / count, 6) if count else 0.0,
        "scout_cumulative_return_pct": _compounded_return(scout_returns),
        "exploration_cumulative_return_pct": _compounded_return(exploration_returns),
        "combined_cumulative_return_pct": _compounded_return(combined_returns),
        "benchmark_cumulative_return_pct": _compounded_return(benchmark_returns),
        "scout_max_drawdown_pct": _maximum_drawdown(scout_returns),
        "exploration_max_drawdown_pct": _maximum_drawdown(exploration_returns),
        "combined_max_drawdown_pct": _maximum_drawdown(combined_returns),
        "benchmark_max_drawdown_pct": _maximum_drawdown(benchmark_returns),
        "scout_daily_return_stddev_pct": round(pstdev(scout_returns), 6) if len(scout_returns) > 1 else 0.0,
        "exploration_daily_return_stddev_pct": round(pstdev(exploration_returns), 6) if len(exploration_returns) > 1 else 0.0,
        "combined_daily_return_stddev_pct": round(pstdev(combined_returns), 6) if len(combined_returns) > 1 else 0.0,
        "benchmark_daily_return_stddev_pct": round(pstdev(benchmark_returns), 6) if len(benchmark_returns) > 1 else 0.0,
        "scout_positive_day_rate_pct": round(sum(value > 0 for value in scout_returns) / count * 100, 6) if count else 0.0,
        "exploration_positive_day_rate_pct": round(sum(value > 0 for value in exploration_returns) / count * 100, 6) if count else 0.0,
        "combined_positive_day_rate_pct": round(sum(value > 0 for value in combined_returns) / count * 100, 6) if count else 0.0,
        "benchmark_positive_day_rate_pct": round(sum(value > 0 for value in benchmark_returns) / count * 100, 6) if count else 0.0,
        "realized_pnl_capture_pct": round(scout_pnl / benchmark_pnl * 100, 6) if benchmark_pnl > 0 else None,
        "average_top_10_capture_rate_pct": round(sum(captures) / count, 6) if count else 0.0,
        "unreachable_mover_count": unreachable_count,
        "unreachable_pct": (
            round(unreachable_count / top_mover_count * 100, 6)
            if top_mover_count else 0.0
        ),
        "execution_policy_review_count": len(execution_policy_review),
    }
    return (
        aggregate,
        hypotheses,
        feature_evidence,
        candidate_threshold_reviews,
        execution_policy_review,
    )


def _build_state(
    dates,
    tickers,
    days,
    failures,
    aggregate,
    hypotheses,
    feature_evidence,
    candidate_threshold_reviews,
    execution_policy_review,
    threshold_pct,
    exploration_top_k,
    dataset_split,
    allowed_partitions,
    holdout_unlocked,
    queue_snapshot,
    max_workers,
    universe_mode,
) -> dict[str, Any]:
    ordered_days = [days[value] for value in dates if value in days]
    completed = [item["trading_date"] for item in ordered_days if item["status"] in {"COMPLETE", "PARTIAL"}]
    evidence_days = [
        item
        for item in ordered_days
        if item["status"] in {"COMPLETE", "PARTIAL"}
        and item.get("research_evidence") is True
    ]
    if not completed and any(item["status"] == "UNSUPPORTED" for item in ordered_days):
        status = "UNSUPPORTED"
    elif not completed:
        status = "FAILED"
    elif failures or any(item["status"] == "PARTIAL" for item in ordered_days):
        status = "PARTIAL"
    else:
        status = "COMPLETE"
    return {
        "version": STATE_VERSION,
        "run_id": f"trainer-{dates[0]}-to-{dates[-1]}",
        "mode": "RESEARCH_ONLY",
        "universe_mode": universe_mode,
        "research_evidence": universe_mode == HISTORICAL_RESEARCH and bool(evidence_days),
        "promotion_eligible": (
            universe_mode == HISTORICAL_RESEARCH
            and bool(evidence_days)
            and not any(
                item.get("research_evidence") is not True
                for item in ordered_days
                if item["status"] in {"COMPLETE", "PARTIAL"}
            )
            and all(item.get("promotion_eligible") is True for item in evidence_days)
        ),
        "trainer_version": "scout_trainer_v1.2",
        "source": "MASSIVE",
        "massive_plan": load_massive_plan(),
        "status": status,
        "requested_dates": dates,
        "requested_tickers": tickers,
        "selection_policy": {
            "scoring_threshold_pct_override": threshold_pct,
            "exploration_top_k": exploration_top_k,
        },
        "dataset_policy": {
            "split_method": "CHRONOLOGICAL_60_20_20",
            "date_partitions": dataset_split,
            "allowed_partitions": sorted(allowed_partitions),
            "holdout_unlocked": holdout_unlocked,
            "locked_holdout_dates": sorted(
                value
                for value, partition in dataset_split.items()
                if partition == "HOLDOUT" and not holdout_unlocked
            ),
        },
        "completed_dates": completed,
        "failed_dates": failures,
        "days": ordered_days,
        "quarantined_days": [
            item["trading_date"]
            for item in ordered_days
            if item.get("research_evidence") is not True
        ],
        "aggregate_performance": aggregate,
        "hypotheses": hypotheses,
        "candidate_threshold_reviews": candidate_threshold_reviews,
        "execution_policy_review": execution_policy_review,
        "feature_evidence": feature_evidence,
        "queue": queue_snapshot,
        "accelerator": {
            "max_workers": max_workers,
            "worker_scope": "TRADING_DATE",
            "date_queue_resumable": True,
            "ticker_resume_via_checkpoint_and_cache": True,
            "adaptive_rate_limiting": True,
            "bounded_retry_handling": True,
        },
        "controls": {
            "real_money_execution_allowed": False,
            "production_mutation_allowed": False,
            "automatic_promotion_allowed": False,
            "minimum_independent_occurrences": MINIMUM_OCCURRENCES,
            "out_of_sample_validation_required": True,
            "blind_holdout_required": True,
            "manual_promotion_required": True,
        },
        "artifacts": {
            "state": "trainer_run_state.json",
            "queue": "replay_queue.json",
            "summary_report": "trainer_summary_report.pdf",
        },
    }


def run_multi_day_trainer(
    client: Any,
    tickers: list[str] | None,
    trading_dates: list[str],
    *,
    cache_root: Path,
    output_root: Path,
    threshold_pct: float | None = None,
    exploration_top_k: int = 0,
    resume: bool = True,
    day_runner: DayRunner = run_massive_alpha_batch,
    max_workers: int = 1,
    allowed_partitions: tuple[str, ...] | list[str] = (
        "DEVELOPMENT",
        "VALIDATION",
    ),
    holdout_unlocked: bool = False,
    universe_mode: str = CI_FIXTURE,
) -> dict[str, Any]:
    """Run queued historical days with bounded workers and atomic checkpoints."""
    requested_dates = _validate_dates(trading_dates)
    requested_tickers = sorted({ticker.upper() for ticker in (tickers or [])})
    if universe_mode == CI_FIXTURE and not requested_tickers:
        raise ValueError("ci_fixture mode requires at least one ticker.")
    if universe_mode == HISTORICAL_RESEARCH and requested_tickers:
        raise ValueError("Hardcoded tickers are forbidden in historical_research mode.")
    if universe_mode not in {CI_FIXTURE, HISTORICAL_RESEARCH}:
        raise ValueError(f"Unknown universe mode: {universe_mode}")
    if not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be between 1 and 8.")

    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "trainer_run_state.json"
    prior = _read_json(state_path) if resume and state_path.exists() else {}
    active_plan = load_massive_plan()
    expected_policy = {
        "scoring_threshold_pct_override": threshold_pct,
        "exploration_top_k": exploration_top_k,
    }
    reset_queue = not resume
    if prior and (
        prior.get("version") != STATE_VERSION
        or prior.get("selection_policy") != expected_policy
        or prior.get("requested_tickers") != requested_tickers
        or prior.get("universe_mode") != universe_mode
        or prior.get("massive_plan") != active_plan
    ):
        quarantine_legacy_results(output_root)
        _preserve_superseded(state_path)
        prior = {}
        reset_queue = True
    dates = _validate_dates(prior.get("requested_dates", []) + requested_dates) if prior else requested_dates
    dataset_split = build_chronological_splits(dates)
    prior_split = prior.get("dataset_policy", {}).get("date_partitions", {})
    changed_partitions = {
        value: (partition, dataset_split.get(value))
        for value, partition in prior_split.items()
        if dataset_split.get(value) != partition
    }
    if changed_partitions:
        raise ValueError(
            "Date-range expansion would change frozen development/validation/"
            "holdout assignments; use a new output root for a new experiment."
        )
    runnable_dates = enforce_partition_access(
        dataset_split,
        allowed_partitions,
        holdout_unlocked=holdout_unlocked,
    )
    prior_days = {item["trading_date"]: item for item in prior.get("days", [])}
    days: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {
        key: value for key, value in prior.get("failed_dates", {}).items() if key in dates
    }

    queue_path = output_root / "replay_queue.json"
    if reset_queue:
        _preserve_superseded(queue_path)
    queue = ReplayQueue(queue_path)
    queue.enqueue(
        (
            trading_date,
            None,
            "DAY_REPLAY",
            dataset_split[trading_date],
        )
        for trading_date in runnable_dates
    )

    for trading_date in runnable_dates:
        previous = prior_days.get(trading_date)
        day_dir = output_root / "days" / trading_date
        reusable_artifact_exists = (
            (day_dir / "postmortem.json").exists()
            if previous and previous.get("research_evidence") is True
            else (day_dir / "research_alpha_batch_manifest.json").exists()
        )
        if (
            previous
            and previous["status"] == "COMPLETE"
            and reusable_artifact_exists
        ):
            previous = dict(previous)
            previous["partition"] = dataset_split[trading_date]
            days[trading_date] = previous
            queue.complete(
                ReplayQueue.task_id(trading_date, None, "DAY_REPLAY")
            )
        elif previous and previous["status"] == "PARTIAL":
            # A partial day may contain completed ticker checkpoints alongside
            # retryable ticker failures. Reopen the parent so the day runner
            # can reuse completed children and claim the remaining ticker work.
            queue.reopen(
                ReplayQueue.task_id(trading_date, None, "DAY_REPLAY"),
                "PARTIAL_DAY_REQUIRES_TICKER_RESUME",
            )

    def run_task(task: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        task_id = task["task_id"]
        claimed = queue.claim(task_id)
        if claimed is None:
            return task["trading_date"], None, None
        trading_date = task["trading_date"]
        day_dir = output_root / "days" / trading_date
        try:
            manifest = day_runner(
                client,
                requested_tickers,
                trading_date,
                cache_root=cache_root,
                output_dir=day_dir,
                threshold_pct=threshold_pct,
                exploration_top_k=exploration_top_k,
                dataset_partition=dataset_split[trading_date],
                universe_mode=universe_mode,
            )
            postmortem_path = day_dir / "postmortem.json"
            postmortem = (
                _read_json(postmortem_path)
                if manifest.get("research_evidence") is True
                and postmortem_path.exists()
                else {}
            )
            record = {
                "trading_date": trading_date,
                "partition": dataset_split[trading_date],
                "status": manifest["status"],
                "artifact_directory": f"days/{trading_date}",
                "scored_ticker_count": len(manifest["scored_tickers"]),
                "skipped_ticker_count": len(manifest["skipped"]),
                "eligible_symbol_count": len(
                    manifest.get("universe_eligible_tickers", manifest["scored_tickers"])
                ),
                "universe_mode": manifest["universe_mode"],
                "massive_plan": manifest.get("massive_plan", active_plan),
                "universe_manifest_hash": manifest["universe_manifest_hash"],
                "universe_coverage": manifest["universe_coverage"],
                "research_evidence": manifest["research_evidence"],
                "promotion_eligible": manifest["promotion_eligible"],
                "unreachable_mover_count": int(
                    postmortem.get("unreachable_mover_count", 0)
                ),
                "unreachable_pct": float(
                    postmortem.get("unreachable_pct", 0.0)
                ),
                "execution_policy_review_count": len(
                    postmortem.get("execution_policy_review", [])
                ),
            }
            if manifest["status"] == "COMPLETE":
                queue.complete(task_id)
            elif manifest["status"] == "UNSUPPORTED":
                queue.fail(
                    task_id,
                    "POINT_IN_TIME_UNIVERSE_UNSUPPORTED",
                    retryable=False,
                )
            else:
                queue.fail(
                    task_id,
                    "PARTIAL_DAY_REQUIRES_TICKER_RESUME",
                    retryable=True,
                )
            return trading_date, record, None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            queue.fail(task_id, error, retryable=True)
            return trading_date, {
                "trading_date": trading_date,
                "partition": dataset_split[trading_date],
                "status": "FAILED",
                "artifact_directory": f"days/{trading_date}",
                "scored_ticker_count": 0,
                "skipped_ticker_count": len(requested_tickers),
                "eligible_symbol_count": 0,
                "universe_mode": universe_mode,
                "massive_plan": active_plan,
                "universe_manifest_hash": None,
                "universe_coverage": "incomplete" if universe_mode == HISTORICAL_RESEARCH else "fixture",
                "research_evidence": False,
                "promotion_eligible": False,
                "unreachable_mover_count": 0,
                "unreachable_pct": 0.0,
                "execution_policy_review_count": 0,
            }, error

    def checkpoint() -> dict[str, Any]:
        successful = [
            item for item in days.values()
            if item["status"] in {"COMPLETE", "PARTIAL"}
            and item.get("research_evidence") is True
        ]
        (
            aggregate,
            hypotheses,
            feature_evidence,
            candidate_threshold_reviews,
            execution_policy_review,
        ) = _aggregate(successful, output_root)
        state = _build_state(
            dates,
            requested_tickers,
            days,
            failures,
            aggregate,
            hypotheses,
            feature_evidence,
            candidate_threshold_reviews,
            execution_policy_review,
            threshold_pct,
            exploration_top_k,
            dataset_split,
            allowed_partitions,
            holdout_unlocked,
            queue.snapshot(),
            max_workers,
            universe_mode,
        )
        _write_json(state_path, state)
        return state

    pending = queue.pending(stage="DAY_REPLAY")
    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_task, task) for task in pending]
            for future in as_completed(futures):
                trading_date, record, error = future.result()
                if record is not None:
                    days[trading_date] = record
                if error is None:
                    failures.pop(trading_date, None)
                else:
                    failures[trading_date] = error
                checkpoint()

    state = checkpoint()
    validate_contract("multi_day_trainer_run", state)
    _write_json(state_path, state)
    quarantine_legacy_results(output_root)
    generate_trainer_summary_pdf(state, output_root / "trainer_summary_report.pdf")
    return state
