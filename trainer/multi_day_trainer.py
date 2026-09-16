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
from trainer.trainer_summary_report import generate_trainer_summary_pdf
from trainer.validate_contracts import validate_contract


MINIMUM_OCCURRENCES = 30
DayRunner = Callable[..., dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def _hypothesis_key(stage: str) -> tuple[str, str, str]:
    if stage == "HARD_GUARDRAIL":
        return ("aggregate_liquidity", "GUARDRAIL_REVIEW", "Review whether the research liquidity guardrail excludes repeatable tradable movers.")
    if stage == "BELOW_SELECTION_THRESHOLD":
        return ("selection_threshold", "THRESHOLD_REVIEW", "Review whether the Alpha selection threshold suppresses repeatable tradable movers.")
    if stage == "MISSING_DATA":
        return ("data_completeness", "DATA_QUALITY_REVIEW", "Review repeatable provider gaps before treating missing observations as market evidence.")
    return (stage.lower(), "PIPELINE_REVIEW", f"Review repeatable misses at the {stage} stage.")


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
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    scout_returns: list[float] = []
    benchmark_returns: list[float] = []
    scout_pnl = 0.0
    benchmark_pnl = 0.0
    captures: list[float] = []
    results = defaultdict(int)
    evidence: dict[str, dict[str, Any]] = {}
    feature_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in sorted(day_records, key=lambda item: item["trading_date"]):
        day_dir = root / record["artifact_directory"]
        partition = record.get("partition", "DEVELOPMENT")
        benchmark = _read_json(day_dir / "benchmark_result.json")
        postmortem = _read_json(day_dir / "postmortem.json")
        scout_summary = benchmark["scout_summary"]
        benchmark_summary = benchmark["benchmark_summary"]
        scout_returns.append(float(scout_summary["realized_return_pct"]))
        benchmark_returns.append(float(benchmark_summary["realized_return_pct"]))
        scout_pnl += float(scout_summary["realized_pnl_usd"])
        benchmark_pnl += float(benchmark_summary["realized_pnl_usd"])
        captures.append(float(benchmark["comparison"]["top_10_capture_rate_pct"]))
        results[postmortem["result"]] += 1

        # Only development misses may create or strengthen a hypothesis.
        # Validation and holdout dates measure frozen hypotheses; allowing them
        # to train the proposal would leak out-of-sample evidence.
        for miss in (
            postmortem["missed_opportunities"]
            if partition == "DEVELOPMENT"
            else []
        ):
            feature_id, proposal_type, hypothesis = _hypothesis_key(miss["failure_stage"])
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
                "failure_stage": miss["failure_stage"],
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
        "benchmark_total_realized_pnl_usd": round(benchmark_pnl, 4),
        "scout_average_daily_return_pct": round(sum(scout_returns) / count, 6) if count else 0.0,
        "benchmark_average_daily_return_pct": round(sum(benchmark_returns) / count, 6) if count else 0.0,
        "scout_cumulative_return_pct": _compounded_return(scout_returns),
        "benchmark_cumulative_return_pct": _compounded_return(benchmark_returns),
        "scout_max_drawdown_pct": _maximum_drawdown(scout_returns),
        "benchmark_max_drawdown_pct": _maximum_drawdown(benchmark_returns),
        "scout_daily_return_stddev_pct": round(pstdev(scout_returns), 6) if len(scout_returns) > 1 else 0.0,
        "benchmark_daily_return_stddev_pct": round(pstdev(benchmark_returns), 6) if len(benchmark_returns) > 1 else 0.0,
        "scout_positive_day_rate_pct": round(sum(value > 0 for value in scout_returns) / count * 100, 6) if count else 0.0,
        "benchmark_positive_day_rate_pct": round(sum(value > 0 for value in benchmark_returns) / count * 100, 6) if count else 0.0,
        "realized_pnl_capture_pct": round(scout_pnl / benchmark_pnl * 100, 6) if benchmark_pnl > 0 else None,
        "average_top_10_capture_rate_pct": round(sum(captures) / count, 6) if count else 0.0,
        "top_10_capture_target_pct": 70.0,
        "top_10_capture_target_met": bool(captures) and (sum(captures) / count) >= 70.0,
    }
    return aggregate, hypotheses, feature_evidence


def _build_state(
    dates,
    tickers,
    days,
    failures,
    aggregate,
    hypotheses,
    feature_evidence,
    threshold_pct,
    exploration_top_k,
    dataset_split,
    allowed_partitions,
    holdout_unlocked,
    queue_snapshot,
    max_workers,
) -> dict[str, Any]:
    ordered_days = [days[value] for value in dates if value in days]
    completed = [item["trading_date"] for item in ordered_days if item["status"] in {"COMPLETE", "PARTIAL"}]
    if not completed:
        status = "FAILED"
    elif failures or any(item["status"] == "PARTIAL" for item in ordered_days):
        status = "PARTIAL"
    else:
        status = "COMPLETE"
    return {
        "version": "multi_day_trainer_run_v2.0",
        "run_id": f"trainer-{dates[0]}-to-{dates[-1]}",
        "mode": "RESEARCH_ONLY",
        "trainer_version": "scout_trainer_v1.1",
        "source": "MASSIVE",
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
        "aggregate_performance": aggregate,
        "hypotheses": hypotheses,
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
    tickers: list[str],
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
) -> dict[str, Any]:
    """Run queued historical days with bounded workers and atomic checkpoints."""
    requested_dates = _validate_dates(trading_dates)
    requested_tickers = sorted({ticker.upper() for ticker in tickers})
    if not requested_tickers:
        raise ValueError("At least one ticker is required.")
    if not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be between 1 and 8.")

    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "trainer_run_state.json"
    prior = _read_json(state_path) if resume and state_path.exists() else {}
    expected_policy = {
        "scoring_threshold_pct_override": threshold_pct,
        "exploration_top_k": exploration_top_k,
    }
    reset_queue = not resume
    if prior and (
        prior.get("selection_policy") != expected_policy
        or prior.get("requested_tickers") != requested_tickers
    ):
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
        queue_path.unlink(missing_ok=True)
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
        if (
            previous
            and previous["status"] in {"COMPLETE", "PARTIAL"}
            and (day_dir / "postmortem.json").exists()
        ):
            previous = dict(previous)
            previous["partition"] = dataset_split[trading_date]
            days[trading_date] = previous
            queue.complete(
                ReplayQueue.task_id(trading_date, None, "DAY_REPLAY")
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
            )
            record = {
                "trading_date": trading_date,
                "partition": dataset_split[trading_date],
                "status": manifest["status"],
                "artifact_directory": f"days/{trading_date}",
                "scored_ticker_count": len(manifest["scored_tickers"]),
                "skipped_ticker_count": len(manifest["skipped"]),
            }
            queue.complete(task_id)
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
            }, error

    def checkpoint() -> dict[str, Any]:
        successful = [item for item in days.values() if item["status"] in {"COMPLETE", "PARTIAL"}]
        aggregate, hypotheses, feature_evidence = _aggregate(
            successful, output_root
        )
        state = _build_state(
            dates,
            requested_tickers,
            days,
            failures,
            aggregate,
            hypotheses,
            feature_evidence,
            threshold_pct,
            exploration_top_k,
            dataset_split,
            allowed_partitions,
            holdout_unlocked,
            queue.snapshot(),
            max_workers,
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
    generate_trainer_summary_pdf(state, output_root / "trainer_summary_report.pdf")
    return state
