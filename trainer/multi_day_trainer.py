from __future__ import annotations

from collections import defaultdict
from datetime import date
import json
from pathlib import Path
from typing import Any, Callable

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


def _aggregate(day_records: list[dict[str, Any]], root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scout_returns: list[float] = []
    benchmark_returns: list[float] = []
    scout_pnl = 0.0
    benchmark_pnl = 0.0
    captures: list[float] = []
    results = defaultdict(int)
    evidence: dict[str, dict[str, Any]] = {}

    for record in day_records:
        day_dir = root / record["artifact_directory"]
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

        for miss in postmortem["missed_opportunities"]:
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
                "benchmark_rank": miss["benchmark_rank"],
                "failure_stage": miss["failure_stage"],
                "failure_reason_codes": miss.get("failure_reason_codes", []),
            }
            occurrence_key = (occurrence["trading_date"], occurrence["ticker"])
            known = {(entry["trading_date"], entry["ticker"]) for entry in item["evidence"]}
            if occurrence_key not in known:
                item["evidence"].append(occurrence)

    hypotheses = []
    for item in evidence.values():
        item["evidence"].sort(key=lambda entry: (entry["trading_date"], entry["ticker"]))
        item["independent_occurrence_count"] = len(item["evidence"])
        item["evidence_requirement_met"] = item["independent_occurrence_count"] >= MINIMUM_OCCURRENCES
        item["status"] = "READY_FOR_VALIDATION" if item["evidence_requirement_met"] else "COLLECTING_EVIDENCE"
        item["production_mutation_allowed"] = False
        hypotheses.append(item)
    hypotheses.sort(key=lambda item: (-item["independent_occurrence_count"], item["hypothesis_id"]))

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
        "average_top_10_capture_rate_pct": round(sum(captures) / count, 6) if count else 0.0,
        "top_10_capture_target_pct": 70.0,
        "top_10_capture_target_met": bool(captures) and (sum(captures) / count) >= 70.0,
    }
    return aggregate, hypotheses


def _build_state(
    dates,
    tickers,
    days,
    failures,
    aggregate,
    hypotheses,
    threshold_pct,
    exploration_top_k,
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
        "version": "multi_day_trainer_run_v1.0",
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
        "completed_dates": completed,
        "failed_dates": failures,
        "days": ordered_days,
        "aggregate_performance": aggregate,
        "hypotheses": hypotheses,
        "controls": {
            "real_money_execution_allowed": False,
            "production_mutation_allowed": False,
            "automatic_promotion_allowed": False,
            "minimum_independent_occurrences": MINIMUM_OCCURRENCES,
            "out_of_sample_validation_required": True,
            "blind_holdout_required": True,
            "manual_promotion_required": True,
        },
        "artifacts": {"state": "trainer_run_state.json", "summary_report": "trainer_summary_report.pdf"},
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
) -> dict[str, Any]:
    """Run bounded historical Alpha days sequentially and persist cumulative evidence."""
    requested_dates = _validate_dates(trading_dates)
    requested_tickers = sorted({ticker.upper() for ticker in tickers})
    if not requested_tickers:
        raise ValueError("At least one ticker is required.")

    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "trainer_run_state.json"
    prior = _read_json(state_path) if resume and state_path.exists() else {}
    expected_policy = {
        "scoring_threshold_pct_override": threshold_pct,
        "exploration_top_k": exploration_top_k,
    }
    if prior and prior.get("selection_policy") != expected_policy:
        prior = {}
    dates = _validate_dates(prior.get("requested_dates", []) + requested_dates) if prior else requested_dates
    prior_days = {item["trading_date"]: item for item in prior.get("days", [])}
    days: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {
        key: value for key, value in prior.get("failed_dates", {}).items() if key in dates
    }

    for trading_date in dates:
        day_dir = output_root / "days" / trading_date
        previous = prior_days.get(trading_date)
        if previous and previous["status"] in {"COMPLETE", "PARTIAL"} and (day_dir / "postmortem.json").exists():
            days[trading_date] = previous
            continue
        try:
            manifest = day_runner(
                client,
                requested_tickers,
                trading_date,
                cache_root=cache_root,
                output_dir=day_dir,
                threshold_pct=threshold_pct,
                exploration_top_k=exploration_top_k,
            )
            days[trading_date] = {
                "trading_date": trading_date,
                "status": manifest["status"],
                "artifact_directory": f"days/{trading_date}",
                "scored_ticker_count": len(manifest["scored_tickers"]),
                "skipped_ticker_count": len(manifest["skipped"]),
            }
            failures.pop(trading_date, None)
        except Exception as exc:
            failures[trading_date] = f"{type(exc).__name__}: {exc}"
            days[trading_date] = {
                "trading_date": trading_date,
                "status": "FAILED",
                "artifact_directory": f"days/{trading_date}",
                "scored_ticker_count": 0,
                "skipped_ticker_count": len(requested_tickers),
            }

        successful = [item for item in days.values() if item["status"] in {"COMPLETE", "PARTIAL"}]
        aggregate, hypotheses = _aggregate(successful, output_root)
        _write_json(
            state_path,
            _build_state(
                dates,
                requested_tickers,
                days,
                failures,
                aggregate,
                hypotheses,
                threshold_pct,
                exploration_top_k,
            ),
        )

    successful = [item for item in days.values() if item["status"] in {"COMPLETE", "PARTIAL"}]
    aggregate, hypotheses = _aggregate(successful, output_root)
    state = _build_state(
        dates,
        requested_tickers,
        days,
        failures,
        aggregate,
        hypotheses,
        threshold_pct,
        exploration_top_k,
    )
    validate_contract("multi_day_trainer_run", state)
    _write_json(state_path, state)
    generate_trainer_summary_pdf(state, output_root / "trainer_summary_report.pdf")
    return state
