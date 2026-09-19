from __future__ import annotations

from trainer.output_paths import daily_path, day_file, migrate_daily_files

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

from trainer.benchmark import build_same_universe_benchmark
from trainer.execution_costs import load_execution_costs
from trainer.flatfile_snapshot import (
    build_flatfile_snapshot,
    load_flatfile_replay_config,
)
from trainer.outcome_grader import grade_replay_outcomes
from trainer.postmortem import build_postmortem
from trainer.postmortem_report import generate_postmortem_pdf
from trainer.providers.massive import MassiveClient
from trainer.providers.massive_flatfiles import (
    DAY_AGGS_DATASET,
    MINUTE_AGGS_DATASET,
    MassiveFlatFileStore,
)
from trainer.rate_control import AdaptiveRateLimiter, load_massive_plan
from trainer.report_generator import generate_scout_pdf_report
from trainer.replay_report import generate_daily_replay_report
from trainer.scorable_outcomes import export_daily_outcomes
from trainer.research_scout_alpha import run_research_scout_alpha
from trainer.trading_calendar import generate_trading_dates
from trainer.universe_builder import (
    build_point_in_time_universe,
    load_universe_config,
    previous_trading_sessions,
)
from trainer.validate_contracts import validate_contract


DayRunner = Callable[..., dict[str, Any]]
ProgressCallback = Callable[[dict[str, Any]], None]
REPLAY_PHASES = (
    "download",
    "universe",
    "snapshot",
    "scoring",
    "grading",
    "benchmark",
    "postmortem",
    "report",
)


class FlatFileReplayError(Exception):
    """Raised when a flat-file replay run cannot safely resume."""


@contextmanager
def _timed_phase(trading_date: str, phase: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        print(
            "[flatfile_replay] "
            f"trading_date={trading_date} phase={phase} "
            f"elapsed_seconds={elapsed:.2f}",
            file=sys.stderr,
            flush=True,
        )


def _new_day_record(
    trading_date: str,
    *,
    smoke_mode: bool,
    max_tickers: int | None,
) -> dict[str, Any]:
    return {
        "trading_date": trading_date,
        "status": "IN_PROGRESS",
        "current_phase": None,
        "phase_status": {phase: "PENDING" for phase in REPLAY_PHASES},
        "phase_wall_time_seconds": {phase: None for phase in REPLAY_PHASES},
        "smoke_mode": smoke_mode,
        "max_tickers": max_tickers,
        "research_evidence": False,
        "universe_size": 0,
        "universe_manifest_hash": None,
        "scored_ticker_count": 0,
        "bar_statistics": None,
        "scorability_statistics": None,
        "excluded_tickers": [],
        "files": [],
        "reference_cache": {
            "hits": 0,
            "fetches": 0,
            "quarter_reuse_hits": 0,
            "errors": 0,
            "definitive_misses": 0,
            "transient_retries": 0,
            "unresolved_failures": 0,
            "listed_after_lagged_date": 0,
        },
        "artifacts": {},
        "error": None,
    }


def _normalize_day_record(
    record: dict[str, Any],
    *,
    smoke_mode: bool,
    max_tickers: int | None,
) -> dict[str, Any]:
    normalized = deepcopy(record)
    normalized.setdefault("current_phase", None)
    normalized.setdefault(
        "phase_status",
        {
            phase: (
                "COMPLETE"
                if normalized.get("status") == "COMPLETE"
                else "PENDING"
            )
            for phase in REPLAY_PHASES
        },
    )
    for phase in REPLAY_PHASES:
        normalized["phase_status"].setdefault(
            phase,
            "COMPLETE" if normalized.get("status") == "COMPLETE" else "PENDING",
        )
    phase_times = normalized.setdefault("phase_wall_time_seconds", {})
    for phase in REPLAY_PHASES:
        phase_times.setdefault(phase, None)
    normalized.setdefault("smoke_mode", smoke_mode)
    normalized.setdefault("max_tickers", max_tickers)
    normalized.setdefault("bar_statistics", None)
    normalized.setdefault("scorability_statistics", None)
    normalized.setdefault("excluded_tickers", [])
    normalized.setdefault(
        "research_evidence",
        normalized.get("status") == "COMPLETE" and not smoke_mode,
    )
    reference_cache = normalized.setdefault("reference_cache", {})
    for metric in (
        "hits",
        "fetches",
        "quarter_reuse_hits",
        "errors",
        "definitive_misses",
        "transient_retries",
        "unresolved_failures",
        "listed_after_lagged_date",
    ):
        reference_cache.setdefault(metric, 0)
    return normalized


@contextmanager
def _tracked_phase(
    progress: dict[str, Any],
    phase: str,
    callback: ProgressCallback | None,
):
    started = time.perf_counter()
    progress["current_phase"] = phase
    progress["phase_status"][phase] = "IN_PROGRESS"
    if callback is not None:
        callback(deepcopy(progress))
    try:
        with _timed_phase(progress["trading_date"], phase):
            yield
    except BaseException:
        progress["phase_wall_time_seconds"][phase] = round(
            time.perf_counter() - started, 3
        )
        progress["phase_status"][phase] = "FAILED"
        if callback is not None:
            callback(deepcopy(progress))
        raise
    else:
        progress["phase_wall_time_seconds"][phase] = round(
            time.perf_counter() - started, 3
        )
        progress["phase_status"][phase] = "COMPLETE"
        if callback is not None:
            callback(deepcopy(progress))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_paths_exist(output_root: Path, day: dict[str, Any]) -> bool:
    return all(
        (output_root / relative).is_file()
        for relative in day.get("artifacts", {}).values()
    )


def _reference_cache_metrics(universe: dict[str, Any]) -> dict[str, int]:
    raw = universe["source"]["query_parameters"].get(
        "ticker_overview_cache", {}
    )
    return {
        "hits": int(raw.get("hits", 0)),
        "fetches": int(raw.get("fetches", 0)),
        "quarter_reuse_hits": int(raw.get("quarter_reuse_hits", 0)),
        "errors": int(raw.get("errors", 0)),
        "definitive_misses": int(raw.get("definitive_misses", 0)),
        "transient_retries": int(raw.get("transient_retries", 0)),
        "unresolved_failures": int(raw.get("unresolved_failures", 0)),
        "listed_after_lagged_date": int(
            raw.get("listed_after_lagged_date", 0)
        ),
    }


def _persisted_reference_cache_metrics(
    output_root: Path,
    trading_date: str,
) -> dict[str, int]:
    path = (
        day_file(output_root, trading_date, "daily_universe_manifest")
    )
    if not path.is_file():
        return {
            "hits": 0,
            "fetches": 0,
            "quarter_reuse_hits": 0,
            "errors": 0,
            "definitive_misses": 0,
            "transient_retries": 0,
            "unresolved_failures": 0,
            "listed_after_lagged_date": 0,
        }
    try:
        universe = json.loads(path.read_text(encoding="utf-8"))
        return _reference_cache_metrics(universe)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {
            "hits": 0,
            "fetches": 0,
            "quarter_reuse_hits": 0,
            "errors": 0,
            "definitive_misses": 0,
            "transient_retries": 0,
            "unresolved_failures": 0,
            "listed_after_lagged_date": 0,
        }


def _ensure_day_files(
    flatfiles: MassiveFlatFileStore,
    trading_date: str,
    lookback_sessions: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    lookback_dates = previous_trading_sessions(
        trading_date, lookback_sessions
    )
    cached = []
    for value in lookback_dates:
        cached.append(
            flatfiles.ensure_day(DAY_AGGS_DATASET, value).to_manifest()
        )
        cached.append(
            flatfiles.ensure_day(MINUTE_AGGS_DATASET, value).to_manifest()
        )
    cached.append(
        flatfiles.ensure_day(MINUTE_AGGS_DATASET, trading_date).to_manifest()
    )
    deduplicated = {
        (item["dataset"], item["trading_date"]): item for item in cached
    }
    return lookback_dates, [
        deduplicated[key] for key in sorted(deduplicated)
    ]


def _run_flatfile_day_impl(
    reference_client: Any,
    flatfiles: MassiveFlatFileStore,
    trading_date: str,
    *,
    output_root: Path,
    lookback_sessions: int,
    strategy_capital: float,
    threshold_pct: float | None,
    exploration_top_k: int,
    reference_cache_root: Path = Path("data/reference_cache"),
    morning_freeze_time: str = "09:15:00",
    comparison_policy_paths: list[Path | str] | None = None,
    max_tickers: int | None = None,
    progress_callback: ProgressCallback | None = None,
    progress: dict[str, Any],
) -> dict[str, Any]:
    """Download, build, and grade one research day from flat files."""
    smoke_mode = max_tickers is not None
    day_dir = output_root / "days" / trading_date
    day_dir.mkdir(parents=True, exist_ok=True)
    with _tracked_phase(progress, "download", progress_callback):
        _, cached_files = _ensure_day_files(
            flatfiles, trading_date, lookback_sessions
        )
        progress["files"] = cached_files
    universe_path = daily_path(day_dir, day_dir.name, "daily_universe_manifest")
    snapshot_path = daily_path(day_dir, day_dir.name, "historical_snapshot")
    scout_path = daily_path(day_dir, day_dir.name, "research_alpha_output")
    scout_pdf_path = daily_path(day_dir, day_dir.name, "research_alpha_report")
    outcome_path = daily_path(day_dir, day_dir.name, "end_of_day_outcome")
    benchmark_path = daily_path(day_dir, day_dir.name, "benchmark_result")
    postmortem_path = daily_path(day_dir, day_dir.name, "postmortem")
    postmortem_pdf_path = daily_path(day_dir, day_dir.name, "postmortem_report")
    replay_report_path = daily_path(day_dir, day_dir.name, "replay_report")

    with _tracked_phase(progress, "universe", progress_callback):
        freeze_config = {"morning_freeze_time": morning_freeze_time}
        freeze_label = morning_freeze_time.replace(":", "")[:4]
        universe = build_point_in_time_universe(
            reference_client,
            flatfiles,
            trading_date,
            universe_path,
            replay_id=f"{trading_date}-{freeze_label}-flatfile-replay",
            reference_cache_root=reference_cache_root,
            max_tickers=max_tickers,
            freeze_config=freeze_config,
        )
        reference_cache = _reference_cache_metrics(universe)
        relative = lambda path: str(path.relative_to(output_root))
        artifacts = {"daily_universe_manifest": relative(universe_path)}
        progress.update({
            "universe_size": universe["eligible_symbol_count"],
            "universe_manifest_hash": universe["manifest_hash"],
            "research_evidence": universe["research_evidence"],
            "reference_cache": reference_cache,
            "artifacts": artifacts,
        })
    if (
        universe["coverage_status"] != "complete"
        or universe["eligible_symbol_count"] == 0
    ):
        progress.update({
            "status": "UNSUPPORTED",
            "current_phase": None,
            "error": ";".join(universe["coverage_reasons"])
            or "EMPTY_ELIGIBLE_UNIVERSE",
        })
        return progress

    with _tracked_phase(progress, "snapshot", progress_callback):
        snapshot_result = build_flatfile_snapshot(
            trading_date,
            universe,
            flatfiles,
            lookback_sessions=lookback_sessions,
            config=freeze_config,
        )
        _write_json(snapshot_path, snapshot_result.snapshot)
        artifacts["historical_snapshot"] = relative(snapshot_path)
        progress["bar_statistics"] = snapshot_result.bar_statistics
        progress["excluded_tickers"] = getattr(
            snapshot_result, "excluded_tickers", []
        )
    with _tracked_phase(progress, "scoring", progress_callback):
        scout = run_research_scout_alpha(
            snapshot_result.snapshot,
            threshold_pct=threshold_pct,
            exploration_top_k=exploration_top_k,
        )
        _write_json(scout_path, scout)
        artifacts["scout_output"] = relative(scout_path)
        scorable_count = int(scout["scorable_candidate_count"])
        not_scorable_count = int(scout["not_scorable_candidate_count"])
        universe_count = len(scout["candidates"])
        progress["scored_ticker_count"] = scorable_count
        progress["scorability_statistics"] = {
            "universe_ticker_count": universe_count,
            "scorable_ticker_count": scorable_count,
            "not_scorable_ticker_count": not_scorable_count,
            "not_scorable_share": (
                not_scorable_count / universe_count if universe_count else 0.0
            ),
        }
        if scorable_count == 0:
            progress.update({
                "status": "FAILED",
                "error": "NO_SCORABLE_TICKERS",
                "artifacts": artifacts,
            })
            for phase in REPLAY_PHASES[REPLAY_PHASES.index("grading"):]:
                progress["phase_status"][phase] = "SKIPPED"
            if progress_callback is not None:
                progress_callback(deepcopy(progress))
            return progress
        generate_scout_pdf_report(scout, scout_pdf_path)
        artifacts["scout_report"] = relative(scout_pdf_path)

    outcome_snapshot = deepcopy(snapshot_result.snapshot)
    outcome_snapshot["execution_policy_version"] = (
        "execution_policy_v1.0_hypothetical"
    )
    with _tracked_phase(progress, "grading", progress_callback):
        outcome = grade_replay_outcomes(
            outcome_snapshot,
            scout,
            snapshot_result.outcome_bars,
            strategy_capital,
            comparison_policy_paths=comparison_policy_paths,
            excluded_tickers=getattr(snapshot_result, "excluded_tickers", []),
        )
        _write_json(outcome_path, outcome)
        artifacts["end_of_day_outcome"] = relative(outcome_path)
    with _tracked_phase(progress, "benchmark", progress_callback):
        benchmark = build_same_universe_benchmark(
            outcome_snapshot,
            scout,
            outcome,
            strategy_capital,
        )
        _write_json(benchmark_path, benchmark)
        artifacts["benchmark_result"] = relative(benchmark_path)
        scorable_path, eligible_path = export_daily_outcomes(
            day_dir, outcome_snapshot, scout, outcome, benchmark
        )
        artifacts["scorable_outcomes"] = relative(scorable_path)
        artifacts["eligible_outcomes"] = relative(eligible_path)
    if smoke_mode:
        postmortem_started = time.perf_counter()
        progress["current_phase"] = "postmortem"
        progress["phase_status"]["postmortem"] = "IN_PROGRESS"
        if progress_callback is not None:
            progress_callback(deepcopy(progress))
        with _timed_phase(trading_date, "postmortem"):
            # Postmortems are Trainer inputs and intentionally reject
            # non-research evidence. A capped smoke run stops at the benchmark
            # rather than weakening that boundary.
            pass
        progress["phase_status"]["postmortem"] = "SKIPPED"
        progress["phase_wall_time_seconds"]["postmortem"] = round(
            time.perf_counter() - postmortem_started,
            6,
        )
        if progress_callback is not None:
            progress_callback(deepcopy(progress))
    else:
        with _tracked_phase(progress, "postmortem", progress_callback):
            postmortem = build_postmortem(
                outcome_snapshot,
                scout,
                benchmark,
                outcome_result=outcome,
                bar_statistics=progress["bar_statistics"],
                scorability_statistics=progress["scorability_statistics"],
                strategy_capital_usd=strategy_capital,
            )
            _write_json(postmortem_path, postmortem)
            generate_postmortem_pdf(
                benchmark,
                postmortem,
                postmortem_pdf_path,
                outcome_result=outcome,
            )
            artifacts.update({
                "postmortem": relative(postmortem_path),
                "postmortem_report": relative(postmortem_pdf_path),
            })
    report_started = time.perf_counter()
    with _tracked_phase(progress, "report", progress_callback):
        generate_daily_replay_report(
            day_dir,
            replay_report_path,
            day_record=progress,
        )
        artifacts["replay_report"] = relative(replay_report_path)
        # A final phase cannot know its own completed duration before rendering.
        # Use the first pass to measure it, then render the persisted copy with
        # that wall-time value while still inside the tracked report phase.
        progress["phase_wall_time_seconds"]["report"] = round(
            time.perf_counter() - report_started,
            3,
        )
        generate_daily_replay_report(
            day_dir,
            replay_report_path,
            day_record=progress,
        )
    progress.update({
        "status": "COMPLETE",
        "current_phase": None,
        "artifacts": artifacts,
        "error": None,
    })
    return progress


def run_flatfile_day(
    reference_client: Any,
    flatfiles: MassiveFlatFileStore,
    trading_date: str,
    *,
    output_root: Path,
    lookback_sessions: int,
    strategy_capital: float,
    threshold_pct: float | None,
    exploration_top_k: int,
    reference_cache_root: Path = Path("data/reference_cache"),
    morning_freeze_time: str = "09:15:00",
    comparison_policy_paths: list[Path | str] | None = None,
    max_tickers: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run one day while preserving the last entered phase on any failure."""
    progress = _new_day_record(
        trading_date,
        smoke_mode=max_tickers is not None,
        max_tickers=max_tickers,
    )
    try:
        return _run_flatfile_day_impl(
            reference_client,
            flatfiles,
            trading_date,
            output_root=output_root,
            lookback_sessions=lookback_sessions,
            strategy_capital=strategy_capital,
            threshold_pct=threshold_pct,
            exploration_top_k=exploration_top_k,
            reference_cache_root=reference_cache_root,
            morning_freeze_time=morning_freeze_time,
            comparison_policy_paths=comparison_policy_paths,
            max_tickers=max_tickers,
            progress_callback=progress_callback,
            progress=progress,
        )
    except BaseException as exc:
        progress["status"] = "FAILED"
        progress["error"] = f"{type(exc).__name__}: {exc}"
        if progress_callback is not None:
            progress_callback(deepcopy(progress))
        raise


def run_flatfile_replay(
    reference_client: Any,
    flatfiles: MassiveFlatFileStore,
    trading_dates: list[str],
    *,
    output_root: Path,
    lookback_sessions: int = 20,
    strategy_capital: float = 2500.0,
    threshold_pct: float | None = None,
    exploration_top_k: int = 0,
    max_tickers: int | None = None,
    morning_freeze_time: str = "09:15:00",
    comparison_policy_paths: list[Path | str] | None = None,
    reference_cache_root: Path = Path("data/reference_cache"),
    resume: bool = True,
    max_wall_seconds: float | None = None,
    day_runner: DayRunner = run_flatfile_day,
) -> dict[str, Any]:
    run_started = time.monotonic()
    dates = sorted({date.fromisoformat(value).isoformat() for value in trading_dates})
    if not dates:
        raise FlatFileReplayError("At least one trading date is required.")
    if lookback_sessions < 1:
        raise FlatFileReplayError("lookback_sessions must be at least one.")
    if strategy_capital <= 0:
        raise FlatFileReplayError("strategy_capital must be positive.")
    if max_wall_seconds is not None and max_wall_seconds < 0:
        raise FlatFileReplayError("max_wall_seconds must be non-negative.")
    from trainer.replay_engine import configured_freeze_datetime
    configured_freeze_datetime(
        dates[0], {"morning_freeze_time": morning_freeze_time}
    )
    if (
        max_tickers is not None
        and (
            not isinstance(max_tickers, int)
            or isinstance(max_tickers, bool)
            or max_tickers < 1
        )
    ):
        raise FlatFileReplayError("max_tickers must be a positive integer.")
    smoke_mode = max_tickers is not None
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "flatfile_replay_manifest.json"
    selection_policy = {
        "threshold_pct": threshold_pct,
        "exploration_top_k": exploration_top_k,
    }
    normalized_comparison_paths = [
        str(value) for value in (comparison_policy_paths or [])
    ]
    active_plan = load_massive_plan()
    universe_config = load_universe_config()
    universe_policy = {
        "version": universe_config["version"],
        "shares_outstanding_lag_days": universe_config[
            "shares_outstanding_lag_days"
        ],
        "ticker_overview_cache_policy": (
            "EARLIER_OR_EQUAL_SAME_CALENDAR_QUARTER"
        ),
    }
    if active_plan["flat_files"] is not True:
        raise FlatFileReplayError(
            "The active Massive plan does not declare flat-file access."
        )
    prior: dict[str, Any] = {}
    if resume and manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = (
            prior.get("requested_dates") == dates
            and prior.get("massive_plan") == active_plan
            and prior.get("baseline_lookback_sessions") == lookback_sessions
            and prior.get("strategy_capital_usd") == strategy_capital
            and prior.get("morning_freeze_time") == morning_freeze_time
            and prior.get("selection_policy") == selection_policy
            and prior.get("comparison_policy_paths", [])
            == normalized_comparison_paths
            and prior.get("universe_policy") == universe_policy
            and prior.get("smoke_mode", False) is smoke_mode
            and prior.get("max_tickers") == max_tickers
        )
        if not identity:
            raise FlatFileReplayError(
                "Existing run manifest does not match this replay request; "
                "use a new output directory."
            )
    if prior:
        migrate_daily_files(output_root, prior)
    prior_days = {
        item["trading_date"]: item for item in prior.get("days", [])
    }
    days: dict[str, dict[str, Any]] = {
        value: _normalize_day_record(
            item,
            smoke_mode=smoke_mode,
            max_tickers=max_tickers,
        )
        for value, item in prior_days.items()
        if value in dates
    }
    failures: dict[str, str] = dict(prior.get("failed_dates", {}))

    def checkpoint(
        *,
        status_override: str | None = None,
        current_trading_date: str | None = None,
        current_phase: str | None = None,
    ) -> dict[str, Any]:
        ordered = [days[value] for value in dates if value in days]
        completed = [
            item["trading_date"]
            for item in ordered
            if item["status"] == "COMPLETE"
        ]
        remaining = [value for value in dates if value not in completed]
        if status_override is not None:
            status = status_override
        elif len(completed) == len(dates):
            status = "COMPLETE"
        elif completed:
            status = "PARTIAL"
        else:
            status = "FAILED"
        manifest = {
            "cost_model_id": load_execution_costs()["cost_model_id"],
            "version": "flatfile_replay_manifest_v1.0",
            "file_naming": "dated_v1",
            "run_id": f"flatfile-{dates[0]}-to-{dates[-1]}",
            "start_date": dates[0],
            "end_date": dates[-1],
            "status": status,
            "current_trading_date": current_trading_date,
            "current_phase": current_phase,
            "smoke_mode": smoke_mode,
            "max_tickers": max_tickers,
            "massive_plan": active_plan,
            "datasets": [MINUTE_AGGS_DATASET, DAY_AGGS_DATASET],
            "baseline_lookback_sessions": lookback_sessions,
            "strategy_capital_usd": strategy_capital,
            "morning_freeze_time": morning_freeze_time,
            "comparison_policy_paths": normalized_comparison_paths,
            "selection_policy": selection_policy,
            "universe_policy": universe_policy,
            "requested_dates": dates,
            "completed_dates": completed,
            "remaining_dates": remaining,
            "failed_dates": failures,
            "reference_cache_summary": {
                key: sum(
                    int(item.get("reference_cache", {}).get(key, 0))
                    for item in ordered
                )
                for key in (
                    "hits",
                    "fetches",
                    "quarter_reuse_hits",
                    "errors",
                    "definitive_misses",
                    "transient_retries",
                    "unresolved_failures",
                    "listed_after_lagged_date",
                )
            },
            "days": ordered,
        }
        validate_contract("flatfile_replay_manifest", manifest)
        _write_json(manifest_path, manifest)
        return manifest

    checkpoint(status_override="IN_PROGRESS", current_phase="STARTING")

    for trading_date in dates:
        previous = prior_days.get(trading_date)
        if (
            previous
            and previous.get("status") == "COMPLETE"
            and _artifact_paths_exist(output_root, previous)
        ):
            failures.pop(trading_date, None)
            continue
        if (
            max_wall_seconds is not None
            and time.monotonic() - run_started > max_wall_seconds
        ):
            remaining = [
                value
                for value in dates
                if days.get(value, {}).get("status") != "COMPLETE"
            ]
            manifest = checkpoint(status_override="PAUSED_WALL_BUDGET")
            print(
                "[flatfile_replay] status=PAUSED_WALL_BUDGET "
                f"remaining_dates={','.join(remaining)}",
                file=sys.stderr,
                flush=True,
            )
            return manifest
        days[trading_date] = _new_day_record(
            trading_date,
            smoke_mode=smoke_mode,
            max_tickers=max_tickers,
        )

        def persist_progress(record: dict[str, Any]) -> None:
            days[trading_date] = deepcopy(record)
            checkpoint(
                status_override="IN_PROGRESS",
                current_trading_date=trading_date,
                current_phase=record["current_phase"],
            )

        try:
            record = day_runner(
                reference_client,
                flatfiles,
                trading_date,
                output_root=output_root,
                lookback_sessions=lookback_sessions,
                strategy_capital=strategy_capital,
                morning_freeze_time=morning_freeze_time,
                comparison_policy_paths=normalized_comparison_paths,
                threshold_pct=threshold_pct,
                exploration_top_k=exploration_top_k,
                max_tickers=max_tickers,
                reference_cache_root=reference_cache_root,
                progress_callback=persist_progress,
            )
            record = _normalize_day_record(
                record,
                smoke_mode=smoke_mode,
                max_tickers=max_tickers,
            )
            days[trading_date] = record
            if record["status"] == "COMPLETE":
                failures.pop(trading_date, None)
            else:
                failures[trading_date] = record["error"] or record["status"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            days[trading_date] = {
                **days[trading_date],
                "status": "FAILED",
                "current_phase": days[trading_date].get("current_phase"),
                "reference_cache": _persisted_reference_cache_metrics(
                    output_root, trading_date
                ),
                "error": error,
            }
            failures[trading_date] = error
        checkpoint(
            status_override="IN_PROGRESS",
            current_trading_date=trading_date,
            current_phase=days[trading_date].get("current_phase"),
        )
    last_failed_date = next(
        (
            value
            for value in reversed(dates)
            if days.get(value, {}).get("status") == "FAILED"
        ),
        None,
    )
    return checkpoint(
        current_trading_date=last_failed_date,
        current_phase=(
            days[last_failed_date].get("current_phase")
            if last_failed_date is not None
            else None
        ),
    )


def parse_args() -> argparse.Namespace:
    config = load_flatfile_replay_config()
    parser = argparse.ArgumentParser(
        description="Run resumable Massive flat-file historical replay."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--cache-root", type=Path, default=Path(config["cache_root"])
    )
    parser.add_argument(
        "--reference-cache-root",
        type=Path,
        default=Path(config["reference_cache_root"]),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--lookback-sessions",
        type=int,
        default=int(config["baseline_lookback_sessions"]),
    )
    parser.add_argument(
        "--strategy-capital",
        type=float,
        default=float(config["strategy_capital_usd"]),
    )
    parser.add_argument("--threshold-pct", type=float)
    parser.add_argument("--exploration-top-k", type=int, default=0)
    parser.add_argument(
        "--max-tickers",
        type=int,
        help=(
            "Cap the sorted universe before overview fetches. This enables "
            "smoke mode and makes all resulting artifacts non-research."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help=(
            "Pause successfully between days after this wall-clock budget. "
            "A day already in progress is never interrupted."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_flatfile_replay_config()
    dates = generate_trading_dates(args.start, args.end)
    output_root = args.output_root or (
        Path(config["output_root"])
        / f"{args.start}-to-{args.end}"
    )
    reference_client = MassiveClient.from_environment(
        rate_limiter=AdaptiveRateLimiter()
    )
    flatfiles = MassiveFlatFileStore.from_environment(
        cache_root=args.cache_root
    )
    result = run_flatfile_replay(
        reference_client,
        flatfiles,
        dates,
        output_root=output_root,
        lookback_sessions=args.lookback_sessions,
        strategy_capital=args.strategy_capital,
        morning_freeze_time=str(
            config["morning_freeze_time"]
        ),
        comparison_policy_paths=config["comparison_policy_paths"],
        threshold_pct=args.threshold_pct,
        exploration_top_k=args.exploration_top_k,
        max_tickers=args.max_tickers,
        reference_cache_root=args.reference_cache_root,
        resume=not args.no_resume,
        max_wall_seconds=args.max_wall_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
