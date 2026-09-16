from __future__ import annotations

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
from trainer.research_scout_alpha import run_research_scout_alpha
from trainer.trading_calendar import generate_trading_dates
from trainer.universe_builder import (
    build_point_in_time_universe,
    load_universe_config,
    previous_trading_sessions,
)
from trainer.validate_contracts import validate_contract


DayRunner = Callable[..., dict[str, Any]]


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
    }


def _persisted_reference_cache_metrics(
    output_root: Path,
    trading_date: str,
) -> dict[str, int]:
    path = (
        output_root
        / "days"
        / trading_date
        / "daily_universe_manifest.json"
    )
    if not path.is_file():
        return {
            "hits": 0,
            "fetches": 0,
            "quarter_reuse_hits": 0,
            "errors": 0,
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
) -> dict[str, Any]:
    """Download, build, and grade one research day from flat files."""
    day_dir = output_root / "days" / trading_date
    day_dir.mkdir(parents=True, exist_ok=True)
    with _timed_phase(trading_date, "download"):
        _, cached_files = _ensure_day_files(
            flatfiles, trading_date, lookback_sessions
        )
    universe_path = day_dir / "daily_universe_manifest.json"
    snapshot_path = day_dir / "historical_snapshot.json"
    scout_path = day_dir / "research_alpha_output.json"
    scout_pdf_path = day_dir / "research_alpha_report.pdf"
    outcome_path = day_dir / "end_of_day_outcome.json"
    benchmark_path = day_dir / "benchmark_result.json"
    postmortem_path = day_dir / "postmortem.json"
    postmortem_pdf_path = day_dir / "postmortem_report.pdf"

    with _timed_phase(trading_date, "universe"):
        universe = build_point_in_time_universe(
            reference_client,
            flatfiles,
            trading_date,
            universe_path,
            replay_id=f"{trading_date}-0700-flatfile-replay",
            reference_cache_root=reference_cache_root,
        )
    reference_cache = _reference_cache_metrics(universe)
    relative = lambda path: str(path.relative_to(output_root))
    artifacts = {"daily_universe_manifest": relative(universe_path)}
    if (
        universe["coverage_status"] != "complete"
        or universe["eligible_symbol_count"] == 0
    ):
        return {
            "trading_date": trading_date,
            "status": "UNSUPPORTED",
            "universe_size": universe["eligible_symbol_count"],
            "universe_manifest_hash": universe["manifest_hash"],
            "scored_ticker_count": 0,
            "padded_bar_statistics": None,
            "files": cached_files,
            "reference_cache": reference_cache,
            "artifacts": artifacts,
            "error": ";".join(universe["coverage_reasons"])
            or "EMPTY_ELIGIBLE_UNIVERSE",
        }

    with _timed_phase(trading_date, "snapshot"):
        snapshot_result = build_flatfile_snapshot(
            trading_date,
            universe,
            flatfiles,
            lookback_sessions=lookback_sessions,
        )
        _write_json(snapshot_path, snapshot_result.snapshot)
    with _timed_phase(trading_date, "scoring"):
        scout = run_research_scout_alpha(
            snapshot_result.snapshot,
            threshold_pct=threshold_pct,
            exploration_top_k=exploration_top_k,
        )
        _write_json(scout_path, scout)
        generate_scout_pdf_report(scout, scout_pdf_path)

    outcome_snapshot = deepcopy(snapshot_result.snapshot)
    outcome_snapshot["execution_policy_version"] = (
        "execution_policy_v1.0_hypothetical"
    )
    with _timed_phase(trading_date, "grading"):
        outcome = grade_replay_outcomes(
            outcome_snapshot,
            scout,
            snapshot_result.outcome_bars,
            strategy_capital,
        )
        _write_json(outcome_path, outcome)
    with _timed_phase(trading_date, "benchmark"):
        benchmark = build_same_universe_benchmark(
            outcome_snapshot,
            scout,
            outcome,
            strategy_capital,
        )
        _write_json(benchmark_path, benchmark)
    with _timed_phase(trading_date, "postmortem"):
        postmortem = build_postmortem(
            outcome_snapshot,
            scout,
            benchmark,
        )
        _write_json(postmortem_path, postmortem)
        generate_postmortem_pdf(
            benchmark, postmortem, postmortem_pdf_path
        )
    artifacts.update({
        "historical_snapshot": relative(snapshot_path),
        "scout_output": relative(scout_path),
        "scout_report": relative(scout_pdf_path),
        "end_of_day_outcome": relative(outcome_path),
        "benchmark_result": relative(benchmark_path),
        "postmortem": relative(postmortem_path),
        "postmortem_report": relative(postmortem_pdf_path),
    })
    return {
        "trading_date": trading_date,
        "status": "COMPLETE",
        "universe_size": universe["eligible_symbol_count"],
        "universe_manifest_hash": universe["manifest_hash"],
        "scored_ticker_count": len(scout["candidates"]),
        "padded_bar_statistics": snapshot_result.padded_bar_statistics,
        "files": cached_files,
        "reference_cache": reference_cache,
        "artifacts": artifacts,
        "error": None,
    }


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
    reference_cache_root: Path = Path("data/reference_cache"),
    resume: bool = True,
    day_runner: DayRunner = run_flatfile_day,
) -> dict[str, Any]:
    dates = sorted({date.fromisoformat(value).isoformat() for value in trading_dates})
    if not dates:
        raise FlatFileReplayError("At least one trading date is required.")
    if lookback_sessions < 1:
        raise FlatFileReplayError("lookback_sessions must be at least one.")
    if strategy_capital <= 0:
        raise FlatFileReplayError("strategy_capital must be positive.")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "flatfile_replay_manifest.json"
    selection_policy = {
        "threshold_pct": threshold_pct,
        "exploration_top_k": exploration_top_k,
    }
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
            and prior.get("selection_policy") == selection_policy
            and prior.get("universe_policy") == universe_policy
        )
        if not identity:
            raise FlatFileReplayError(
                "Existing run manifest does not match this replay request; "
                "use a new output directory."
            )
    prior_days = {
        item["trading_date"]: item for item in prior.get("days", [])
    }
    days: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = dict(prior.get("failed_dates", {}))

    def checkpoint() -> dict[str, Any]:
        ordered = [days[value] for value in dates if value in days]
        completed = [
            item["trading_date"]
            for item in ordered
            if item["status"] == "COMPLETE"
        ]
        if len(completed) == len(dates):
            status = "COMPLETE"
        elif completed:
            status = "PARTIAL"
        else:
            status = "FAILED"
        manifest = {
            "version": "flatfile_replay_manifest_v1.0",
            "run_id": f"flatfile-{dates[0]}-to-{dates[-1]}",
            "start_date": dates[0],
            "end_date": dates[-1],
            "status": status,
            "massive_plan": active_plan,
            "datasets": [MINUTE_AGGS_DATASET, DAY_AGGS_DATASET],
            "baseline_lookback_sessions": lookback_sessions,
            "strategy_capital_usd": strategy_capital,
            "selection_policy": selection_policy,
            "universe_policy": universe_policy,
            "requested_dates": dates,
            "completed_dates": completed,
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
                )
            },
            "days": ordered,
        }
        validate_contract("flatfile_replay_manifest", manifest)
        _write_json(manifest_path, manifest)
        return manifest

    for trading_date in dates:
        previous = prior_days.get(trading_date)
        if (
            previous
            and previous.get("status") == "COMPLETE"
            and _artifact_paths_exist(output_root, previous)
        ):
            days[trading_date] = previous
            failures.pop(trading_date, None)
            continue
        try:
            record = day_runner(
                reference_client,
                flatfiles,
                trading_date,
                output_root=output_root,
                lookback_sessions=lookback_sessions,
                strategy_capital=strategy_capital,
                threshold_pct=threshold_pct,
                exploration_top_k=exploration_top_k,
                reference_cache_root=reference_cache_root,
            )
            days[trading_date] = record
            if record["status"] == "COMPLETE":
                failures.pop(trading_date, None)
            else:
                failures[trading_date] = record["error"] or record["status"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            days[trading_date] = {
                "trading_date": trading_date,
                "status": "FAILED",
                "universe_size": 0,
                "universe_manifest_hash": None,
                "scored_ticker_count": 0,
                "padded_bar_statistics": None,
                "files": [],
                "reference_cache": _persisted_reference_cache_metrics(
                    output_root, trading_date
                ),
                "artifacts": {},
                "error": error,
            }
            failures[trading_date] = error
        checkpoint()
    return checkpoint()


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
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dates = generate_trading_dates(args.start, args.end)
    output_root = args.output_root or (
        Path(load_flatfile_replay_config()["output_root"])
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
        threshold_pct=args.threshold_pct,
        exploration_top_k=args.exploration_top_k,
        reference_cache_root=args.reference_cache_root,
        resume=not args.no_resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
