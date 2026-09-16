from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trainer.historical_cache import CacheError, HistoricalCache
from trainer.historical_news import CatalystError, backfill_news_snapshot
from trainer.benchmark import build_same_universe_benchmark
from trainer.massive_alpha_snapshot import build_massive_alpha_snapshot, regular_session_bars
from trainer.outcome_grader import grade_replay_outcomes
from trainer.postmortem import build_postmortem
from trainer.postmortem_report import generate_postmortem_pdf
from trainer.providers.base import ProviderError
from trainer.providers.massive import MassiveClient
from trainer.replay_queue import ReplayQueue
from trainer.report_generator import generate_scout_pdf_report
from trainer.research_scout_alpha import ResearchScoutError, run_research_scout_alpha
from trainer.universe_collector import collect_ticker_overviews
from trainer.universe_manifest import (
    CI_FIXTURE,
    HISTORICAL_RESEARCH,
    UniverseManifestError,
    evidence_metadata,
    load_or_resolve_manifest,
)
from trainer.validate_contracts import validate_contract


ET = ZoneInfo("America/New_York")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _empty_snapshot(
    trading_date: str, universe_metadata: dict[str, Any]
) -> dict[str, Any]:
    day = date.fromisoformat(trading_date)
    return {
        "replay_id": f"{trading_date}-0700-research-alpha-batch",
        "trading_date": trading_date,
        "freeze_timestamp": datetime.combine(day, time(7), tzinfo=ET).isoformat(),
        "timezone": "America/New_York",
        "universe_version": "research_universe_v1.0",
        "scout_version": "research_scout_alpha_v1.0",
        "execution_policy_version": "execution_disabled",
        "feature_registry_version": "feature_registry_alpha_v1.0",
        "data_source": {
            "provider": MassiveClient.provider_name,
            "feed_version": MassiveClient.feed_version,
        },
        **universe_metadata,
        "securities": [],
    }


def run_massive_alpha_batch(
    client: MassiveClient,
    tickers: list[str] | None,
    trading_date: str,
    *,
    cache_root: Path,
    output_dir: Path,
    threshold_pct: float | None = None,
    exploration_top_k: int = 0,
    dataset_partition: str = "DEVELOPMENT",
    universe_mode: str = CI_FIXTURE,
) -> dict[str, Any]:
    """Screen, collect, score, and report one bounded research-only Alpha batch."""
    target = date.fromisoformat(trading_date)
    requested = sorted({ticker.upper() for ticker in (tickers or [])})
    if universe_mode == CI_FIXTURE and not requested:
        raise ValueError("ci_fixture mode requires at least one ticker.")
    if universe_mode == HISTORICAL_RESEARCH and requested:
        raise ValueError("Hardcoded tickers are forbidden in historical_research mode.")
    if universe_mode not in {CI_FIXTURE, HISTORICAL_RESEARCH}:
        raise ValueError(f"Unknown universe mode: {universe_mode}")
    if dataset_partition not in {"DEVELOPMENT", "VALIDATION", "HOLDOUT"}:
        raise ValueError("Unknown dataset partition.")

    output_dir.mkdir(parents=True, exist_ok=True)
    universe_path = output_dir / "research_universe.json"
    universe_manifest_path = output_dir / "daily_universe_manifest.json"
    scout_output_path = output_dir / "research_alpha_output.json"
    pdf_path = output_dir / "research_alpha_report.pdf"
    outcome_path = output_dir / "end_of_day_outcome.json"
    benchmark_path = output_dir / "benchmark_result.json"
    postmortem_path = output_dir / "postmortem.json"
    postmortem_pdf_path = output_dir / "postmortem_report.pdf"
    manifest_path = output_dir / "research_alpha_batch_manifest.json"
    news_backfill_path = output_dir / "historical_news_backfill.json"
    catalyst_metrics_path = output_dir / "catalyst_shadow_metrics.json"
    ticker_queue_path = output_dir / "ticker_replay_queue.json"
    ticker_result_dir = output_dir / "ticker_results"

    replay_id = f"{trading_date}-0700-research-alpha-batch"
    if universe_mode == CI_FIXTURE:
        universe = collect_ticker_overviews(
            client,
            requested,
            trading_date,
            universe_path,
            continue_on_error=True,
        )
        daily_universe = load_or_resolve_manifest(
            universe_manifest_path,
            provider=client,
            trading_date=trading_date,
            replay_id=replay_id,
            universe_mode=universe_mode,
            fixture_tickers=requested,
            fixture_universe=universe,
        )
    else:
        daily_universe = load_or_resolve_manifest(
            universe_manifest_path,
            provider=client,
            trading_date=trading_date,
            replay_id=replay_id,
            universe_mode=universe_mode,
        )
        universe = {
            "status": (
                "COMPLETE"
                if daily_universe["coverage_status"] == "complete"
                else "UNSUPPORTED"
            ),
            "eligible_tickers": sorted(
                item["ticker"]
                for item in daily_universe["securities"]
                if item["inclusion"]
            ),
            "eligible_securities": [
                {
                    "ticker": item["ticker"],
                    "primary_exchange": item["listing_venue"],
                }
                for item in daily_universe["securities"]
                if item["inclusion"]
            ],
            "rejected": {
                item["ticker"]: ",".join(item["reason_codes"])
                for item in daily_universe["securities"]
                if not item["inclusion"]
            },
        }
    metadata = evidence_metadata(daily_universe)
    if daily_universe["coverage_status"] == "incomplete":
        empty_queue = ReplayQueue(ticker_queue_path).snapshot()
        manifest = {
            "version": "research_alpha_batch_v1.1",
            "trading_date": trading_date,
            "status": "UNSUPPORTED",
            "mode": "RESEARCH_ONLY",
            "dataset_partition": dataset_partition,
            "source": client.provider_name,
            "massive_plan": daily_universe["massive_plan"],
            **metadata,
            "evidence_ineligibility_reasons": daily_universe["coverage_reasons"],
            "requested_tickers": [],
            "universe_eligible_tickers": universe["eligible_tickers"],
            "scored_tickers": [],
            "universe_rejections": universe["rejected"],
            "skipped": {},
            "news_backfill_errors": {},
            "ticker_queue": empty_queue,
            "artifacts": {
                "universe": None,
                "daily_universe_manifest": universe_manifest_path.name,
                "scout_output": None,
                "pdf_report": None,
                "end_of_day_outcome": None,
                "benchmark_result": None,
                "postmortem": None,
                "postmortem_report": None,
                "historical_news_backfill": None,
                "catalyst_shadow_metrics": None,
                "ticker_queue": ticker_queue_path.name,
            },
        }
        validate_contract("research_alpha_batch", manifest)
        _write_json(manifest_path, manifest)
        return manifest
    cache = HistoricalCache(cache_root)
    start = (target - timedelta(days=45)).isoformat()
    snapshot = _empty_snapshot(trading_date, metadata)
    skipped: dict[str, str] = {}
    outcome_bars: dict[str, list[dict[str, Any]]] = {}
    news_snapshots: dict[str, dict[str, Any]] = {}
    catalyst_metrics: list[dict[str, Any]] = []
    news_backfill_errors: dict[str, str] = {}
    retrieved_at = datetime.now(timezone.utc).isoformat()
    securities = universe.get("eligible_securities", [])
    ticker_queue = ReplayQueue(ticker_queue_path)
    ticker_queue.enqueue(
        (trading_date, security["ticker"], "TICKER_REPLAY", dataset_partition)
        for security in securities
    )

    for security in securities:
        ticker = security["ticker"]
        task_id = ReplayQueue.task_id(trading_date, ticker, "TICKER_REPLAY")
        ticker_result_path = ticker_result_dir / f"{ticker}.json"
        known = {
            task["task_id"]: task for task in ticker_queue.snapshot()["tasks"]
        }[task_id]
        if known["status"] == "COMPLETE" and ticker_result_path.exists():
            ticker_result = json.loads(ticker_result_path.read_text(encoding="utf-8"))
            snapshot["securities"].extend(ticker_result["securities"])
            outcome_bars[ticker] = ticker_result["outcome_bars"]
            if ticker_result.get("news_snapshot") is not None:
                news_snapshots[ticker] = ticker_result["news_snapshot"]
            if ticker_result.get("catalyst_metrics") is not None:
                catalyst_metrics.append(ticker_result["catalyst_metrics"])
            if ticker_result.get("news_error") is not None:
                news_backfill_errors[ticker] = ticker_result["news_error"]
            continue
        if ticker_queue.claim(task_id) is None:
            skipped[ticker] = "TICKER_QUEUE_NOT_CLAIMABLE"
            continue
        try:
            daily = cache.get_daily_prices(
                client, ticker, start, trading_date
            )["records"]
            intraday = cache.get_intraday_prices(
                client,
                ticker,
                start,
                trading_date,
                purpose="ALPHA_SELECTION_BASELINE",
            )["records"]
            single = build_massive_alpha_snapshot(
                ticker,
                trading_date,
                daily,
                intraday,
                exchange=security["primary_exchange"],
                universe_metadata=metadata,
            )
            ticker_outcome_bars = regular_session_bars(intraday, trading_date)
        except (ProviderError, CacheError, ResearchScoutError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            skipped[ticker] = error
            ticker_queue.fail(task_id, error, retryable=True)
            continue
        snapshot["securities"].extend(single["securities"])
        outcome_bars[ticker] = ticker_outcome_bars
        news_snapshot = None
        metrics = None
        news_error = None
        if hasattr(client, "get_news"):
            try:
                news_snapshot, metrics = backfill_news_snapshot(
                    client,
                    ticker,
                    trading_date,
                    cache=cache,
                    retrieved_at=retrieved_at,
                )
                news_snapshots[ticker] = news_snapshot
                catalyst_metrics.append(metrics)
            except (
                ProviderError,
                CacheError,
                ResearchScoutError,
                CatalystError,
                ValueError,
            ) as exc:
                news_error = f"{type(exc).__name__}: {exc}"
                news_backfill_errors[ticker] = news_error
        _write_json(
            ticker_result_path,
            {
                "ticker": ticker,
                "trading_date": trading_date,
                "dataset_partition": dataset_partition,
                "securities": single["securities"],
                "outcome_bars": ticker_outcome_bars,
                "news_snapshot": news_snapshot,
                "catalyst_metrics": metrics,
                "news_error": news_error,
            },
        )
        ticker_queue.complete(task_id)

    _write_json(
        news_backfill_path,
        {
            "version": "historical_news_backfill_v1.0",
            "trading_date": trading_date,
            "mode": "RESEARCH_ONLY",
            "snapshots": news_snapshots,
            "errors": news_backfill_errors,
        },
    )
    _write_json(
        catalyst_metrics_path,
        {
            "version": "catalyst_shadow_metrics_v1.0",
            "trading_date": trading_date,
            "mode": "SHADOW_ONLY",
            "production_score_changed": False,
            "ticker_metrics": sorted(
                catalyst_metrics, key=lambda item: item["ticker"]
            ),
        },
    )

    result = run_research_scout_alpha(
        snapshot,
        threshold_pct=threshold_pct,
        exploration_top_k=exploration_top_k,
    )
    _write_json(scout_output_path, result)
    generate_scout_pdf_report(result, pdf_path)

    if metadata["research_evidence"]:
        outcome_snapshot = {
            **snapshot,
            "execution_policy_version": "execution_policy_v1.0_hypothetical",
        }
        outcome = grade_replay_outcomes(
            outcome_snapshot, result, outcome_bars, strategy_capital=2500.0
        )
        benchmark = build_same_universe_benchmark(
            outcome_snapshot, result, outcome, strategy_capital=2500.0
        )
        postmortem = build_postmortem(outcome_snapshot, result, benchmark)
        _write_json(outcome_path, outcome)
        _write_json(benchmark_path, benchmark)
        _write_json(postmortem_path, postmortem)
        generate_postmortem_pdf(benchmark, postmortem, postmortem_pdf_path)

    scored = sorted(candidate["ticker"] for candidate in result["candidates"])
    if not scored:
        status = "FAILED"
    elif skipped or news_backfill_errors or universe["status"] != "COMPLETE":
        status = "PARTIAL"
    else:
        status = "COMPLETE"
    manifest = {
        "version": "research_alpha_batch_v1.1",
        "trading_date": trading_date,
        "status": status,
        "mode": "RESEARCH_ONLY",
        "dataset_partition": dataset_partition,
        "source": "MASSIVE",
        "massive_plan": daily_universe["massive_plan"],
        **metadata,
        "evidence_ineligibility_reasons": daily_universe["coverage_reasons"],
        "requested_tickers": requested,
        "universe_eligible_tickers": universe["eligible_tickers"],
        "scored_tickers": scored,
        "universe_rejections": universe["rejected"],
        "skipped": skipped,
        "news_backfill_errors": news_backfill_errors,
        "ticker_queue": ticker_queue.snapshot(),
        "artifacts": {
            "universe": universe_path.name if universe_mode == CI_FIXTURE else None,
            "daily_universe_manifest": universe_manifest_path.name,
            "scout_output": scout_output_path.name,
            "pdf_report": pdf_path.name,
            "end_of_day_outcome": outcome_path.name if metadata["research_evidence"] else None,
            "benchmark_result": benchmark_path.name if metadata["research_evidence"] else None,
            "postmortem": postmortem_path.name if metadata["research_evidence"] else None,
            "postmortem_report": postmortem_pdf_path.name if metadata["research_evidence"] else None,
            "historical_news_backfill": news_backfill_path.name,
            "catalyst_shadow_metrics": catalyst_metrics_path.name,
            "ticker_queue": ticker_queue_path.name,
        },
    }
    validate_contract("research_alpha_batch", manifest)
    _write_json(manifest_path, manifest)
    return manifest
