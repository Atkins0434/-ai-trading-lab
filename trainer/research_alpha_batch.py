from __future__ import annotations

from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trainer.historical_cache import CacheError, HistoricalCache
from trainer.massive_alpha_snapshot import build_massive_alpha_snapshot
from trainer.providers.base import ProviderError
from trainer.providers.massive import MassiveClient
from trainer.report_generator import generate_scout_pdf_report
from trainer.research_scout_alpha import ResearchScoutError, run_research_scout_alpha
from trainer.universe_collector import collect_ticker_overviews
from trainer.validate_contracts import validate_contract


ET = ZoneInfo("America/New_York")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _empty_snapshot(trading_date: str) -> dict[str, Any]:
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
        "securities": [],
    }


def run_massive_alpha_batch(
    client: MassiveClient,
    tickers: list[str],
    trading_date: str,
    *,
    cache_root: Path,
    output_dir: Path,
    threshold_pct: float | None = None,
) -> dict[str, Any]:
    """Screen, collect, score, and report one bounded research-only Alpha batch."""
    target = date.fromisoformat(trading_date)
    requested = sorted({ticker.upper() for ticker in tickers})
    if not requested:
        raise ValueError("At least one ticker is required for an Alpha batch.")

    output_dir.mkdir(parents=True, exist_ok=True)
    universe_path = output_dir / "research_universe.json"
    scout_output_path = output_dir / "research_alpha_output.json"
    pdf_path = output_dir / "research_alpha_report.pdf"
    manifest_path = output_dir / "research_alpha_batch_manifest.json"

    universe = collect_ticker_overviews(
        client,
        requested,
        trading_date,
        universe_path,
        continue_on_error=True,
    )
    cache = HistoricalCache(cache_root)
    start = (target - timedelta(days=45)).isoformat()
    snapshot = _empty_snapshot(trading_date)
    skipped: dict[str, str] = {}

    for security in universe.get("eligible_securities", []):
        ticker = security["ticker"]
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
            )
        except (ProviderError, CacheError, ResearchScoutError) as exc:
            skipped[ticker] = f"{type(exc).__name__}: {exc}"
            continue
        snapshot["securities"].extend(single["securities"])

    result = run_research_scout_alpha(snapshot, threshold_pct=threshold_pct)
    _write_json(scout_output_path, result)
    generate_scout_pdf_report(result, pdf_path)

    scored = sorted(candidate["ticker"] for candidate in result["candidates"])
    if not scored:
        status = "FAILED"
    elif skipped or universe["status"] != "COMPLETE":
        status = "PARTIAL"
    else:
        status = "COMPLETE"
    manifest = {
        "version": "research_alpha_batch_v1.0",
        "trading_date": trading_date,
        "status": status,
        "mode": "RESEARCH_ONLY",
        "source": "MASSIVE",
        "requested_tickers": requested,
        "universe_eligible_tickers": universe["eligible_tickers"],
        "scored_tickers": scored,
        "universe_rejections": universe["rejected"],
        "skipped": skipped,
        "artifacts": {
            "universe": universe_path.name,
            "scout_output": scout_output_path.name,
            "pdf_report": pdf_path.name,
        },
    }
    validate_contract("research_alpha_batch", manifest)
    _write_json(manifest_path, manifest)
    return manifest
