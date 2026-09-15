from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import time
from typing import Any, Callable

from trainer.providers.massive import MassiveClient
from trainer.validate_contracts import validate_contract


PRIMARY_EXCHANGES = {"XNAS", "XNYS", "XASE"}
MIN_MARKET_CAP = 300_000_000
MAX_MARKET_CAP = 15_000_000_000


class RequestRateLimiter:
    """Deterministic request spacing for Massive Free's five calls/minute."""

    def __init__(
        self,
        minimum_interval_seconds: float = 12.1,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.minimum_interval_seconds = minimum_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last_request is not None:
            remaining = self.minimum_interval_seconds - (now - self._last_request)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last_request = now


def discover_common_stocks(client: MassiveClient, as_of_date: str) -> list[str]:
    """Discover deterministic primary-exchange US common-stock symbols."""
    date.fromisoformat(as_of_date)
    records = client.get_tickers(as_of_date)
    return sorted(
        {
            record["ticker"].upper()
            for record in records
            if record.get("active") is True
            and record.get("type") == "CS"
            and record.get("primary_exchange") in PRIMARY_EXCHANGES
        }
    )


def _new_manifest(tickers: list[str], as_of_date: str) -> dict[str, Any]:
    requested = sorted({ticker.upper() for ticker in tickers})
    return {
        "version": "research_universe_v1.0",
        "as_of_date": as_of_date,
        "status": "PARTIAL" if requested else "COMPLETE",
        "source": "MASSIVE",
        "requested_tickers": requested,
        "completed_tickers": [],
        "remaining_tickers": list(requested),
        "eligible_tickers": [],
        "rejected": {},
        "estimated_remaining_minutes": len(requested) / 5,
    }


def _load_or_create(path: Path, tickers: list[str], as_of_date: str) -> dict[str, Any]:
    expected = sorted({ticker.upper() for ticker in tickers})
    if not path.exists():
        return _new_manifest(expected, as_of_date)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_contract("research_universe", manifest)
    if manifest["as_of_date"] != as_of_date or manifest["requested_tickers"] != expected:
        raise ValueError("Checkpoint request does not match date and ticker universe.")
    return manifest


def _save(path: Path, manifest: dict[str, Any]) -> None:
    validate_contract("research_universe", manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def collect_ticker_overviews(
    client: MassiveClient,
    tickers: list[str],
    as_of_date: str,
    checkpoint_path: Path,
    *,
    max_new: int | None = None,
) -> dict[str, Any]:
    """Resume point-in-time market-cap filtering and checkpoint after every ticker."""
    manifest = _load_or_create(checkpoint_path, tickers, as_of_date)
    remaining = list(manifest["remaining_tickers"])
    budget = len(remaining) if max_new is None else max(0, max_new)
    for ticker in remaining[:budget]:
        overview = client.get_ticker_overview(ticker, as_of_date)
        market_cap = overview.get("market_cap")
        exchange = overview.get("primary_exchange")
        if exchange not in PRIMARY_EXCHANGES or overview.get("type") != "CS":
            manifest["rejected"][ticker] = "NOT_PRIMARY_EXCHANGE_COMMON_STOCK"
        elif not isinstance(market_cap, (int, float)):
            manifest["rejected"][ticker] = "MARKET_CAP_MISSING"
        elif not MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP:
            manifest["rejected"][ticker] = "MARKET_CAP_OUT_OF_RANGE"
        else:
            manifest["eligible_tickers"].append(ticker)
            manifest["eligible_tickers"].sort()
        manifest["completed_tickers"].append(ticker)
        manifest["completed_tickers"].sort()
        manifest["remaining_tickers"].remove(ticker)
        manifest["estimated_remaining_minutes"] = len(manifest["remaining_tickers"]) / 5
        manifest["status"] = (
            "COMPLETE" if not manifest["remaining_tickers"] else "PARTIAL"
        )
        _save(checkpoint_path, manifest)
    _save(checkpoint_path, manifest)
    return manifest
