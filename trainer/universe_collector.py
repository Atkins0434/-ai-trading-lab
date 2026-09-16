from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import time
from typing import Any, Callable

from trainer.providers.massive import MassiveClient
from trainer.providers.base import ProviderError
from trainer.rate_control import MASSIVE_PLAN_PATH, load_massive_plan
from trainer.validate_contracts import validate_contract


PRIMARY_EXCHANGES = {"XNAS", "XNYS", "XASE"}
MIN_MARKET_CAP = 300_000_000
MAX_MARKET_CAP = 15_000_000_000
_USE_CONFIG = object()


class RequestRateLimiter:
    """Deterministic request spacing when the active Massive plan has a cap."""

    def __init__(
        self,
        requests_per_minute: float | None | object = _USE_CONFIG,
        *,
        plan_path: Path = MASSIVE_PLAN_PATH,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_minute is _USE_CONFIG:
            requests_per_minute = load_massive_plan(plan_path)[
                "rest_calls_per_minute"
            ]
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive.")
        self.requests_per_minute = requests_per_minute
        self.minimum_interval_seconds = (
            0.0
            if requests_per_minute is None
            else (60.0 / requests_per_minute) + 0.1
        )
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None

    def wait(self) -> None:
        if self.minimum_interval_seconds == 0:
            return
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
    plan = load_massive_plan()
    rate = plan["rest_calls_per_minute"]
    return {
        "version": "research_universe_v1.0",
        "as_of_date": as_of_date,
        "status": "PARTIAL" if requested else "COMPLETE",
        "source": "MASSIVE",
        "massive_plan": plan,
        "universe_mode": "ci_fixture",
        "research_evidence": False,
        "promotion_eligible": False,
        "requested_tickers": requested,
        "completed_tickers": [],
        "remaining_tickers": list(requested),
        "eligible_tickers": [],
        "eligible_securities": [],
        "rejected": {},
        "estimated_remaining_minutes": (
            len(requested) / rate if rate is not None else 0.0
        ),
    }


def _load_or_create(path: Path, tickers: list[str], as_of_date: str) -> dict[str, Any]:
    expected = sorted({ticker.upper() for ticker in tickers})
    active_plan = load_massive_plan()
    if not path.exists():
        return _new_manifest(expected, as_of_date)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        "universe_mode" not in manifest
        or manifest.get("massive_plan") != active_plan
    ):
        suffix = 0
        while True:
            marker = "" if suffix == 0 else f".{suffix}"
            legacy = path.with_name(
                f"{path.stem}.legacy-plan-or-fixture{marker}{path.suffix}"
            )
            if not legacy.exists():
                path.replace(legacy)
                break
            suffix += 1
        return _new_manifest(expected, as_of_date)
    validate_contract("research_universe", manifest)
    manifest.setdefault("eligible_securities", [])
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
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Resume point-in-time market-cap filtering and checkpoint after every ticker."""
    manifest = _load_or_create(checkpoint_path, tickers, as_of_date)
    remaining = list(manifest["remaining_tickers"])
    budget = len(remaining) if max_new is None else max(0, max_new)
    for ticker in remaining[:budget]:
        try:
            overview = client.get_ticker_overview(ticker, as_of_date)
        except ProviderError:
            if not continue_on_error:
                raise
            manifest["rejected"][ticker] = "OVERVIEW_PROVIDER_ERROR"
            overview = None
        if overview is None:
            pass
        else:
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
                manifest["eligible_securities"].append({
                    "ticker": ticker,
                    "primary_exchange": exchange,
                    "market_cap_usd": market_cap,
                    "as_of_date": as_of_date,
                })
                manifest["eligible_securities"].sort(key=lambda item: item["ticker"])
        manifest["completed_tickers"].append(ticker)
        manifest["completed_tickers"].sort()
        manifest["remaining_tickers"].remove(ticker)
        rate = manifest["massive_plan"]["rest_calls_per_minute"]
        manifest["estimated_remaining_minutes"] = (
            len(manifest["remaining_tickers"]) / rate
            if rate is not None
            else 0.0
        )
        manifest["status"] = (
            "COMPLETE" if not manifest["remaining_tickers"] else "PARTIAL"
        )
        _save(checkpoint_path, manifest)
    _save(checkpoint_path, manifest)
    return manifest
