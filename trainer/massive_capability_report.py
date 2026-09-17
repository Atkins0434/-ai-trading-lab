from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from trainer.providers.base import ProviderError
from trainer.providers.massive import MassiveClient, parse_massive_timestamp
from trainer.rate_control import active_massive_plan
from trainer.replay_engine import configured_freeze_datetime
from trainer.validate_contracts import load_json
from trainer.universe_collector import RequestRateLimiter


MARKET_TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_REPORT = Path("reports/massive/capability_report.json")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "scout_alpha_v1.json"


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = [parse_massive_timestamp(record) for record in records]
    return {
        "available": True,
        "record_count": len(records),
        "earliest_timestamp": min(timestamps, default=None),
        "latest_timestamp": max(timestamps, default=None),
    }


def _probe(operation: Callable[[], list[dict[str, Any]]]) -> dict[str, Any]:
    try:
        return _summarize(operation())
    except ProviderError as exc:
        return {
            "available": False,
            "record_count": 0,
            "earliest_timestamp": None,
            "latest_timestamp": None,
            "error": str(exc),
        }


def _partition_intraday(
    records: list[dict[str, Any]],
    trading_date: str,
    config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    premarket = []
    last_60_minutes = []
    regular = []
    freeze = configured_freeze_datetime(trading_date, config)
    last_hour_start = freeze - timedelta(minutes=60)
    for record in records:
        observed = datetime.fromisoformat(
            parse_massive_timestamp(record)
        ).astimezone(MARKET_TIMEZONE)
        if observed.date().isoformat() != trading_date:
            continue
        observed_time = observed.time()
        if time(4, 0) <= observed_time and observed < freeze:
            premarket.append(record)
            if observed >= last_hour_start:
                last_60_minutes.append(record)
        if time(9, 30) <= observed_time < time(16, 0):
            regular.append(record)
    return premarket, last_60_minutes, regular


def build_report(
    client: MassiveClient,
    ticker: str,
    trading_date: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_config = config or load_json(CONFIG_PATH)
    plan = active_massive_plan(client)
    target_date = date.fromisoformat(trading_date)
    intraday_result: dict[str, Any]
    try:
        intraday = client.get_intraday_prices(
            ticker,
            trading_date,
            target_date.isoformat(),
        )
        premarket, last_60_minutes, regular = _partition_intraday(
            intraday, trading_date, active_config
        )
        intraday_result = {
            "full_day": _summarize(intraday),
            "premarket_0400_to_freeze_et": _summarize(premarket),
            "premarket_last_60m_et": _summarize(last_60_minutes),
            "regular_0930_to_1600_et": _summarize(regular),
        }
    except ProviderError as exc:
        unavailable = {
            "available": False,
            "record_count": 0,
            "earliest_timestamp": None,
            "latest_timestamp": None,
            "error": str(exc),
        }
        intraday_result = {
            "full_day": unavailable,
            "premarket_0400_to_freeze_et": unavailable,
            "premarket_last_60m_et": unavailable,
            "regular_0930_to_1600_et": unavailable,
        }

    return {
        "provider": client.provider_name,
        "feed_version": client.feed_version,
        "plan_under_test": plan["plan"],
        "massive_plan": plan,
        "ticker": ticker.upper(),
        "trading_date": trading_date,
        "freeze_time": (
            f"{configured_freeze_datetime(trading_date, active_config).time().isoformat()} "
            "America/New_York"
        ),
        "daily": _probe(
            lambda: client.get_daily_prices(
                ticker,
                trading_date,
                trading_date,
            )
        ),
        "intraday": intraday_result,
        "selection_contract_passed": (
            intraday_result["premarket_last_60m_et"]["record_count"]
            >= int(active_config["minimum_real_bars_60m"])
        ),
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the configured Massive historical stock capabilities."
    )
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--date", default="2026-09-14")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limiter = RequestRateLimiter()
    client = MassiveClient.from_environment(before_request=limiter.wait)
    report = build_report(client, args.ticker, args.date)
    write_report(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["selection_contract_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
