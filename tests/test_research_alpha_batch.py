from __future__ import annotations

from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from trainer.research_alpha_batch import run_massive_alpha_batch


ET = ZoneInfo("America/New_York")


def aggregate(timestamp: datetime, price: float, volume: float = 1000) -> dict:
    return {
        "t": int(timestamp.timestamp() * 1000),
        "o": price,
        "h": price + 0.05,
        "l": price - 0.05,
        "c": price + 0.02,
        "v": volume,
    }


def alpha_inputs() -> tuple[list[dict], list[dict]]:
    target = date(2026, 9, 14)
    trading_days = []
    cursor = target - timedelta(days=1)
    while len(trading_days) < 20:
        if cursor.weekday() < 5:
            trading_days.append(cursor)
        cursor -= timedelta(days=1)
    trading_days.reverse()
    daily = [
        aggregate(datetime.combine(day, time(0), tzinfo=ET), 10 + index / 100, 1_000_000)
        for index, day in enumerate(trading_days)
    ]
    intraday = []
    for day in trading_days + [target]:
        for minute in range(180):
            timestamp = datetime.combine(day, time(4), tzinfo=ET) + timedelta(minutes=minute)
            intraday.append(aggregate(timestamp, 10 + minute / 1000, 5000))
    return daily, intraday


class FakeClient:
    provider_name = "MASSIVE"
    feed_version = "massive_rest_v2_aggs"

    def __init__(self) -> None:
        self.daily, self.intraday = alpha_inputs()
        target = date(2026, 9, 14)
        for minute in range(390):
            timestamp = datetime.combine(target, time(9, 30), tzinfo=ET) + timedelta(minutes=minute)
            self.intraday.append(aggregate(timestamp, 10 + minute / 1000, 8000))

    def get_ticker_overview(self, ticker, as_of_date):
        market_cap = 2_000_000_000 if ticker == "GOOD" else 20_000_000_000
        return {
            "ticker": ticker,
            "type": "CS",
            "primary_exchange": "XNAS",
            "market_cap": market_cap,
        }

    def get_daily_prices(self, ticker, start, end):
        return self.daily

    def get_intraday_prices(self, ticker, start, end, frequency="1min"):
        return self.intraday


def test_batch_screens_scores_and_creates_auditable_artifacts(tmp_path: Path):
    output_dir = tmp_path / "reports" / "2026-09-14"
    manifest = run_massive_alpha_batch(
        FakeClient(),
        ["GOOD", "BIG"],
        "2026-09-14",
        cache_root=tmp_path / "cache",
        output_dir=output_dir,
        threshold_pct=0,
    )

    result = json.loads((output_dir / "research_alpha_output.json").read_text())
    assert manifest["status"] == "COMPLETE"
    assert manifest["scored_tickers"] == ["GOOD"]
    assert manifest["universe_rejections"] == {"BIG": "MARKET_CAP_OUT_OF_RANGE"}
    assert result["mode"] == "RESEARCH_ONLY"
    assert result["candidates"][0]["execution_eligible"] is False
    assert (output_dir / "research_alpha_report.pdf").read_bytes().startswith(b"%PDF")
    outcome = json.loads((output_dir / "end_of_day_outcome.json").read_text())
    benchmark = json.loads((output_dir / "benchmark_result.json").read_text())
    postmortem = json.loads((output_dir / "postmortem.json").read_text())
    assert outcome["outcomes"][0]["mfe_pct"] > 0
    assert benchmark["benchmark_method"] == "TOP_10_MOVERS_SAME_UNIVERSE"
    assert postmortem["feature_proposals"] == []
    assert "no Production Scout configuration was modified" in postmortem["notes"][0]
    assert (output_dir / "postmortem_report.pdf").read_bytes().startswith(b"%PDF")
    assert (output_dir / "historical_news_backfill.json").exists()
    assert (output_dir / "catalyst_shadow_metrics.json").exists()
    assert manifest["news_backfill_errors"] == {}
    assert manifest["ticker_queue"]["counts"]["COMPLETE"] == 1
    assert (output_dir / "ticker_replay_queue.json").exists()
