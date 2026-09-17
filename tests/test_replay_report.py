from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from pypdf import PdfReader

from trainer.replay_report import (
    SECTIONS,
    build_daily_report_model,
    generate_cumulative_replay_report,
    generate_daily_replay_report,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "replay_report"
    / "2024-03-15-smoke-artifacts.json"
)


def _bundle() -> dict:
    return json.loads(FIXTURE.read_text())


def _materialize(tmp_path: Path, bundle: dict) -> Path:
    day_dir = tmp_path / "days" / "2024-03-15"
    day_dir.mkdir(parents=True)
    names = {
        "universe": "daily_universe_manifest.json",
        "snapshot": "historical_snapshot.json",
        "scout": "research_alpha_output.json",
        "outcome": "end_of_day_outcome.json",
        "benchmark": "benchmark_result.json",
        "postmortem": "postmortem.json",
    }
    for key, name in names.items():
        if bundle.get(key) is not None:
            (day_dir / name).write_text(json.dumps(bundle[key]) + "\n")
    return day_dir


def _text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def test_smoke_fixture_is_small_and_intraday_paths_are_stripped():
    assert FIXTURE.stat().st_size < 200_000
    assert all(not item["intraday_path"] for item in _bundle()["outcome"]["outcomes"])


def test_daily_report_renders_sections_and_expected_rows(tmp_path: Path):
    bundle = _bundle()
    day_dir = _materialize(tmp_path, bundle)
    report = generate_daily_replay_report(
        day_dir,
        day_record=bundle["day_record"],
    )
    text = _text(report)
    model = build_daily_report_model(bundle, day_record=bundle["day_record"])

    assert all(heading in text for heading in SECTIONS)
    assert "GROSS — no execution costs modeled" in text
    assert len(model["trade_rows"]) == 1
    assert len(model["mover_rows"]) == 3
    assert len(model["scored_candidates"]) == 1
    assert sum(count for _, count in model["histogram"]) == 5
    assert "ACHR" in text


def test_zero_selection_report_uses_one_sentence_form(tmp_path: Path):
    bundle = deepcopy(_bundle())
    selected = bundle["scout"]["candidates"][0]
    selected["qualification_selected"] = False
    selected["research_selected"] = False
    selected["rejection_reasons"] = ["BELOW_RESEARCH_THRESHOLD"]
    day_dir = _materialize(tmp_path, bundle)

    text = _text(generate_daily_replay_report(day_dir, day_record=bundle["day_record"]))

    assert (
        "No Scout trades were selected because no candidate reached the configured threshold."
        in text
    )


def test_daily_report_renders_execution_policy_comparison(tmp_path: Path):
    bundle = deepcopy(_bundle())
    primary = bundle["outcome"]["outcomes"][0]
    primary["selected"] = True
    primary["execution_result"].update({
        "policy_id": "execution_policy_v1.0",
        "exit_mode": "PERCENT",
        "sizing_mode": "FIXED_FRACTION",
        "maximum_position_drawdown_pct": -1.25,
        "trade_executed": True,
    })
    atr_execution = {
        **primary["execution_result"],
        "policy_id": "execution_policy_atr_v1.0",
        "exit_mode": "ATR",
        "sizing_mode": "RISK_PER_TRADE",
        "exit_price": 6.12,
        "exit_reason": "PROFIT_TARGET",
        "realized_pnl_usd": 300.0,
        "realized_return_pct": 20.0,
        "capture_ratio": 1.0,
    }
    bundle["outcome"]["policy_comparisons"] = [{
        "policy_id": "execution_policy_atr_v1.0",
        "exit_mode": "ATR",
        "sizing_mode": "RISK_PER_TRADE",
        "executions": [{
            "ticker": "ACHR",
            "cohort": "SCOUT_SELECTION",
            "benchmark_rank": None,
            "execution_result": atr_execution,
        }],
        "summary": {
            "net_realized_pnl_usd": 300.0,
            "realized_return_pct": 12.0,
            "win_rate_pct": 100.0,
            "average_winner_pct": 20.0,
            "average_loser_pct": None,
            "average_capture_ratio": 1.0,
            "max_drawdown_pct": -1.25,
            "exit_reason_counts": {
                "TRAILING_STOP": 0,
                "PROFIT_TARGET": 1,
                "SESSION_END": 0,
                "ENTRY_REJECTED": {},
            },
        },
    }]
    day_dir = _materialize(tmp_path, bundle)

    report = generate_daily_replay_report(
        day_dir, day_record=bundle["day_record"]
    )
    text = _text(report)
    model = build_daily_report_model(bundle, day_record=bundle["day_record"])

    assert len(model["execution_policies"]) == 2
    assert "Execution policy comparison" in text
    assert "atr_v1.0" in text
    assert "PROFIT_TARGET" in text


def test_cumulative_report_covers_completed_days(tmp_path: Path):
    bundle = _bundle()
    _materialize(tmp_path, bundle)
    manifest = {
        "run_id": "flatfile-run-fixture",
        "completed_dates": ["2024-03-15"],
        "days": [{"trading_date": "2024-03-15", **bundle["day_record"]}],
    }
    (tmp_path / "flatfile_replay_manifest.json").write_text(
        json.dumps(manifest) + "\n"
    )

    report = generate_cumulative_replay_report(tmp_path)
    text = _text(report)

    assert "Cumulative Replay Summary" in text
    assert "2024-03-15" in text
    assert "Miss-category totals across completed days" in text
