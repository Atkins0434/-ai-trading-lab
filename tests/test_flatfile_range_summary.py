from __future__ import annotations

import json
from pathlib import Path

from trainer.flatfile_range_summary import render_range_summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_range_summary_handles_complete_failed_and_pending_days(
    tmp_path: Path,
):
    complete_date = "2024-03-01"
    day_root = tmp_path / "days" / complete_date
    _write_json(day_root / "research_alpha_output.json", {
        "candidates": [{
            "ticker": "MOVE",
            "qualification_selected": True,
            "research_selected": True,
        }],
    })
    _write_json(day_root / "benchmark_result.json", {
        "execution_policy_version": "execution_policy_v1.0",
        "combined_summary": {"realized_return_pct": 5.0},
        "comparison": {"result_code": "SCOUT_OUTPERFORMED"},
    })
    _write_json(day_root / "end_of_day_outcome.json", {
        "outcomes": [{
            "ticker": "MOVE",
            "selected": True,
            "execution_result": {
                "policy_id": "execution_policy_v1.0",
                "capture_ratio": 0.4,
            },
        }],
        "policy_comparisons": [{
            "policy_id": "execution_policy_atr_v1.0",
            "exit_mode": "ATR",
            "summary": {"realized_return_pct": 4.25},
            "executions": [{
                "cohort": "SCOUT_SELECTION",
                "execution_result": {"capture_ratio": 0.35},
            }],
        }],
    })
    _write_json(day_root / "postmortem.json", {
        "reachability": {
            "bars_0": 3,
            "bars_1_9": 2,
            "bars_10_29": 1,
            "visible_scored_low": 1,
            "visible_guardrail_rejected": 1,
            "picked": 2,
            "opening_range_reachable_count": 4,
        }
    })
    manifest = {
        "status": "PAUSED_WALL_BUDGET",
        "requested_dates": [
            complete_date,
            "2024-03-04",
            "2024-03-05",
        ],
        "completed_dates": [complete_date],
        "remaining_dates": ["2024-03-04", "2024-03-05"],
        "days": [
            {
                "trading_date": complete_date,
                "status": "COMPLETE",
                "universe_size": 2455,
                "scored_ticker_count": 32,
                "excluded_tickers": [{"ticker": "TPC"}],
                "phase_wall_time_seconds": {"snapshot": 900.0, "grading": 60.0},
            },
            {
                "trading_date": "2024-03-04",
                "status": "FAILED",
                "universe_size": 2400,
                "scored_ticker_count": 0,
                "excluded_tickers": [],
                "phase_wall_time_seconds": {"download": 5.0},
            },
            {
                "trading_date": "2024-03-05",
                "status": "IN_PROGRESS",
                "universe_size": 0,
                "scored_ticker_count": 0,
                "excluded_tickers": [],
                "phase_wall_time_seconds": {},
            },
        ],
    }

    summary = render_range_summary(manifest, tmp_path)

    assert "2024-03-01 status=COMPLETE eligible=2455 scorable=32 selected=1" in summary
    assert "primary_return=5.0000% atr_return=4.2500%" in summary
    assert "2024-03-04 status=FAILED" in summary
    assert "2024-03-05 status=IN_PROGRESS" in summary
    assert "Days completed/requested: 1/3" in summary
    assert "Days remaining: 2024-03-04, 2024-03-05" in summary
    assert "WIN/TIE/MISS: 1/0/0" in summary
    assert "execution_policy_v1.0 cumulative return: 5.0000%" in summary
    assert "execution_policy_atr_v1.0 cumulative return: 4.2500%" in summary
    assert "execution_policy_v1.0 cumulative capture ratio: 0.4000" in summary
    assert "execution_policy_atr_v1.0 cumulative capture ratio: 0.3500" in summary
    assert (
        "Top-10 reachability: bars_0=3 bars_1_9=2 bars_10_29=1 "
        "visible_scored_low=1 visible_guardrail_rejected=1 picked=2 "
        "opening_range_reachable=4"
    ) in summary
    assert "Total wall time: 965.0000 seconds" in summary
