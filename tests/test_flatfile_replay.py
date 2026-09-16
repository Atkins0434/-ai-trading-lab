from __future__ import annotations

import json
from pathlib import Path

from trainer.flatfile_replay import run_flatfile_replay


def test_flatfile_replay_resumes_completed_dates_without_reprocessing(tmp_path: Path):
    calls = []

    def day_runner(
        reference_client,
        flatfiles,
        trading_date,
        *,
        output_root,
        **kwargs,
    ):
        calls.append(trading_date)
        marker = output_root / "days" / trading_date / "complete.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"date": trading_date}) + "\n")
        return {
            "trading_date": trading_date,
            "status": "COMPLETE",
            "universe_size": 2,
            "universe_manifest_hash": "sha256:" + "a" * 64,
            "scored_ticker_count": 2,
            "padded_bar_statistics": {
                "premarket_padded_bar_count": 0,
                "regular_padded_bar_count": 0,
            },
            "files": [],
            "artifacts": {
                "marker": str(marker.relative_to(output_root)),
            },
            "error": None,
        }

    dates = ["2018-01-02", "2018-01-03"]
    first = run_flatfile_replay(
        object(),
        object(),
        dates,
        output_root=tmp_path,
        lookback_sessions=1,
        day_runner=day_runner,
    )
    second = run_flatfile_replay(
        object(),
        object(),
        dates,
        output_root=tmp_path,
        lookback_sessions=1,
        day_runner=day_runner,
    )

    assert first["status"] == "COMPLETE"
    assert second["status"] == "COMPLETE"
    assert second["completed_dates"] == dates
    assert calls == dates
