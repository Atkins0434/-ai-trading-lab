from __future__ import annotations

from trainer.benchmark import build_same_universe_benchmark
from trainer.postmortem import build_postmortem


def test_benchmark_and_postmortem_preserve_research_only_boundary():
    evidence = {
        "universe_mode": "historical_research",
        "universe_manifest_hash": "sha256:" + "a" * 64,
        "universe_coverage": "complete",
        "research_evidence": True,
        "promotion_eligible": True,
    }
    snapshot = {
        "replay_id": "test-replay",
        "trading_date": "2026-09-14",
        "universe_version": "research_universe_v1.0",
        **evidence,
    }
    scout = {
        "scout_version": "research_scout_alpha_v1.0",
        **evidence,
        "candidates": [{
            "ticker": "MISS",
            "research_eligible": True,
            "research_selected": False,
            "score_pct": 50.0,
            "rejection_reasons": ["BELOW_RESEARCH_THRESHOLD"],
            "component_scores": {
                "relative_volume": {"status": "OBSERVED", "score": 4, "raw_value": 3.0}
            },
        }],
    }
    bars = [{"timestamp": "2026-09-14T13:30:00+00:00", "open": 10.0, "high": 11.0, "low": 10.0, "close": 10.8, "volume": 1000}]
    outcomes = {
        "execution_policy_version": "execution_policy_v1.0_hypothetical",
        **evidence,
        "outcomes": [{
            "ticker": "MISS",
            "selected": False,
            "mfe_pct": 10.0,
            "mae_pct": 0.0,
            "maximum_capturable_move_pct": 10.0,
            "intraday_path": bars,
            "execution_result": {"trade_executed": False, "realized_return_pct": 0.0},
        }],
    }
    benchmark = build_same_universe_benchmark(snapshot, scout, outcomes, 2500.0)
    postmortem = build_postmortem(snapshot, scout, benchmark)

    assert benchmark["comparison"]["result_code"] == "SCOUT_UNDERPERFORMED"
    assert postmortem["result"] == "MISS"
    assert postmortem["missed_opportunities"][0]["failure_stage"] == "BELOW_SELECTION_THRESHOLD"
    assert postmortem["feature_proposals"] == []
