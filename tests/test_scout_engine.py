from pathlib import Path

from trainer.replay_engine import load_historical_snapshot
from trainer.scout_engine import run_scout


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = (
    ROOT
    / "fixtures"
    / "2018-01-02"
    / "historical_snapshot.json"
)


def get_result():
    snapshot = load_historical_snapshot(FIXTURE)
    return run_scout(snapshot, threshold_pct=85.0)


def by_ticker(result):
    return {
        candidate["ticker"]: candidate
        for candidate in result["candidates"]
    }


def test_expected_candidate_count():
    result = get_result()

    assert result["qualifying_candidate_count"] == 1


def test_winr_is_selected():
    result = get_result()
    candidates = by_ticker(result)

    winr = candidates["WINR"]

    assert winr["eligible"] is True
    assert winr["selected"] is True
    assert winr["score_pct"] == 87.5
    assert winr["rank"] == 1


def test_wide_is_rejected_for_spread():
    result = get_result()
    candidates = by_ticker(result)

    wide = candidates["WIDE"]

    assert wide["eligible"] is False
    assert wide["selected"] is False
    assert (
        wide["guardrails"]["spread"]["reason_code"]
        == "SPREAD_HARD_REJECT"
    )
    assert "SPREAD_HARD_REJECT" in wide["rejection_reasons"]


def test_thin_is_rejected_for_liquidity():
    result = get_result()
    candidates = by_ticker(result)

    thin = candidates["THIN"]

    assert thin["eligible"] is False
    assert thin["selected"] is False
    assert (
        thin["guardrails"]["liquidity"]["reason_code"]
        == "LIQUIDITY_BELOW_MINIMUM"
    )
    assert "LIQUIDITY_BELOW_MINIMUM" in thin["rejection_reasons"]


def test_meh_passes_gates_but_fails_score():
    result = get_result()
    candidates = by_ticker(result)

    meh = candidates["MEH"]

    assert meh["eligible"] is True
    assert meh["selected"] is False
    assert meh["score_pct"] < 85.0
    assert "BELOW_SELECTION_THRESHOLD" in meh["rejection_reasons"]


def test_only_selected_candidates_receive_rank():
    result = get_result()
    candidates = by_ticker(result)

    assert candidates["WINR"]["rank"] == 1
    assert candidates["WIDE"]["rank"] is None
    assert candidates["THIN"]["rank"] is None
    assert candidates["MEH"]["rank"] is None
