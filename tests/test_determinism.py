from pathlib import Path

from trainer.replay_engine import load_historical_snapshot
from trainer.scout_audit import generate_audit_json
from trainer.scout_engine import run_scout


FIXTURE_PATH = Path(
    "fixtures/2018-01-02/historical_snapshot.json"
)


def test_same_snapshot_produces_same_scout_result():
    snapshot_one = load_historical_snapshot(
        FIXTURE_PATH
    )
    snapshot_two = load_historical_snapshot(
        FIXTURE_PATH
    )

    result_one = run_scout(snapshot_one)
    result_two = run_scout(snapshot_two)

    assert result_one == result_two


def test_same_scout_result_produces_same_hash():
    snapshot = load_historical_snapshot(
        FIXTURE_PATH
    )

    result_one = run_scout(snapshot)
    result_two = run_scout(snapshot)

    audit_one = generate_audit_json(
        result_one
    )
    audit_two = generate_audit_json(
        result_two
    )

    assert (
        audit_one["determinism_hash"]
        == audit_two["determinism_hash"]
    )


def test_determinism_hash_is_stable_for_fixture():
    snapshot = load_historical_snapshot(
        FIXTURE_PATH
    )

    result = run_scout(snapshot)
    audit = generate_audit_json(result)

    assert audit["determinism_hash"] == (
        "c4f51ddbfae9af8c62a64cd34f87f988"
        "f1ce1a9913bc3f81661120ea27da7fc9"
    )
