from pathlib import Path

from trainer.replay_engine import load_historical_snapshot
from trainer.scout_audit import generate_audit_json
from trainer.scout_engine import load_scout_config, run_scout


FIXTURE_PATH = Path(
    "fixtures/2018-01-02/historical_snapshot.json"
)
LEGACY_CONFIG = load_scout_config()
LEGACY_CONFIG["session"]["morning_freeze_time"] = "07:00:00"
LEGACY_CONFIG["minimum_real_bars_60m"] = 0


def test_same_snapshot_produces_same_scout_result():
    snapshot_one = load_historical_snapshot(
        FIXTURE_PATH, config=LEGACY_CONFIG
    )
    snapshot_two = load_historical_snapshot(
        FIXTURE_PATH, config=LEGACY_CONFIG
    )

    result_one = run_scout(snapshot_one, config=LEGACY_CONFIG)
    result_two = run_scout(snapshot_two, config=LEGACY_CONFIG)

    assert result_one == result_two


def test_same_scout_result_produces_same_hash():
    snapshot = load_historical_snapshot(
        FIXTURE_PATH, config=LEGACY_CONFIG
    )

    result_one = run_scout(snapshot, config=LEGACY_CONFIG)
    result_two = run_scout(snapshot, config=LEGACY_CONFIG)

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
        FIXTURE_PATH, config=LEGACY_CONFIG
    )

    result = run_scout(snapshot, config=LEGACY_CONFIG)
    audit = generate_audit_json(result)

    assert audit["determinism_hash"] == (
        "d807a73ea2f56c1ccd89ac41000cd2d9"
        "1a7322d6d15d592947b4bf3334e1d6a7"
    )
