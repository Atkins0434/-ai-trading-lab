import json
from pathlib import Path

import pytest

from trainer.flatfile_replay import _new_day_record, run_flatfile_replay
from trainer.replay_provenance import current_provenance, provenance_differences
from trainer.replay_scheduled import prepare_resume


def runner(calls):
    def day(client, store, trading_date, *, output_root, **kwargs):
        calls.append(trading_date)
        (output_root / "completed-marker").write_text("original evidence")
        return {**_new_day_record(trading_date, smoke_mode=False, max_tickers=None), "status": "COMPLETE"}
    return day


def run(root, calls, **kwargs):
    return run_flatfile_replay(object(), object(), ["2024-03-15"], output_root=root, day_runner=runner(calls), **kwargs)


def save(root, manifest):
    (root / "flatfile_replay_manifest.json").write_text(json.dumps(manifest))


def test_identical_provenance_resumes(tmp_path):
    root = tmp_path / "replay"
    calls = []
    first = run(root, calls)
    second = run(root, calls)
    assert calls == ["2024-03-15"]
    assert first["provenance"] == second["provenance"] == current_provenance([])
    assert second["status"] == "COMPLETE"
    assert second["forced_resume"] is False
    assert not list(tmp_path.glob("replay.stale-*"))


@pytest.mark.parametrize("field", ["scout_id", "rubric_version", "feature_registry_id", "metric_count", "dataset_version", "comparison_policy_paths", "cost_model_id", "git_sha"])
def test_mismatched_provenance_archives_and_runs_fresh(tmp_path, capsys, field):
    root = tmp_path / "replay"
    calls = []
    first = run(root, calls)
    first["provenance"][field] = 12 if field == "metric_count" else "old"
    save(root, first)
    original = (root / "flatfile_replay_manifest.json").read_bytes()
    fresh = run(root, calls)
    assert calls == ["2024-03-15", "2024-03-15"]
    assert fresh["status"] == "COMPLETE"
    assert fresh["provenance"] == current_provenance([])
    stale, = tmp_path.glob("replay.stale-*")
    assert (stale / "flatfile_replay_manifest.json").read_bytes() == original
    assert (stale / "completed-marker").read_text() == "original evidence"
    lines = [line for line in capsys.readouterr().err.splitlines() if 'provenance mismatch' in line]
    assert len(lines) == 1 and field in lines[0]


def test_force_resume_is_auditable_and_never_silently_reused(tmp_path):
    root = tmp_path / "replay"
    calls = []
    first = run(root, calls)
    first["provenance"]["metric_count"] = 12
    save(root, first)
    forced = run(root, calls, force_resume=True)
    assert forced["forced_resume"] is True
    assert forced["previous_provenance"]["metric_count"] == 12
    assert calls == ["2024-03-15"]
    assert not list(tmp_path.glob("replay.stale-*"))
    assert run(root, calls)["forced_resume"] is False
    assert len(calls) == 2


def test_legacy_manifest_logs_all_missing_fields(tmp_path, capsys):
    root = tmp_path / "replay"
    calls = []
    first = run(root, calls)
    del first["provenance"]
    save(root, first)
    run(root, calls)
    message = capsys.readouterr().err
    assert all(key in message for key in current_provenance([]))
    assert len(calls) == 2


@pytest.mark.parametrize("mismatch", [False, True])
def test_scheduled_restore_uses_the_same_check(tmp_path, mismatch):
    root = tmp_path / "flatfile_replay" / "2024-04-A"
    root.mkdir(parents=True)
    provenance = current_provenance()
    if mismatch:
        provenance["dataset_version"] -= 1
    save(root, {"status":"PAUSED_WALL_BUDGET", "start_date":"2024-04-01", "end_date":"2024-04-15", "provenance":provenance})
    unit = dict(id=root.name, status="PAUSED", start="2024-04-01", end="2024-04-15", days_completed=2)
    assert prepare_resume(root, unit, True) is not mismatch
    assert unit["days_completed"] == (0 if mismatch else 2)
    assert bool(list(root.parent.glob(root.name + ".stale-*"))) is mismatch


def test_workflow_output_keys_are_versioned_without_changing_data_keys():
    root = Path(__file__).resolve().parents[1]
    for kind in ("day", "range"):
        source = (root / f".github/workflows/flat-file-replay-{kind}.yml").read_text()
        output_keys = [line for line in source.splitlines() if "replay-output-" in line]
        assert len(output_keys) == 3
        for line in output_keys:
            assert all(f"cache_config.outputs.{key}" in line for key in ("scout_id", "metric_count", "dataset_version", "provenance_key"))
        for line in source.splitlines():
            if "massive-flatfiles-" in line or "massive-reference-" in line:
                assert "provenance_key" not in line and "scout_id" not in line
    for workflow in (root / ".github/workflows").glob("*.yml"):
        assert "--force-resume" not in workflow.read_text()
