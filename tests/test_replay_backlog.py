from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from trainer.output_paths import DAY_FILE_KINDS, day_file
from trainer.replay_backlog import mark, next_unit, read, write
from trainer.replay_scheduled import (
    DAY_KINDS, RUN_FILES, compact_copy, finish, prepare_resume, prune_flatfiles,
    render_index,
)


def backlog():
    return {"version": "replay_backlog_v1.0", "dataset_version": 1, "units": [
        {"id": "A", "start": "2024-04-01", "end": "2024-04-15", "status": "PENDING"},
        {"id": "B", "start": "2024-04-16", "end": "2024-04-30", "status": "PAUSED", "dataset_version_run": 1},
        {"id": "C", "status": "COMPLETE", "dataset_version_run": 1},
        {"id": "D", "status": "FAILED", "dataset_version_run": 1},
    ]}


def test_next_prefers_paused_and_never_auto_retries_failed():
    state = backlog()
    assert next_unit(state)["id"] == "B"
    state["units"][1]["status"] = "COMPLETE"
    assert next_unit(state)["id"] == "A"
    state["units"][0]["status"] = "FAILED"
    assert next_unit(state) is None
    assert next_unit(state, "D")["id"] == "D"


def test_stale_versions_requeue_and_do_not_resume():
    state = backlog()
    state["dataset_version"] = 2
    assert next_unit(state, "B")["status"] == "PENDING"
    assert next_unit(state, "C")["status"] == "PENDING"
    assert next_unit(state, "D")["status"] == "PENDING"


def test_mark_round_trip_and_cli(tmp_path):
    state = backlog()
    fields = {"code_sha": "abc", "last_run_id": "42", "days_requested": 11}
    touched = mark(state, "A", "IN_PROGRESS", fields)
    assert touched["attempts"] == 1
    assert touched["dataset_version_run"] == 1
    assert touched["started_at"]
    path = tmp_path / "backlog.json"
    write(path, state)
    assert read(path) == state
    result = subprocess.run([sys.executable, "-m", "trainer.replay_backlog", "--path", str(path), "mark", "A", "FAILED", "--fields", '{"last_error":"fixture failure"}'], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["last_error"] == "fixture failure"
    assert read(path)["units"][0]["code_sha"] == "abc"
    with pytest.raises(ValueError):
        mark(state, "A", "BOGUS")
    with pytest.raises(ValueError):
        mark(state, "A", "PENDING", {"id": "../bad"})


def test_compact_copy_positive_allowlist(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "results"
    source.mkdir()
    day = "2024-04-01"
    for kind in DAY_FILE_KINDS:
        path = day_file(source, day, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture content")
    for name in RUN_FILES:
        (source / name).write_bytes(b"fixture content")
    write(source / "flatfile_replay_manifest.json", {"completed_dates": [day]})
    (source / "new_giant_payload.bin").write_bytes(b"do not copy")
    (source / "replay_stdout.json").write_text("do not copy")
    compact_copy(source, dest)
    assert set(DAY_KINDS) == {"postmortem", "benchmark_result", "scorable_outcomes", "eligible_outcomes", "replay_report"}
    assert set(RUN_FILES) == {"flatfile_replay_manifest.json", "replay_summary.pdf", "replay_report.pdf", "scorable_outcomes.csv", "eligible_outcomes.csv"}
    expected = {Path(name) for name in RUN_FILES} | {day_file(Path('.'), day, kind) for kind in DAY_KINDS}
    assert {p.relative_to(dest) for p in dest.rglob('*') if p.is_file()} == expected
    for relative in expected:
        assert (dest / relative).read_bytes() == (source / relative).read_bytes()
    # Explicit reruns remove obsolete compact results, not append to them.
    write(source / "flatfile_replay_manifest.json", {"completed_dates": []})
    compact_copy(source, dest)
    assert not any(p.is_file() for p in (dest / "days").rglob('*'))


@pytest.mark.parametrize('status,download_ok,expected', [
    ('PAUSED_WALL_BUDGET', True, True), ('COMPLETE', True, False),
    ('PAUSED_WALL_BUDGET', False, False), (None, True, False),
])
def test_restore_validates_manifest_or_restarts(tmp_path, status, download_ok, expected):
    unit = deepcopy(backlog()['units'][1])
    unit['days_completed'] = 3
    root = tmp_path / 'flatfile_replay' / unit['id']
    root.mkdir(parents=True)
    if status:
        write(root / 'flatfile_replay_manifest.json', {'status': status, 'start_date': unit['start'], 'end_date': unit['end']})
    assert prepare_resume(root, unit, download_ok) is expected
    if not expected:
        assert unit['status'] == 'PENDING'
        assert unit['days_completed'] == 0
        assert list(root.iterdir()) == []
    else:
        assert unit['days_completed'] == 3


def test_finish_day_failure_overrides_pause_and_index(tmp_path):
    state = backlog()
    root = tmp_path / 'A'
    write(root / 'flatfile_replay_manifest.json', {
        'status': 'PAUSED_WALL_BUDGET', 'completed_dates': ['2024-04-01'],
        'requested_dates': ['2024-04-01', '2024-04-02'],
        'failed_dates': {'2024-04-02': 'fixture failure'},
    })
    unit = finish(state, 'A', root, False)
    assert unit['status'] == 'FAILED'
    assert unit['days_completed'] == 1
    assert 'fixture failure' in unit['last_error']
    mark(state, 'A', 'FAILED', {'code_sha': 'abc123', 'last_run_id': '123'})
    write(day_file(root, '2024-04-01', 'benchmark_result'), {
        'comparison': {'result_code': 'SCOUT_OUTPERFORMED'},
        'scout_summary': {'realized_return_pct': 4.2},
    })
    write(day_file(root, '2024-04-01', 'postmortem'), {
        'reachability': {'bars_0': 5, 'picked': 1},
        'execution_policy_review': [{
            'exit_mode': 'ATR', 'cohort_summaries': {'SCOUT_SELECTION': {'realized_return_pct': 3.1}}
        }],
    })
    index = render_index(state, tmp_path, 'owner/repo')
    for expected in ['1/2', '1/0/0', '4.200000%', '3.100000%', 'bars_0=5', 'abc123', 'https://github.com/owner/repo/actions/runs/123']:
        assert expected in index


def test_prune_only_out_of_range_flatfiles(tmp_path):
    for name in ['2024-03-29.csv.gz', '2024-03-29.csv.gz.metadata.json', '2024-04-01.csv.gz', 'unrelated.json']:
        (tmp_path / name).write_text('fixture')
    prune_flatfiles(tmp_path, '2024-04-01', '2024-04-15')
    assert {p.name for p in tmp_path.iterdir()} == {'2024-04-01.csv.gz', 'unrelated.json'}


@pytest.mark.parametrize('manifest_status,failed,expected', [
    ('COMPLETE', False, 'COMPLETE'), ('PAUSED_WALL_BUDGET', False, 'PAUSED'),
    ('COMPLETE', True, 'FAILED'), ('PARTIAL', False, 'FAILED'),
])
def test_terminal_state_mapping(tmp_path, manifest_status, failed, expected):
    state = backlog()
    write(tmp_path / 'flatfile_replay_manifest.json', {
        'status': manifest_status, 'requested_dates': ['2024-04-01'],
        'completed_dates': ['2024-04-01'],
    })
    assert finish(state, 'A', tmp_path, failed)['status'] == expected


def test_corrupt_resume_is_clean_restart(tmp_path):
    unit = deepcopy(backlog()['units'][1])
    root = tmp_path / 'flatfile_replay' / unit['id']
    root.mkdir(parents=True)
    (root / 'flatfile_replay_manifest.json').write_text('{truncated')
    assert not prepare_resume(root, unit, True)
    assert unit['days_completed'] == 0


def test_scheduled_workflow_recovery_and_permissions():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / '.github/workflows/flat-file-replay-scheduled.yml').read_text()
    for required in ['cron: "0 */6 * * *"', 'timeout-minutes: 360', 'actions/download-artifact@v4',
                     'run-id: ${{ steps.unit.outputs.last_run_id }}', 'github-token: ${{ github.token }}',
                     'actions: read', 'contents: write', 'retention-days: 90', '--max-wall-seconds 16200',
                     'multi-day-trainer-${{ github.repository }}', 'cancel-in-progress: false']:
        assert required in workflow
    assert workflow.index('Restore exact prior') < workflow.index('Validate resume') < workflow.index('name: Run replay')
    assert workflow.index('name: Prune flat files') < workflow.index('name: Save bounded flat-file cache')
    for path in (root / '.github/workflows').glob('*.yml'):
        text = path.read_text()
        if '\n  push:' in text:
            assert 'branches-ignore:\n      - replay-results' in text
