from pathlib import Path
import pytest

from trainer.output_paths import DAY_FILE_KINDS, day_file, legacy_name, migrate_daily_files


@pytest.mark.parametrize("kind", DAY_FILE_KINDS)
def test_every_kind_has_dated_path(tmp_path, kind):
    path = day_file(tmp_path, "2024-03-15", kind)
    assert path.parent == tmp_path / "days" / "2024-03-15"
    assert path.name == f"{kind}_2024-03-15{Path(legacy_name(kind)).suffix}"


def test_migration_preserves_bytes_and_updates_manifest(tmp_path, capsys):
    directory = tmp_path / "days" / "2024-03-15"
    directory.mkdir(parents=True)
    artifacts = {}
    for kind in DAY_FILE_KINDS:
        old = directory / legacy_name(kind)
        old.write_bytes(b"unchanged bytes\n")
        artifacts[kind] = str(old.relative_to(tmp_path))
    manifest = {"days": [{"trading_date": "2024-03-15", "artifacts": artifacts}]}
    migrate_daily_files(tmp_path, manifest)
    assert manifest["file_naming"] == "dated_v1"
    for kind in DAY_FILE_KINDS:
        assert not (directory / legacy_name(kind)).exists()
        new = day_file(tmp_path, "2024-03-15", kind)
        assert new.read_bytes() == b"unchanged bytes\n"
        assert manifest["days"][0]["artifacts"][kind] == str(new.relative_to(tmp_path))
    assert len(capsys.readouterr().err.splitlines()) == len(DAY_FILE_KINDS)
    migrate_daily_files(tmp_path, manifest)
    assert capsys.readouterr().err == ""


def test_no_bare_daily_names_outside_path_contract():
    root = Path(__file__).resolve().parents[1]
    # Cumulative CSVs deliberately retain their original run-level names.
    names = [legacy_name(kind) for kind in DAY_FILE_KINDS if not kind.endswith("outcomes")]
    for tree in ("trainer", ".github"):
        for path in (root / tree).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".yml", ".yaml"}:
                continue
            if path.name == "output_paths.py":
                continue
            # Standalone universe collection is not a per-day replay output.
            if path.name in {"run_universe_collection.py", "massive-universe-smoke.yml"}:
                continue
            source = path.read_text()
            for name in names:
                assert name not in source, (path, name)


def test_engine_resumes_legacy_day_without_rerunning(tmp_path):
    import json
    from trainer.flatfile_replay import run_flatfile_replay, _new_day_record

    calls = []

    def runner(client, store, trading_date, *, output_root, **kwargs):
        calls.append(trading_date)
        artifacts = {}
        for kind in DAY_FILE_KINDS:
            path = day_file(output_root, trading_date, kind)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture bytes")
            artifacts[kind] = str(path.relative_to(output_root))
        return {**_new_day_record(trading_date, smoke_mode=False, max_tickers=None), "status": "COMPLETE", "artifacts": artifacts}

    result = run_flatfile_replay(
        object(), object(), ["2024-03-15"], output_root=tmp_path,
        day_runner=runner,
    )
    assert result["file_naming"] == "dated_v1"
    directory = tmp_path / "days" / "2024-03-15"
    assert {path.name for path in directory.iterdir()} == {
        day_file(tmp_path, "2024-03-15", kind).name for kind in DAY_FILE_KINDS
    }
    for kind in DAY_FILE_KINDS:
        day_file(tmp_path, "2024-03-15", kind).rename(directory / legacy_name(kind))
        result["days"][0]["artifacts"][kind] = str(
            (directory / legacy_name(kind)).relative_to(tmp_path)
        )
    result.pop("file_naming")
    (tmp_path / "flatfile_replay_manifest.json").write_text(json.dumps(result))
    resumed = run_flatfile_replay(
        object(), object(), ["2024-03-15"], output_root=tmp_path,
        day_runner=runner,
    )
    assert calls == ["2024-03-15"]
    assert resumed["status"] == "COMPLETE"
    assert resumed["file_naming"] == "dated_v1"
    for kind in DAY_FILE_KINDS:
        assert day_file(tmp_path, "2024-03-15", kind).read_bytes() == b"fixture bytes"
