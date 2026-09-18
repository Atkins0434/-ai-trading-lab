"""Single naming contract for daily replay artifacts and legacy migration."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

DAY_FILE_KINDS = (
    "daily_universe_manifest", "historical_snapshot", "research_alpha_output",
    "end_of_day_outcome", "benchmark_result", "postmortem",
    "scorable_outcomes", "eligible_outcomes", "replay_report",
    "postmortem_report", "research_alpha_report", "research_universe",
    "research_alpha_batch_manifest", "historical_news_backfill",
    "catalyst_shadow_metrics", "ticker_replay_queue",
)


def legacy_name(kind: str) -> str:
    if kind not in DAY_FILE_KINDS:
        raise ValueError(f"Unknown daily artifact kind: {kind}")
    extension = "csv" if kind.endswith("outcomes") else "pdf" if kind.endswith("report") else "json"
    return f"{kind}.{extension}"


def daily_path(day_dir: Path, trading_date: str, kind: str) -> Path:
    """Also support batch callers supplying the daily directory explicitly."""
    date.fromisoformat(trading_date)
    name = Path(legacy_name(kind))
    return Path(day_dir) / f"{name.stem}_{trading_date}{name.suffix}"


def day_file(output_root: Path, trading_date: str, kind: str) -> Path:
    return daily_path(Path(output_root) / "days" / trading_date, trading_date, kind)


def migrate_daily_files(output_root: Path, manifest: dict) -> None:
    """Rename legacy files without changing bytes; never overwrite a target."""
    for record in manifest.get("days", []):
        trading_date = record["trading_date"]
        directory = Path(output_root) / "days" / trading_date
        replacements = {}
        for kind in DAY_FILE_KINDS:
            old = directory / legacy_name(kind)
            new = day_file(output_root, trading_date, kind)
            replacements[old.name] = new.name
            if old.exists():
                if new.exists():
                    raise FileExistsError(f"Migration target already exists: {new}")
                old.rename(new)
                print(f"[output_paths] renamed {old} -> {new}", file=sys.stderr, flush=True)

        def rewrite(value):
            if isinstance(value, dict):
                return {key: rewrite(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            if isinstance(value, str):
                path = Path(value)
                if path.name in replacements:
                    return str(path.with_name(replacements[path.name]))
            return value

        for field in ("artifacts", "files"):
            if field in record:
                record[field] = rewrite(record[field])
    manifest["file_naming"] = "dated_v1"


if __name__ == "__main__":
    print(day_file(Path(sys.argv[1]), sys.argv[2], sys.argv[3]))
