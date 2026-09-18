"""Storage and reporting helpers for the scheduled workflow, not the engine."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
import shutil

from trainer.output_paths import day_file, legacy_name
from trainer.replay_backlog import effective_status, mark, read, write

RUN_FILES = (
    "flatfile_replay_manifest.json", "replay_summary.pdf", legacy_name("replay_report"),
    "scorable_outcomes.csv", "eligible_outcomes.csv",
)
DAY_KINDS = ("postmortem", "benchmark_result", "scorable_outcomes", "eligible_outcomes", "replay_report")


def compact_copy(source: Path, destination: Path) -> None:
    """Positive allowlist: no recursive copy of replay outputs."""
    try:
        manifest = read(source / "flatfile_replay_manifest.json") if (source / "flatfile_replay_manifest.json").is_file() else {}
    except (OSError, ValueError):
        manifest = {}
    allowed = [Path(name) for name in RUN_FILES]
    for trading_date in manifest.get("completed_dates", []):
        date.fromisoformat(trading_date)
        allowed.extend(day_file(Path("."), trading_date, kind) for kind in DAY_KINDS)
    destination.mkdir(parents=True, exist_ok=True)
    # Remove obsolete tracked results on explicit reruns/version changes too.
    for existing in destination.rglob("*"):
        if existing.is_file() and existing.relative_to(destination) not in allowed:
            existing.unlink()
    for relative in allowed:
        target = destination / relative
        original = source / relative
        if original.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        elif target.is_file():
            target.unlink()


def prepare_resume(output_root: Path, unit: dict, download_ok: bool) -> bool:
    """Invalid/missing artifact means a clean fresh run, never partial evidence."""
    valid = False
    if unit["status"] == "PAUSED" and download_ok:
        try:
            manifest = read(output_root / "flatfile_replay_manifest.json")
            valid = (
                manifest.get("status") == "PAUSED_WALL_BUDGET"
                and manifest.get("start_date") == unit["start"]
                and manifest.get("end_date") == unit["end"]
            )
            # Engine manifests bound start/end to trading sessions, not weekends.
            if not valid and manifest.get("status") == "PAUSED_WALL_BUDGET":
                from trainer.trading_calendar import generate_trading_dates
                valid = manifest.get("requested_dates") == generate_trading_dates(unit["start"], unit["end"])
        except (OSError, ValueError, TypeError):
            pass
    if not valid:
        print(f"[replay_backlog] {unit['id']}: no valid paused artifact; starting PENDING from scratch", flush=True)
        # This directory is explicitly resolved by the workflow for one unit.
        if output_root.exists():
            if output_root.name != unit["id"] or output_root.parent.name != "flatfile_replay":
                raise ValueError("Refusing to clear a non-unit replay directory")
            shutil.rmtree(output_root)
        unit.update(status="PENDING", days_completed=0)
    output_root.mkdir(parents=True, exist_ok=True)
    return valid


def prune_flatfiles(root: Path, start: str, end: str) -> None:
    """Only remove recognized cached data files; never reference cache or source."""
    date.fromisoformat(start)
    date.fromisoformat(end)
    for path in sorted(root.rglob("*")):
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.csv\.gz(?:\.metadata\.json)?", path.name)
        if path.is_file() and match and not start <= match[1] <= end:
            path.unlink()
            print(f"[flatfile_cache] removed {path}", flush=True)


def render_index(backlog: dict, results: Path, repository: str) -> str:
    rows = ["# Replay results", "", "Research diagnostics only; this branch is never merged into main.", "",
            "| Unit | Status | Days | WIN/MISS/TIE | Primary return | ATR return | Reachability | Code SHA | Run |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    labels = {"SCOUT_OUTPERFORMED": "WIN", "SCOUT_UNDERPERFORMED": "MISS", "SCOUT_TIED": "TIE"}
    for unit in backlog["units"]:
        status = effective_status(backlog, unit)
        verdicts, reachability = Counter(), Counter()
        primary, atr = 0.0, 0.0
        root = results / unit["id"]
        path = root / "flatfile_replay_manifest.json"
        manifest = read(path) if path.is_file() and status != "PENDING" else {}
        for trading_date in manifest.get("completed_dates", []):
            path = day_file(root, trading_date, "benchmark_result")
            benchmark = read(path) if path.is_file() else {}
            verdicts[labels.get(benchmark.get("comparison", {}).get("result_code"), "UNKNOWN")] += 1
            primary += float(benchmark.get("scout_summary", {}).get("realized_return_pct") or 0)
            path = day_file(root, trading_date, "postmortem")
            postmortem = read(path) if path.is_file() else {}
            reachability.update(postmortem.get("reachability", {}))
            for policy in postmortem.get("execution_policy_review", []):
                if policy.get("exit_mode") == "ATR":
                    atr += float(policy.get("cohort_summaries", {}).get("SCOUT_SELECTION", {}).get("realized_return_pct") or 0)
        counts = ", ".join(f"{key}={value}" for key, value in sorted(reachability.items())) or "—"
        run = unit.get("last_run_id")
        link = f"[run {run}](https://github.com/{repository}/actions/runs/{run})" if run else "—"
        done = 0 if status == "PENDING" else unit.get("days_completed", 0)
        rows.append(f"| {unit['id']} | {status} | {done}/{unit.get('days_requested', 0)} | {verdicts['WIN']}/{verdicts['MISS']}/{verdicts['TIE']} | {primary:.6f}% | {atr:.6f}% | {counts} | {unit.get('code_sha') or '—'} | {link} |")
    return "\n".join(rows) + "\n"


def finish(backlog: dict, unit_id: str, output_root: Path, failed: bool, error: str | None = None) -> dict:
    path = output_root / "flatfile_replay_manifest.json"
    try:
        manifest = read(path) if path.is_file() else {}
    except (OSError, ValueError):
        manifest = {}
    status = manifest.get("status")
    target = "FAILED" if failed else {"COMPLETE": "COMPLETE", "PAUSED_WALL_BUDGET": "PAUSED"}.get(status, "FAILED")
    if manifest.get("failed_dates") or any(day.get("status") in {"FAILED", "UNSUPPORTED"} for day in manifest.get("days", [])):
        target = "FAILED"
    fields = {"days_completed": len(manifest.get("completed_dates", []))}
    if manifest.get("requested_dates"):
        fields["days_requested"] = len(manifest["requested_dates"])
    details = json.dumps(manifest.get("failed_dates") or {"manifest_status": status})
    fields["last_error"] = (f"{error}; {details}" if error else details) if target == "FAILED" else None
    return mark(backlog, unit_id, target, fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["finish", "prune"])
    parser.add_argument("--backlog", type=Path)
    parser.add_argument("--unit-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--repository")
    parser.add_argument("--failed", action="store_true")
    parser.add_argument("--error")
    parser.add_argument("--cache-root", type=Path, default=Path("data/flatfiles"))
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    if args.command == "prune":
        prune_flatfiles(args.cache_root, args.start, args.end)
        return
    backlog = read(args.backlog)
    unit = finish(backlog, args.unit_id, args.output_root, args.failed, args.error)
    compact_copy(args.output_root, args.results / args.unit_id)
    write(args.backlog, backlog)
    (args.results / "INDEX.md").write_text(render_index(backlog, args.results, args.repository), encoding="utf-8")
    print(json.dumps(unit))


if __name__ == "__main__":
    main()
