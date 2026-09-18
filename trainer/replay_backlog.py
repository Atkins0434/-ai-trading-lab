"""Scheduler bookkeeping only; never participates in replay decisions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

STATUSES = {"PENDING", "IN_PROGRESS", "PAUSED", "COMPLETE", "FAILED"}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def effective_status(backlog: dict, unit: dict) -> str:
    version = unit.get("dataset_version_run")
    if "dataset_version_run" in unit and version != backlog["dataset_version"]:
        return "PENDING"
    return unit["status"]


def next_unit(backlog: dict, unit_id: str | None = None) -> dict | None:
    """A manual ID is an explicit retry; automatic selection never retries FAILED."""
    units = backlog["units"]
    if unit_id:
        units = [unit for unit in units if unit["id"] == unit_id]
        if not units:
            raise ValueError(f"Unknown backlog unit: {unit_id}")
    else:
        units = [unit for unit in units if effective_status(backlog, unit) in {"PAUSED", "PENDING"}]
        units = sorted(units, key=lambda unit: effective_status(backlog, unit) != "PAUSED")
    if not units:
        return None
    selected = deepcopy(units[0])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", selected["id"]):
        raise ValueError("Unit IDs must be safe directory names")
    selected["status"] = effective_status(backlog, selected)
    return selected


def mark(backlog: dict, unit_id: str, status: str, fields: dict | None = None) -> dict:
    if status not in STATUSES:
        raise ValueError(f"Unknown backlog status: {status}")
    unit = next(unit for unit in backlog["units"] if unit["id"] == unit_id)
    fields = fields or {}
    if set(fields) & {"id", "start", "end", "status"}:
        raise ValueError("Fields cannot replace unit identity or status")
    defaults = dict(code_sha=None, dataset_version_run=backlog["dataset_version"], attempts=0,
                    last_run_id=None, started_at=None, completed_at=None,
                    days_completed=0, days_requested=0, last_error=None)
    for key, value in defaults.items():
        unit.setdefault(key, value)
    now = datetime.now(timezone.utc).isoformat()
    if status == "IN_PROGRESS":
        unit.update(attempts=unit["attempts"] + 1, started_at=now,
                    completed_at=None, last_error=None,
                    dataset_version_run=backlog["dataset_version"])
    elif status == "COMPLETE":
        unit["completed_at"] = now
    unit.update(fields)
    unit["status"] = status
    return unit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path(os.environ.get("REPLAY_BACKLOG_PATH", "config/replay_backlog.json")))
    sub = parser.add_subparsers(dest="command", required=True)
    choice = sub.add_parser("next")
    choice.add_argument("--unit-id")
    change = sub.add_parser("mark")
    change.add_argument("id")
    change.add_argument("status", choices=sorted(STATUSES))
    change.add_argument("--fields", default="{}")
    sub.add_parser("status")
    args = parser.parse_args()
    backlog = read(args.path)
    if args.command == "next":
        result = next_unit(backlog, args.unit_id)
    elif args.command == "mark":
        result = mark(backlog, args.id, args.status, json.loads(args.fields))
        write(args.path, backlog)
    else:
        result = deepcopy(backlog)
        for unit in result["units"]:
            unit["status"] = effective_status(backlog, unit)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
