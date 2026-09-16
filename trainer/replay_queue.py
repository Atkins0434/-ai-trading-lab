from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Iterable


TERMINAL_STATUSES = {"COMPLETE", "FAILED"}


class ReplayQueue:
    """Atomic, resumable date/ticker work queue for one accelerator run."""

    def __init__(self, path: Path, *, max_attempts: int = 3) -> None:
        self.path = path
        self.max_attempts = max_attempts
        self._lock = threading.Lock()
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": "replay_queue_v1.0", "tasks": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("version") != "replay_queue_v1.0":
            raise ValueError("Unsupported replay queue version.")
        for task in payload.get("tasks", {}).values():
            if task["status"] == "IN_PROGRESS":
                task["status"] = "PENDING"
                task["last_error"] = "RECOVERED_INTERRUPTED_TASK"
        return payload

    @staticmethod
    def task_id(trading_date: str, ticker: str | None, stage: str) -> str:
        return f"{trading_date}:{ticker or '*'}:{stage}"

    def enqueue(
        self,
        items: Iterable[tuple[str, str | None, str, str]],
    ) -> None:
        """Enqueue (date, ticker, stage, partition) tuples idempotently."""
        with self._lock:
            for trading_date, ticker, stage, partition in items:
                task_id = self.task_id(trading_date, ticker, stage)
                self._state["tasks"].setdefault(
                    task_id,
                    {
                        "task_id": task_id,
                        "trading_date": trading_date,
                        "ticker": ticker,
                        "stage": stage,
                        "partition": partition,
                        "status": "PENDING",
                        "attempts": 0,
                        "last_error": None,
                        "updated_at": None,
                    },
                )
            self._save_unlocked()

    def pending(self, *, stage: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(task)
                for task in sorted(
                    self._state["tasks"].values(), key=lambda item: item["task_id"]
                )
                if task["status"] == "PENDING"
                and (stage is None or task["stage"] == stage)
            ]

    def claim(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._state["tasks"].get(task_id)
            if task is None or task["status"] != "PENDING":
                return None
            task["status"] = "IN_PROGRESS"
            task["attempts"] += 1
            task["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save_unlocked()
            return dict(task)

    def complete(self, task_id: str) -> None:
        self._finish(task_id, "COMPLETE", None)

    def fail(self, task_id: str, error: str, *, retryable: bool = True) -> None:
        with self._lock:
            task = self._state["tasks"][task_id]
            task["status"] = (
                "PENDING"
                if retryable and task["attempts"] < self.max_attempts
                else "FAILED"
            )
            task["last_error"] = error
            task["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save_unlocked()

    def _finish(self, task_id: str, status: str, error: str | None) -> None:
        with self._lock:
            task = self._state["tasks"][task_id]
            task["status"] = status
            task["last_error"] = error
            task["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save_unlocked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            tasks = list(self._state["tasks"].values())
            counts = {
                status: sum(task["status"] == status for task in tasks)
                for status in ("PENDING", "IN_PROGRESS", "COMPLETE", "FAILED")
            }
            return {
                "version": self._state["version"],
                "counts": counts,
                "tasks": [dict(task) for task in sorted(tasks, key=lambda x: x["task_id"])],
            }

    def _save_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
