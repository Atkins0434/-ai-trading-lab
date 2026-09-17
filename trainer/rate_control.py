from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable


RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
ROOT = Path(__file__).resolve().parent.parent
MASSIVE_PLAN_PATH = ROOT / "config" / "massive_plan.json"
_USE_CONFIG = object()
DEFAULT_REFERENCE_FETCH_WORKERS = 8


class MassivePlanError(ValueError):
    """Raised when the configured Massive subscription contract is invalid."""


def _validated_plan(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "plan",
        "rest_calls_per_minute",
        "history_years",
        "flat_files",
    }
    allowed = required | {"reference_fetch_workers"}
    if not required.issubset(payload) or not set(payload).issubset(allowed):
        raise MassivePlanError(
            "Massive plan config must contain the provider fields "
            f"{sorted(required)} and may include reference_fetch_workers."
        )
    if not isinstance(payload["plan"], str) or not payload["plan"].strip():
        raise MassivePlanError("Massive plan name must be a non-empty string.")
    limit = payload["rest_calls_per_minute"]
    if limit is not None and (
        not isinstance(limit, (int, float))
        or isinstance(limit, bool)
        or limit <= 0
    ):
        raise MassivePlanError(
            "rest_calls_per_minute must be null or a positive number."
        )
    years = payload["history_years"]
    if not isinstance(years, int) or isinstance(years, bool) or years <= 0:
        raise MassivePlanError("history_years must be a positive integer.")
    if not isinstance(payload["flat_files"], bool):
        raise MassivePlanError("flat_files must be a boolean.")
    workers = payload.get(
        "reference_fetch_workers", DEFAULT_REFERENCE_FETCH_WORKERS
    )
    if (
        not isinstance(workers, int)
        or isinstance(workers, bool)
        or workers < 1
    ):
        raise MassivePlanError(
            "reference_fetch_workers must be a positive integer."
        )
    return {
        "plan": payload["plan"].strip(),
        "rest_calls_per_minute": limit,
        "history_years": years,
        "flat_files": payload["flat_files"],
    }


def load_massive_plan(path: Path = MASSIVE_PLAN_PATH) -> dict[str, Any]:
    """Load and validate the active Massive subscription capabilities."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MassivePlanError(f"Massive plan config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MassivePlanError(
            f"Invalid Massive plan JSON at line {exc.lineno}, column {exc.colno}."
        ) from exc
    if not isinstance(payload, dict):
        raise MassivePlanError("Massive plan config must be a JSON object.")
    return _validated_plan(payload)


def load_reference_fetch_workers(
    path: Path = MASSIVE_PLAN_PATH,
) -> int:
    """Return reference concurrency without changing evidence metadata."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MassivePlanError(f"Massive plan config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MassivePlanError(
            f"Invalid Massive plan JSON at line {exc.lineno}, column {exc.colno}."
        ) from exc
    if not isinstance(payload, dict):
        raise MassivePlanError("Massive plan config must be a JSON object.")
    _validated_plan(payload)
    return int(
        payload.get(
            "reference_fetch_workers", DEFAULT_REFERENCE_FETCH_WORKERS
        )
    )


def active_massive_plan(client: Any | None = None) -> dict[str, Any]:
    """Return a validated copy of the plan attached to a client or config."""
    attached = getattr(client, "massive_plan", None) if client is not None else None
    return _validated_plan(attached) if attached is not None else load_massive_plan()


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_backoff_seconds: float = 1.0
    maximum_backoff_seconds: float = 60.0

    def delay_for_attempt(self, attempt: int) -> float:
        if attempt < 1:
            raise ValueError("attempt must be at least 1.")
        return min(
            self.maximum_backoff_seconds,
            self.base_backoff_seconds * (2 ** (attempt - 1)),
        )


class AdaptiveRateLimiter:
    """Thread-safe request pacing with shared throttle recovery."""

    def __init__(
        self,
        requests_per_minute: float | None | object = _USE_CONFIG,
        *,
        plan_path: Path = MASSIVE_PLAN_PATH,
        maximum_interval_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_minute is _USE_CONFIG:
            requests_per_minute = load_massive_plan(plan_path)[
                "rest_calls_per_minute"
            ]
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive.")
        self.requests_per_minute = requests_per_minute
        self.minimum_interval_seconds = (
            0.0
            if requests_per_minute is None
            else 60.0 / requests_per_minute
        )
        self.current_interval_seconds = self.minimum_interval_seconds
        self.maximum_interval_seconds = maximum_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            reservation = max(now, self._next_request_at)
            delay = reservation - now
            self._next_request_at = reservation + self.current_interval_seconds
        if delay > 0:
            self._sleep(delay)

    def record_success(self) -> None:
        with self._lock:
            self.current_interval_seconds = max(
                self.minimum_interval_seconds,
                self.current_interval_seconds * 0.9,
            )

    def record_throttle(self, retry_after_seconds: float | None = None) -> None:
        with self._lock:
            now = self._clock()
            if self.minimum_interval_seconds > 0:
                self.current_interval_seconds = min(
                    self.maximum_interval_seconds,
                    max(
                        self.minimum_interval_seconds,
                        self.current_interval_seconds * 2,
                    ),
                )
            cooldown = (
                retry_after_seconds
                if retry_after_seconds is not None
                else self.current_interval_seconds
            )
            self._next_request_at = max(self._next_request_at, now + cooldown)

    def snapshot(self) -> dict[str, float | None]:
        with self._lock:
            return {
                "requests_per_minute": self.requests_per_minute,
                "minimum_interval_seconds": self.minimum_interval_seconds,
                "current_interval_seconds": self.current_interval_seconds,
                "maximum_interval_seconds": self.maximum_interval_seconds,
            }
