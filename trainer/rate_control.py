from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable


RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


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
        requests_per_minute: float = 5.0,
        *,
        maximum_interval_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive.")
        self.minimum_interval_seconds = 60.0 / requests_per_minute
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

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "minimum_interval_seconds": self.minimum_interval_seconds,
                "current_interval_seconds": self.current_interval_seconds,
                "maximum_interval_seconds": self.maximum_interval_seconds,
            }
