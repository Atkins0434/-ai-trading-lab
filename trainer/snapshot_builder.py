from __future__ import annotations

from datetime import datetime
from typing import Any

from trainer.providers.base import ProviderError
from trainer.providers.tiingo import market_observation, parse_tiingo_timestamp
from trainer.replay_engine import ReplayError, assert_no_future_data


def admit_records_before_freeze(
    records: list[dict[str, Any]],
    freeze_timestamp: str,
) -> list[dict[str, Any]]:
    """Return only records known by the decision boundary.

    Filtering is explicit because date-based APIs may include records later
    than 07:00 on the requested calendar date.
    """
    admitted = []
    for record in records:
        timestamp = parse_tiingo_timestamp(record)
        try:
            assert_no_future_data(timestamp, freeze_timestamp)
        except ReplayError:
            continue
        admitted.append(record)
    return admitted


def latest_market_observation(
    records: list[dict[str, Any]],
    provider_field: str,
    freeze_timestamp: str,
) -> dict[str, Any]:
    """Build one provenance-preserving observation from admitted records."""
    admitted = admit_records_before_freeze(records, freeze_timestamp)
    available = [
        record
        for record in admitted
        if record.get(provider_field) is not None
    ]
    if not available:
        return {
            "value": None,
            "as_of_timestamp": freeze_timestamp,
            "source": "TIINGO",
            "provider_field": provider_field,
        }

    latest = max(
        available,
        key=lambda item: datetime.fromisoformat(
            parse_tiingo_timestamp(item).replace("Z", "+00:00")
        ),
    )
    value = latest[provider_field]
    if not isinstance(value, (int, float)):
        raise ProviderError(
            f"Tiingo field {provider_field} must be numeric or null."
        )
    return market_observation(value, latest, provider_field)
