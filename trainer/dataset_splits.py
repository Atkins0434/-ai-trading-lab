from __future__ import annotations

from typing import Iterable


PARTITIONS = {"DEVELOPMENT", "VALIDATION", "HOLDOUT"}


def build_chronological_splits(trading_dates: Iterable[str]) -> dict[str, str]:
    """Assign ordered sessions without shuffling future regimes into development."""
    dates = sorted(set(trading_dates))
    if not dates:
        raise ValueError("At least one trading date is required.")
    if len(dates) < 5:
        return {value: "DEVELOPMENT" for value in dates}
    holdout_count = max(1, round(len(dates) * 0.2))
    validation_count = max(1, round(len(dates) * 0.2))
    development_end = len(dates) - validation_count - holdout_count
    if development_end < 1:
        raise ValueError("Dataset split requires at least one development date.")
    result: dict[str, str] = {}
    for index, value in enumerate(dates):
        if index < development_end:
            result[value] = "DEVELOPMENT"
        elif index < len(dates) - holdout_count:
            result[value] = "VALIDATION"
        else:
            result[value] = "HOLDOUT"
    return result


def enforce_partition_access(
    split: dict[str, str],
    allowed_partitions: Iterable[str],
    *,
    holdout_unlocked: bool = False,
) -> list[str]:
    allowed = set(allowed_partitions)
    unknown = allowed - PARTITIONS
    if unknown:
        raise ValueError(f"Unknown dataset partitions: {sorted(unknown)}")
    if "HOLDOUT" in allowed and not holdout_unlocked:
        raise PermissionError(
            "Holdout sessions are locked; explicit holdout authorization is required."
        )
    return sorted(value for value, partition in split.items() if partition in allowed)
