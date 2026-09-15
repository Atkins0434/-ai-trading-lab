from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trainer.validate_contracts import (
    ContractError,
    load_json,
    validate_contract,
)


ROOT = Path(__file__).resolve().parent.parent
MARKET_TIMEZONE = ZoneInfo("America/New_York")

EXPECTED_FREEZE_HOUR = 7
EXPECTED_FREEZE_MINUTE = 0


class ReplayError(Exception):
    """Raised when a historical replay violates replay rules."""


def load_historical_snapshot(path: Path) -> dict[str, Any]:
    """
    Load and validate a historical snapshot before it can be used
    by Scout.
    """
    try:
        snapshot = load_json(path)
        validate_contract("historical_snapshot", snapshot)
    except ContractError as exc:
        raise ReplayError(
            f"Historical snapshot failed contract validation: {exc}"
        ) from exc

    validate_freeze_timestamp(snapshot)
    validate_point_in_time_inputs(snapshot)

    return snapshot


def parse_freeze_timestamp(snapshot: dict[str, Any]) -> datetime:
    """Parse the snapshot freeze timestamp."""
    raw_timestamp = snapshot["freeze_timestamp"]

    try:
        freeze_time = datetime.fromisoformat(raw_timestamp)
    except ValueError as exc:
        raise ReplayError(
            f"Invalid freeze timestamp: {raw_timestamp}"
        ) from exc

    if freeze_time.tzinfo is None:
        raise ReplayError(
            "freeze_timestamp must contain timezone information."
        )

    return freeze_time.astimezone(MARKET_TIMEZONE)


def validate_freeze_timestamp(snapshot: dict[str, Any]) -> None:
    """
    Enforce the historical decision boundary.

    Morning Scout snapshots must be frozen at exactly 07:00
    America/New_York on the stated trading date.
    """
    freeze_time = parse_freeze_timestamp(snapshot)

    if snapshot["timezone"] != "America/New_York":
        raise ReplayError(
            "Historical replay timezone must be America/New_York."
        )

    if freeze_time.date().isoformat() != snapshot["trading_date"]:
        raise ReplayError(
            "Freeze timestamp date does not match trading_date."
        )

    if (
        freeze_time.hour != EXPECTED_FREEZE_HOUR
        or freeze_time.minute != EXPECTED_FREEZE_MINUTE
        or freeze_time.second != 0
        or freeze_time.microsecond != 0
    ):
        raise ReplayError(
            "Morning historical snapshot must be frozen at exactly "
            "07:00:00 America/New_York."
        )


def assert_no_future_data(
    observed_timestamp: str,
    freeze_timestamp: str,
) -> None:
    """
    Reject an observation that became available after the historical
    freeze boundary.

    Every timestamped input admitted to the morning Scout snapshot
    should eventually pass through this guard.
    """
    try:
        observed_time = datetime.fromisoformat(observed_timestamp)
        freeze_time = datetime.fromisoformat(freeze_timestamp)
    except ValueError as exc:
        raise ReplayError(
            "Observed and freeze timestamps must be valid ISO-8601."
        ) from exc

    if observed_time.tzinfo is None or freeze_time.tzinfo is None:
        raise ReplayError(
            "Historical timestamps must contain timezone information."
        )

    if observed_time > freeze_time:
        raise ReplayError(
            f"Future-data violation: observation at {observed_timestamp} "
            f"occurs after freeze at {freeze_timestamp}."
        )


def validate_point_in_time_inputs(snapshot: dict[str, Any]) -> None:
    """Validate every selection input against the replay freeze.

    Schema validation proves timestamps exist. This traversal proves none of
    them contains information that was unavailable at the decision boundary.
    """
    freeze_timestamp = snapshot["freeze_timestamp"]

    for security in snapshot["securities"]:
        ticker = security["ticker"]
        timestamped_inputs: list[tuple[str, str]] = [
            (
                "eligibility_as_of_timestamp",
                security["eligibility_as_of_timestamp"],
            )
        ]

        market_cap = security.get("market_cap_usd")
        if market_cap is not None:
            timestamped_inputs.append(
                ("market_cap_usd", market_cap["as_of_timestamp"])
            )

        for field, observation in security["market_data"].items():
            timestamped_inputs.append(
                (f"market_data.{field}", observation["as_of_timestamp"])
            )

        for index, bar in enumerate(security.get("premarket_bars", [])):
            timestamped_inputs.append(
                (f"premarket_bars[{index}]", bar["timestamp"])
            )

        for collection in (
            "news",
            "filings",
            "analyst_actions",
            "social_observations",
        ):
            for index, event in enumerate(security.get(collection, [])):
                timestamped_inputs.append(
                    (
                        f"{collection}[{index}]",
                        event["published_timestamp"],
                    )
                )

        for field, observed_timestamp in timestamped_inputs:
            try:
                assert_no_future_data(
                    observed_timestamp,
                    freeze_timestamp,
                )
            except ReplayError as exc:
                raise ReplayError(
                    f"{ticker}.{field}: {exc}"
                ) from exc


def run_contract_test() -> dict[str, Any]:
    """
    Load our first Jan 2, 2018 fixture and prove that the replay
    boundary accepts it.
    """
    fixture_path = (
        ROOT
        / "fixtures"
        / "2018-01-02"
        / "historical_snapshot.json"
    )

    snapshot = load_historical_snapshot(fixture_path)

    print(
        f"Replay contract PASS: {snapshot['replay_id']} "
        f"frozen at {snapshot['freeze_timestamp']}"
    )

    return snapshot


if __name__ == "__main__":
    run_contract_test()
