import pytest

from trainer.validate_contracts import (
    ContractError,
    SCHEMA_FILES,
    validate_contract,
    validate_schema_definition,
)


def test_all_schema_definitions_are_valid():
    """Every registered JSON schema must itself be valid."""
    for contract_name in SCHEMA_FILES:
        validate_schema_definition(contract_name)


def test_valid_historical_snapshot_passes():
    """A correctly formed minimal historical snapshot should validate."""
    payload = {
        "replay_id": "2018-01-02-morning",
        "trading_date": "2018-01-02",
        "freeze_timestamp": "2018-01-02T07:00:00-05:00",
        "timezone": "America/New_York",
        "universe_version": "universe_v1.0",
        "scout_version": "scout_v1.0",
        "execution_policy_version": "execution_policy_v1.0",
        "feature_registry_version": "feature_registry_v1.0",
        "data_source": {
            "provider": "TEST_FIXTURE",
            "feed_version": "fixture_v1"
        },
        "securities": []
    }

    validate_contract("historical_snapshot", payload)


def test_future_or_unknown_fields_are_rejected():
    """
    Contracts use additionalProperties=false so unexpected
    fields cannot silently enter the pipeline.
    """
    payload = {
        "replay_id": "2018-01-02-morning",
        "trading_date": "2018-01-02",
        "freeze_timestamp": "2018-01-02T07:00:00-05:00",
        "timezone": "America/New_York",
        "universe_version": "universe_v1.0",
        "scout_version": "scout_v1.0",
        "execution_policy_version": "execution_policy_v1.0",
        "feature_registry_version": "feature_registry_v1.0",
        "data_source": {
            "provider": "TEST_FIXTURE",
            "feed_version": "fixture_v1"
        },
        "securities": [],
        "future_closing_price": 999.99
    }

    with pytest.raises(ContractError):
        validate_contract("historical_snapshot", payload)


def test_missing_required_field_is_rejected():
    """Missing required point-in-time metadata must fail validation."""
    payload = {
        "replay_id": "2018-01-02-morning",
        "trading_date": "2018-01-02"
    }

    with pytest.raises(ContractError):
        validate_contract("historical_snapshot", payload)


def test_unknown_contract_is_rejected():
    """Code cannot request an unregistered contract."""
    with pytest.raises(ContractError):
        validate_contract("not_a_real_contract", {})
