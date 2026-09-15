import pytest
from copy import deepcopy

from trainer.replay_engine import (
    ReplayError,
    assert_no_future_data,
    load_historical_snapshot,
    run_contract_test,
    ROOT,
)


def test_jan_2_fixture_passes_replay_contract():
    """Our first historical fixture must pass the replay boundary."""
    snapshot = run_contract_test()

    assert snapshot["trading_date"] == "2018-01-02"
    assert snapshot["freeze_timestamp"] == "2018-01-02T07:00:00-05:00"


def test_fixture_loads_directly():
    """Replay engine should load and validate the Jan 2 fixture."""
    path = (
        ROOT
        / "fixtures"
        / "2018-01-02"
        / "historical_snapshot.json"
    )

    snapshot = load_historical_snapshot(path)

    assert snapshot["replay_id"] == "2018-01-02-morning"


def test_data_before_freeze_is_allowed():
    """Information known before 7:00 AM may enter the snapshot."""
    assert_no_future_data(
        "2018-01-02T06:59:59-05:00",
        "2018-01-02T07:00:00-05:00",
    )


def test_data_exactly_at_freeze_is_allowed():
    """Information timestamped exactly at the freeze is allowed."""
    assert_no_future_data(
        "2018-01-02T07:00:00-05:00",
        "2018-01-02T07:00:00-05:00",
    )


def test_data_after_freeze_is_rejected():
    """Even one second of future information must fail."""
    with pytest.raises(ReplayError, match="Future-data violation"):
        assert_no_future_data(
            "2018-01-02T07:00:01-05:00",
            "2018-01-02T07:00:00-05:00",
        )


def test_obvious_future_data_is_rejected():
    """Later market information cannot leak into the morning run."""
    with pytest.raises(ReplayError, match="Future-data violation"):
        assert_no_future_data(
            "2018-01-02T15:59:00-05:00",
            "2018-01-02T07:00:00-05:00",
        )


def test_timezone_is_required():
    """Naive timestamps are forbidden in historical replay."""
    with pytest.raises(ReplayError, match="timezone information"):
        assert_no_future_data(
            "2018-01-02T06:59:59",
            "2018-01-02T07:00:00-05:00",
        )


def test_loader_rejects_future_market_observation():
    path = (
        ROOT
        / "fixtures"
        / "2018-01-02"
        / "historical_snapshot.json"
    )
    snapshot = load_historical_snapshot(path)
    future = deepcopy(snapshot)
    future["securities"][0]["market_data"]["last_price"][
        "as_of_timestamp"
    ] = "2018-01-02T07:00:01-05:00"

    from trainer.replay_engine import validate_point_in_time_inputs

    with pytest.raises(
        ReplayError,
        match="WINR.market_data.last_price",
    ):
        validate_point_in_time_inputs(future)


def test_loader_rejects_future_information_event():
    path = (
        ROOT
        / "fixtures"
        / "2018-01-02"
        / "historical_snapshot.json"
    )
    snapshot = load_historical_snapshot(path)
    future = deepcopy(snapshot)
    future["securities"][0]["news"][0][
        "published_timestamp"
    ] = "2018-01-02T08:00:00-05:00"

    from trainer.replay_engine import validate_point_in_time_inputs

    with pytest.raises(ReplayError, match=r"WINR.news\[0\]"):
        validate_point_in_time_inputs(future)
