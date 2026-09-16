import pytest

from trainer.catalyst_events import (
    CatalystError,
    build_catalyst_snapshot,
    calculate_shadow_catalyst_metrics,
    normalize_catalyst_event,
)
from trainer.validate_contracts import validate_contract


FREEZE = "2026-09-14T07:00:00-04:00"


def raw_event(**overrides):
    event = {
        "ticker": "JOBY",
        "source_name": "Issuer newsroom",
        "source_reference": "release-123",
        "headline": "Company raises full-year guidance",
        "event_type": "GUIDANCE",
        "published_timestamp": "2026-09-14T05:30:00-04:00",
        "provider_available_timestamp": "2026-09-14T05:31:00-04:00",
        "source_tier": "PRIMARY",
        "verification_status": "VERIFIED_PRIMARY",
        "sentiment": "POSITIVE",
        "relevance_confidence": 0.95,
        "duplicate_group_id": None,
    }
    event.update(overrides)
    return event


def normalized(**overrides):
    return normalize_catalyst_event(
        raw_event(**overrides),
        source_provider="TEST_NEWS",
        feed_version="test_v1",
        ingested_timestamp="2026-09-15T12:00:00+00:00",
    )


def snapshot(events):
    return build_catalyst_snapshot(
        events,
        replay_id="2026-09-14-alpha",
        trading_date="2026-09-14",
        freeze_timestamp=FREEZE,
    )


def test_known_event_is_admitted_even_when_retrieved_later():
    result = snapshot([normalized()])
    validate_contract("catalyst_snapshot", result)
    assert result["summary"] == {"received": 1, "admitted": 1, "excluded": 0}
    assert result["events"][0]["admission_reason"] == "KNOWN_BY_FREEZE"


def test_provider_availability_after_freeze_is_excluded():
    result = snapshot([
        normalized(provider_available_timestamp="2026-09-14T07:00:01-04:00")
    ])
    assert result["events"][0]["admission_reason"] == "AVAILABLE_AFTER_FREEZE"


def test_unknown_provider_availability_is_not_assumed_from_publication():
    result = snapshot([normalized(provider_available_timestamp=None)])
    assert result["events"][0]["admission_reason"] == "UNKNOWN_PROVIDER_AVAILABILITY"


def test_publication_after_freeze_is_excluded():
    result = snapshot([
        normalized(
            published_timestamp="2026-09-14T07:00:01-04:00",
            provider_available_timestamp="2026-09-14T07:00:02-04:00",
        )
    ])
    assert result["events"][0]["admission_reason"] == "PUBLISHED_AFTER_FREEZE"


def test_syndicated_duplicates_count_once():
    first = normalized(duplicate_group_id="guidance-123")
    second_raw = raw_event(
        source_name="News wire",
        source_reference="wire-456",
        duplicate_group_id="guidance-123",
        source_tier="HIGH_QUALITY_WIRE",
        verification_status="SINGLE_SOURCE",
    )
    second = normalize_catalyst_event(
        second_raw,
        source_provider="SECOND_PROVIDER",
        feed_version="wire_v1",
        ingested_timestamp="2026-09-15T12:00:00+00:00",
    )
    result = snapshot([first, second])
    assert result["summary"] == {"received": 2, "admitted": 1, "excluded": 1}
    assert {event["admission_reason"] for event in result["events"]} == {
        "KNOWN_BY_FREEZE",
        "SYNDICATED_DUPLICATE",
    }
    admitted = next(
        event for event in result["events"] if event["admission_status"] == "ADMITTED"
    )
    assert admitted["verification_status"] == "VERIFIED_PRIMARY"


def test_shadow_metrics_do_not_change_alpha_score():
    result = calculate_shadow_catalyst_metrics(snapshot([normalized()]), "JOBY")
    assert [item["points"] for item in result["components"]] == [4, 4, 4]
    assert all(not item["included_in_alpha_score"] for item in result["components"])
    assert result["mode"] == "SHADOW_ONLY"
    assert result["production_score_changed"] is False


def test_observed_negative_catalyst_scores_zero_not_missing():
    event = normalized(sentiment="NEGATIVE")
    result = calculate_shadow_catalyst_metrics(snapshot([event]), "JOBY")
    assert all(item["status"] == "OBSERVED" for item in result["components"])
    assert all(item["points"] == 0 for item in result["components"])


def test_no_admitted_catalyst_stays_missing_not_zero():
    result = calculate_shadow_catalyst_metrics(snapshot([]), "JOBY")
    assert all(item["status"] == "MISSING" for item in result["components"])
    assert all(item["points"] is None for item in result["components"])


def test_unverified_positive_catalyst_scores_zero():
    event = normalized(verification_status="UNVERIFIED")
    result = calculate_shadow_catalyst_metrics(snapshot([event]), "JOBY")
    assert all(item["points"] == 0 for item in result["components"])


def test_dilution_is_exposed_as_guardrail_candidate_only():
    event = normalized(event_type="OFFERING_DILUTION", sentiment="NEGATIVE")
    result = calculate_shadow_catalyst_metrics(snapshot([event]), "JOBY")
    assert result["dilution_guardrail_candidate"] is True
    assert result["production_score_changed"] is False


def test_snapshot_is_deterministic_across_input_order():
    first = normalized()
    second = normalized(
        ticker="AAL",
        headline="Airline updates quarterly guidance",
        source_reference="release-789",
    )
    assert snapshot([first, second])["determinism_hash"] == snapshot(
        [second, first]
    )["determinism_hash"]


def test_naive_timestamp_is_rejected():
    with pytest.raises(CatalystError, match="timezone information"):
        normalized(published_timestamp="2026-09-14T05:30:00")


def test_invalid_confidence_is_rejected():
    with pytest.raises(CatalystError, match="0 to 1"):
        normalized(relevance_confidence=1.2)
