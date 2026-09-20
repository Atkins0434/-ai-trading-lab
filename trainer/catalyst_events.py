from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
import json
import re
from typing import Any, Iterable


class CatalystError(Exception):
    """Raised when catalyst data cannot be admitted safely."""


EVENT_TYPES = {
    "EARNINGS",
    "GUIDANCE",
    "SEC_FILING",
    "FDA_REGULATORY",
    "MERGER_ACQUISITION",
    "CONTRACT_AWARD",
    "PRODUCT",
    "ANALYST_ACTION",
    "OFFERING_DILUTION",
    "MANAGEMENT",
    "LEGAL_REGULATORY",
    "MACRO_SECTOR",
    "OTHER",
}
SOURCE_TIERS = {"PRIMARY", "HIGH_QUALITY_WIRE", "OTHER"}
VERIFICATION_STATUSES = {
    "VERIFIED_PRIMARY",
    "VERIFIED_MULTI_SOURCE",
    "SINGLE_SOURCE",
    "UNVERIFIED",
}
SENTIMENTS = {"POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED", "UNKNOWN"}

QUALITY_POINTS = {
    "EARNINGS": 4,
    "GUIDANCE": 4,
    "SEC_FILING": 4,
    "FDA_REGULATORY": 4,
    "MERGER_ACQUISITION": 4,
    "CONTRACT_AWARD": 3,
    "PRODUCT": 2,
    "ANALYST_ACTION": 2,
    "MANAGEMENT": 2,
    "LEGAL_REGULATORY": 2,
    "MACRO_SECTOR": 1,
    "OTHER": 1,
    "OFFERING_DILUTION": 0,
}
VERIFICATION_POINTS = {
    "VERIFIED_PRIMARY": 4,
    "VERIFIED_MULTI_SOURCE": 4,
    "SINGLE_SOURCE": 1,
    "UNVERIFIED": 0,
}


def _parse_aware(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise CatalystError(f"{field} must be valid ISO-8601.") from exc
    if parsed.tzinfo is None:
        raise CatalystError(f"{field} must contain timezone information.")
    return parsed


def _required_text(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CatalystError(f"{field} is required.")
    return value.strip()


def _headline_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _duplicate_key(event: dict[str, Any]) -> str:
    return event.get("duplicate_group_id") or _canonical_hash(
        {
            "ticker": event["ticker"],
            "event_type": event["event_type"],
            "headline": _headline_key(event["headline"]),
        }
    )


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def normalize_catalyst_event(
    raw: dict[str, Any],
    *,
    source_provider: str,
    feed_version: str,
    ingested_timestamp: str,
) -> dict[str, Any]:
    """Convert one provider record into the provider-neutral event contract."""
    ticker = _required_text(raw, "ticker").upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", ticker):
        raise CatalystError(f"Invalid ticker: {ticker}")

    event_type = _required_text(raw, "event_type").upper()
    source_tier = _required_text(raw, "source_tier").upper()
    verification = _required_text(raw, "verification_status").upper()
    sentiment = _required_text(raw, "sentiment").upper()
    if event_type not in EVENT_TYPES:
        raise CatalystError(f"Unknown event_type: {event_type}")
    if source_tier not in SOURCE_TIERS:
        raise CatalystError(f"Unknown source_tier: {source_tier}")
    if verification not in VERIFICATION_STATUSES:
        raise CatalystError(f"Unknown verification_status: {verification}")
    if sentiment not in SENTIMENTS:
        raise CatalystError(f"Unknown sentiment: {sentiment}")

    published = _required_text(raw, "published_timestamp")
    _parse_aware(published, "published_timestamp")
    available = raw.get("provider_available_timestamp")
    if available is not None:
        _parse_aware(available, "provider_available_timestamp")
    _parse_aware(ingested_timestamp, "ingested_timestamp")

    relevance = raw.get("relevance_confidence")
    if isinstance(relevance, bool) or not isinstance(relevance, (int, float)):
        raise CatalystError("relevance_confidence must be a number from 0 to 1.")
    relevance = float(relevance)
    if not 0 <= relevance <= 1:
        raise CatalystError("relevance_confidence must be a number from 0 to 1.")

    headline = _required_text(raw, "headline")
    identity = {
        "ticker": ticker,
        "source_provider": source_provider,
        "source_reference": raw.get("source_reference"),
        "published_timestamp": published,
        "headline_key": _headline_key(headline),
    }
    return {
        "event_id": f"evt_{_canonical_hash(identity)[:24]}",
        "ticker": ticker,
        "source_provider": source_provider,
        "feed_version": feed_version,
        "source_name": _required_text(raw, "source_name"),
        "source_reference": raw.get("source_reference"),
        "headline": headline,
        "event_type": event_type,
        "published_timestamp": published,
        "provider_available_timestamp": available,
        "ingested_timestamp": ingested_timestamp,
        "source_tier": source_tier,
        "verification_status": verification,
        "sentiment": sentiment,
        "relevance_confidence": relevance,
        "duplicate_group_id": raw.get("duplicate_group_id"),
    }


def _admit(event: dict[str, Any], freeze_timestamp: str, policy: dict[str, Any] | None = None) -> tuple[str, str]:
    freeze = _parse_aware(freeze_timestamp, "freeze_timestamp")
    published = _parse_aware(event["published_timestamp"], "published_timestamp")
    available_raw = event.get("provider_available_timestamp")
    if published > freeze:
        return "EXCLUDED", "PUBLISHED_AFTER_FREEZE"
    if policy and published < freeze - timedelta(days=policy['lookback_days']):
        return "EXCLUDED", "OUTSIDE_LOOKBACK"
    if available_raw is None:
        rules = (policy or {}).get('admission', {})
        if rules.get('unknown_provider_availability_action') != 'ASSUME_AVAILABLE_WITH_LATENCY':
            return "EXCLUDED", "UNKNOWN_PROVIDER_AVAILABILITY"
        assumed = published + timedelta(minutes=rules['provider_latency_minutes'])
        return ("ADMITTED", "ASSUMED_AVAILABLE_WITH_LATENCY") if assumed <= freeze else ("EXCLUDED", "AVAILABLE_AFTER_FREEZE")
    available = _parse_aware(available_raw, "provider_available_timestamp")
    if available < published:
        return "EXCLUDED", "AVAILABILITY_PRECEDES_PUBLICATION"
    if available > freeze:
        return "EXCLUDED", "AVAILABLE_AFTER_FREEZE"
    return "ADMITTED", "KNOWN_BY_FREEZE"


def build_catalyst_snapshot(
    events: Iterable[dict[str, Any]],
    *,
    replay_id: str,
    trading_date: str,
    freeze_timestamp: str,
    policy_version: str = "catalyst_alpha_v1.0",
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Admit, deduplicate, and freeze normalized catalyst events."""
    _parse_aware(freeze_timestamp, "freeze_timestamp")
    evaluated: list[dict[str, Any]] = []
    admitted_by_key: dict[str, dict[str, Any]] = {}

    verification_rank = {
        "VERIFIED_PRIMARY": 4,
        "VERIFIED_MULTI_SOURCE": 3,
        "SINGLE_SOURCE": 2,
        "UNVERIFIED": 1,
    }
    source_rank = {"PRIMARY": 3, "HIGH_QUALITY_WIRE": 2, "OTHER": 1}
    ordered = sorted(
        events,
        key=lambda item: (
            item["ticker"],
            _duplicate_key(item),
            -verification_rank[item["verification_status"]],
            -source_rank[item["source_tier"]],
            item["event_id"],
        ),
    )
    for original in ordered:
        event = dict(original)
        status, reason = _admit(event, freeze_timestamp, policy)
        event["admission_status"] = status
        event["admission_reason"] = reason
        event["duplicate_of"] = None
        if status == "ADMITTED":
            duplicate_key = _duplicate_key(event)
            if duplicate_key in admitted_by_key:
                event["admission_status"] = "EXCLUDED"
                event["admission_reason"] = "SYNDICATED_DUPLICATE"
                event["duplicate_of"] = admitted_by_key[duplicate_key]["event_id"]
            else:
                admitted_by_key[duplicate_key] = event
        evaluated.append(event)

    admitted = sum(e["admission_status"] == "ADMITTED" for e in evaluated)
    snapshot = {
        "contract_version": "catalyst_snapshot_v1.0",
        "replay_id": replay_id,
        "trading_date": trading_date,
        "freeze_timestamp": freeze_timestamp,
        "timezone": "America/New_York",
        "policy_version": policy_version,
        "shadow_mode": policy is None or policy.get("mode") != "SCORED_RESEARCH",
        "events": evaluated,
        "summary": {
            "received": len(evaluated),
            "admitted": admitted,
            "excluded": len(evaluated) - admitted,
        },
    }
    snapshot["determinism_hash"] = _canonical_hash(snapshot)
    return snapshot


def _freshness_points(event: dict[str, Any], freeze_timestamp: str) -> int:
    age_hours = (
        _parse_aware(freeze_timestamp, "freeze_timestamp")
        - _parse_aware(event["published_timestamp"], "published_timestamp")
    ).total_seconds() / 3600
    if age_hours <= 2:
        return 4
    if age_hours <= 6:
        return 3
    if age_hours <= 12:
        return 2
    if age_hours <= 24:
        return 1
    return 0


def calculate_shadow_catalyst_metrics(
    snapshot: dict[str, Any], ticker: str
) -> dict[str, Any]:
    """Calculate metrics 19-21 without changing Alpha or Production scores."""
    observed = [
        event
        for event in snapshot["events"]
        if event["ticker"] == ticker.upper()
        and event["admission_status"] == "ADMITTED"
    ]
    candidates = [
        event
        for event in observed
        if event["sentiment"] == "POSITIVE"
        and event["relevance_confidence"] >= 0.5
        and event["verification_status"]
        in {"VERIFIED_PRIMARY", "VERIFIED_MULTI_SOURCE"}
    ]
    dilution = any(
        event["ticker"] == ticker.upper()
        and event["admission_status"] == "ADMITTED"
        and event["event_type"] == "OFFERING_DILUTION"
        for event in snapshot["events"]
    )
    if not observed:
        component = lambda metric: {
            "metric": metric,
            "status": "MISSING",
            "points": None,
            "included_in_alpha_score": False,
        }
        components = [
            component("catalyst_quality"),
            component("catalyst_verification_confidence"),
            component("catalyst_freshness_relevance"),
        ]
    elif not candidates:
        components = [
            {
                "metric": metric,
                "status": "OBSERVED",
                "points": 0,
                "included_in_alpha_score": False,
            }
            for metric in (
                "catalyst_quality",
                "catalyst_verification_confidence",
                "catalyst_freshness_relevance",
            )
        ]
    else:
        best_quality = max(QUALITY_POINTS[event["event_type"]] for event in candidates)
        best_verification = max(
            VERIFICATION_POINTS[event["verification_status"]] for event in candidates
        )
        best_freshness = max(
            _freshness_points(event, snapshot["freeze_timestamp"])
            for event in candidates
        )
        values = [
            ("catalyst_quality", best_quality),
            ("catalyst_verification_confidence", best_verification),
            ("catalyst_freshness_relevance", best_freshness),
        ]
        components = [
            {
                "metric": metric,
                "status": "OBSERVED",
                "points": points,
                "included_in_alpha_score": False,
            }
            for metric, points in values
        ]

    return {
        "ticker": ticker.upper(),
        "mode": "SHADOW_ONLY",
        "components": components,
        "dilution_guardrail_candidate": dilution,
        "production_score_changed": False,
    }


def calculate_research_catalyst_metrics(
    snapshot: dict[str, Any], ticker: str, policy: dict[str, Any]
) -> dict[str, Any]:
    """Score research catalysts, distinguishing an empty query from a failure."""
    events = [
        event for event in snapshot["events"]
        if event["ticker"] == ticker and event["admission_status"] == "ADMITTED"
    ]
    dilution = next(
        (event for event in events if event["event_type"] == "OFFERING_DILUTION"),
        None,
    )
    best = dilution or max(
        events,
        key=lambda event: (
            QUALITY_POINTS[event["event_type"]],
            VERIFICATION_POINTS[event["verification_status"]],
            _parse_aware(event["published_timestamp"], "published_timestamp"),
            event["event_id"],
        ),
        default=None,
    )
    verification = max(
        (event["verification_status"] for event in events),
        key=lambda value: (VERIFICATION_POINTS[value], value == "VERIFIED_PRIMARY"),
        default="UNVERIFIED",
    )
    relevant = [
        event for event in events
        if event.get("positive_relevance", 0) >= policy["positive_relevance_minimum"]
    ]
    freshest = max(
        relevant,
        key=lambda event: _parse_aware(event["published_timestamp"], "published_timestamp"),
        default=None,
    )
    age = None
    if freshest:
        age = (
            _parse_aware(snapshot["freeze_timestamp"], "freeze_timestamp")
            - _parse_aware(freshest["published_timestamp"], "published_timestamp")
        ).total_seconds() / 60
    freshness = 0
    for score in range(4, 0, -1):
        key = f"{score}_points_max" if score > 1 else "1_point_max"
        if age is not None and age <= policy["freshness_hours"][key] * 60:
            freshness = score
            break
    values = {
        "catalyst_quality": (
            best["event_type"] if best else None,
            QUALITY_POINTS[best["event_type"]] if best else 0,
        ),
        "catalyst_verification_confidence": (verification, VERIFICATION_POINTS[verification]),
        "catalyst_freshness_relevance": (age, freshness),
    }
    components = {
        metric: {
            "status": "SCORED" if events else "NO_CATALYST",
            "score": score,
            "raw_value": raw,
            "maximum_score": 4,
            "as_of_timestamp": snapshot["freeze_timestamp"],
            "reason_code": "ADMITTED_CATALYST" if events else "NO_ADMITTED_CATALYST",
            "calculation_version": "research_catalyst_v1.0",
        }
        for metric, (raw, score) in values.items()
    }
    return {
        "components": components,
        "admitted_event_count": len(events),
        "best_event_type": best["event_type"] if best else None,
        "source_tier": best["source_tier"] if best else None,
        "verification_status": verification,
        "freshest_event_age_minutes": age,
        "sentiment": best["sentiment"] if best else None,
    }
