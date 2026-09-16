from __future__ import annotations

from typing import Any, Iterable


EVIDENCE_FIELDS = (
    "universe_mode",
    "universe_manifest_hash",
    "universe_coverage",
    "research_evidence",
    "promotion_eligible",
)


class EvidenceEligibilityError(Exception):
    """Raised when an artifact cannot be admitted as research evidence."""


def artifact_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in EVIDENCE_FIELDS if field not in artifact]
    if missing:
        raise EvidenceEligibilityError(
            f"Artifact is missing universe evidence fields: {missing}"
        )
    return {field: artifact[field] for field in EVIDENCE_FIELDS}


def assert_matching_universe(*artifacts: dict[str, Any]) -> dict[str, Any]:
    if not artifacts:
        raise EvidenceEligibilityError("At least one artifact is required.")
    expected = artifact_metadata(artifacts[0])
    for artifact in artifacts[1:]:
        actual = artifact_metadata(artifact)
        if actual["universe_manifest_hash"] != expected["universe_manifest_hash"]:
            raise EvidenceEligibilityError(
                "Scout and benchmark inputs use different universe manifests."
            )
        if actual["universe_mode"] != expected["universe_mode"]:
            raise EvidenceEligibilityError("Universe modes do not match.")
    return expected


def require_research_evidence(
    artifact: dict[str, Any], *, consumer: str
) -> dict[str, Any]:
    metadata = artifact_metadata(artifact)
    failures = []
    if metadata["universe_mode"] != "historical_research":
        failures.append("UNIVERSE_MODE_NOT_HISTORICAL_RESEARCH")
    if metadata["universe_coverage"] != "complete":
        failures.append("HISTORICAL_UNIVERSE_COVERAGE_INCOMPLETE")
    if metadata["research_evidence"] is not True:
        failures.append("RESEARCH_EVIDENCE_FALSE")
    if failures:
        raise EvidenceEligibilityError(
            f"{consumer} rejected artifact: {', '.join(failures)}"
        )
    return metadata


def require_promotion_evidence(artifacts: Iterable[dict[str, Any]]) -> None:
    found = False
    for artifact in artifacts:
        found = True
        metadata = require_research_evidence(artifact, consumer="Promotion")
        if metadata["promotion_eligible"] is not True:
            raise EvidenceEligibilityError(
                "Promotion rejected artifact: PROMOTION_ELIGIBLE_FALSE"
            )
    if not found:
        raise EvidenceEligibilityError("Promotion requires evidence artifacts.")
