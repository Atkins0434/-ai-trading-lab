from __future__ import annotations

from typing import Any

from trainer.evidence_eligibility import require_promotion_evidence


def validate_promotion_evidence(artifacts: list[dict[str, Any]]) -> bool:
    """Fail closed unless every input is complete point-in-time evidence."""
    require_promotion_evidence(artifacts)
    return True
