from __future__ import annotations
from trainer.output_paths import legacy_name

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ARTIFACT_NAMES = {
    legacy_name("research_alpha_batch_manifest"),
    "trainer_run_state.json",
    legacy_name("benchmark_result"),
    legacy_name("postmortem"),
}


def classification(
    payload: dict[str, Any], artifact_name: str | None = None
) -> list[str]:
    reasons = []
    if "universe_mode" not in payload:
        reasons.append("LEGACY_UNIVERSE_MODE_MISSING")
    elif payload["universe_mode"] == "ci_fixture":
        reasons.append("STATIC_SYMBOL_FIXTURE")
    if artifact_name == "trainer_run_state.json":
        days = payload.get("days", [])
        if any(day.get("universe_coverage") != "complete" for day in days):
            reasons.append("POINT_IN_TIME_COVERAGE_NOT_COMPLETE")
        if any(not day.get("universe_manifest_hash") for day in days):
            reasons.append("UNIVERSE_MANIFEST_HASH_MISSING")
    else:
        if "universe_manifest_hash" not in payload:
            reasons.append("UNIVERSE_MANIFEST_HASH_MISSING")
        if payload.get("universe_coverage") != "complete":
            reasons.append("POINT_IN_TIME_COVERAGE_NOT_COMPLETE")
    if payload.get("research_evidence") is not True:
        reasons.append("RESEARCH_EVIDENCE_NOT_TRUE")
    if payload.get("promotion_eligible") is not True:
        reasons.append("PROMOTION_ELIGIBILITY_NOT_TRUE")
    return sorted(set(reasons))


def quarantine_legacy_results(root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(root.rglob("*.json")):
        if path.name not in ARTIFACT_NAMES:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reasons = ["ARTIFACT_UNREADABLE"]
        else:
            reasons = classification(payload, path.name)
        if reasons:
            entries.append({
                "artifact": str(path.relative_to(root)),
                "classification": "ENGINEERING_FIXTURE",
                "research_evidence": False,
                "promotion_eligible": False,
                "strategy_performance_claim_allowed": False,
                "reason_codes": reasons,
            })
    result = {
        "version": "legacy_result_quarantine_v1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "entries": entries,
    }
    target = root / "LEGACY_RESEARCH_QUARANTINE.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a non-destructive quarantine registry for legacy replay artifacts."
    )
    parser.add_argument("root", nargs="?", type=Path, default=Path("reports"))
    args = parser.parse_args()
    print(json.dumps(quarantine_legacy_results(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
