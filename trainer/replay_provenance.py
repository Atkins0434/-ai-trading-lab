"""Shared resume identity for the engine, output caches, and artifact restores."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def current_provenance(comparison_policy_paths=None) -> dict:
    # Keep module import usable before optional replay dependencies are installed.
    # trainer.orb reaches contract validation/jsonschema through execution costs.
    from trainer.orb import orb_config_hash

    def config(name):
        return json.loads((ROOT / "config" / name).read_text())

    scout = config("scout_alpha_v1.json")
    registry = config("feature_registry_alpha_v1.json")
    if comparison_policy_paths is None:
        comparison_policy_paths = config("flatfile_replay.json")["comparison_policy_paths"]
    # Read the checkout, not the triggering event SHA (which may be a PR ref).
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {
        "scout_id": scout["scout_id"],
        "rubric_version": scout["rubric_version"],
        "feature_registry_id": registry["feature_registry_id"],
        "metric_count": registry["metric_count"],
        "dataset_version": config("replay_backlog.json")["dataset_version"],
        "comparison_policy_paths": [str(path) for path in comparison_policy_paths],
        "cost_model_id": config("execution_costs.json")["cost_model_id"],
        "orb_config_hash": orb_config_hash(),
        "git_sha": git_sha,
    }


def provenance_differences(manifest: dict, current: dict) -> list[str]:
    restored = manifest.get("provenance")
    restored = restored if isinstance(restored, dict) else {}
    differences = sorted(
        key for key in set(restored) | set(current)
        if key not in restored or key not in current or restored[key] != current[key]
    )
    # Debug-forced outputs remain tainted even after their metadata is updated.
    if manifest.get("forced_resume"):
        differences.append("forced_resume")
    return differences


def check_resume(output_root: Path, manifest: dict, current: dict, *, force=False) -> bool:
    """Archive mismatched outputs before any writer can touch them."""
    differences = provenance_differences(manifest, current)
    if not differences:
        return True
    message = "[flatfile_replay] provenance mismatch: " + ", ".join(differences)
    if force:
        print(message + "; DEBUG force-resume enabled", file=sys.stderr, flush=True)
        return True
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stale = output_root.with_name(f"{output_root.name}.stale-{timestamp}")
    output_root.rename(stale)
    print(message + f"; archived to {stale}; starting fresh", file=sys.stderr, flush=True)
    return False


def main() -> None:
    provenance = current_provenance()
    for key in ("scout_id", "metric_count", "dataset_version"):
        print(f"{key}={provenance[key]}")
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    print(f"provenance_key={digest}")


if __name__ == "__main__":
    main()
