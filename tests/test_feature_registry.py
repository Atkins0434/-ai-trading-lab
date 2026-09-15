import json
from pathlib import Path

from trainer.scout_engine import load_feature_registry


ROOT = Path(__file__).resolve().parent.parent


def test_registry_is_exactly_30_metrics_and_120_points():
    registry = load_feature_registry()

    assert registry["metric_count"] == 30
    assert registry["points_per_metric"] == 4
    assert registry["maximum_points"] == 120
    assert [item["number"] for item in registry["metrics"]] == list(
        range(1, 31)
    )
    assert len({item["id"] for item in registry["metrics"]}) == 30


def test_scout_config_uses_fixed_denominator():
    config = json.loads(
        (ROOT / "config" / "scout_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["scoring"]["metric_count"] == 30
    assert config["scoring"]["maximum_points"] == 120
    assert config["scoring"]["denominator_policy"] == "FIXED_120"
    assert "dynamic_denominator" not in config["scoring"]
