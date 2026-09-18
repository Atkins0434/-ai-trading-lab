from __future__ import annotations
from trainer.output_paths import legacy_name

from pathlib import Path

from trainer.scout_audit import save_audit_reports
from trainer.replay_engine import load_historical_snapshot
from trainer.report_generator import save_scout_reports
from trainer.scout_engine import load_scout_config, run_scout


ROOT = Path(__file__).resolve().parent.parent

TRADING_DATE = "2018-01-02"

FIXTURE_PATH = (
    ROOT
    / "fixtures"
    / TRADING_DATE
    / legacy_name("historical_snapshot")
)


def main() -> None:
    fixture_config = load_scout_config()
    fixture_config["session"]["morning_freeze_time"] = "07:00:00"
    fixture_config["minimum_real_bars_60m"] = 0
    snapshot = load_historical_snapshot(
        FIXTURE_PATH,
        config=fixture_config,
    )

    scout_result = run_scout(
        snapshot,
        config=fixture_config,
    )

    scout_paths = save_scout_reports(
        scout_result,
        TRADING_DATE,
    )

    audit_paths = save_audit_reports(
        scout_result,
        TRADING_DATE,
    )

    print("Scout fixture run complete.")
    print(
        f"Scout Markdown: "
        f"{scout_paths['markdown']}"
    )
    print(
        f"Scout PDF: "
        f"{scout_paths['pdf']}"
    )
    print(
        f"Scout Audit JSON: "
        f"{audit_paths['json']}"
    )
    print(
        f"Scout Audit PDF: "
        f"{audit_paths['pdf']}"
    )


if __name__ == "__main__":
    main()
