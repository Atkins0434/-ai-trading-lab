from __future__ import annotations

from pathlib import Path

from trainer.audit_report import save_audit_reports
from trainer.replay_engine import load_historical_snapshot
from trainer.report_generator import save_scout_reports
from trainer.scout_engine import run_scout


ROOT = Path(__file__).resolve().parent.parent

TRADING_DATE = "2018-01-02"

FIXTURE_PATH = (
    ROOT
    / "fixtures"
    / TRADING_DATE
    / "historical_snapshot.json"
)


def main() -> None:
    snapshot = load_historical_snapshot(
        FIXTURE_PATH
    )

    scout_result = run_scout(
        snapshot
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
        f"Audit JSON: "
        f"{audit_paths['json']}"
    )
    print(
        f"Audit PDF: "
        f"{audit_paths['pdf']}"
    )


if __name__ == "__main__":
    main()
