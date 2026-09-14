from pathlib import Path

from trainer.replay_engine import load_historical_snapshot
from trainer.report_generator import save_scout_report
from trainer.scout_engine import run_scout


ROOT = Path(__file__).resolve().parent.parent

FIXTURE_PATH = (
    ROOT
    / "fixtures"
    / "2018-01-02"
    / "historical_snapshot.json"
)


def main() -> None:
    snapshot = load_historical_snapshot(FIXTURE_PATH)

    scout_result = run_scout(
        snapshot,
        threshold_pct=85.0,
    )

    report_path = save_scout_report(
        scout_result,
        trading_date=snapshot["trading_date"],
    )

    print(f"Scout report created: {report_path}")


if __name__ == "__main__":
    main()
