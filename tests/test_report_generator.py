from pathlib import Path

from trainer.replay_engine import load_historical_snapshot
from trainer.report_generator import candidate_status, generate_scout_pdf_report
from trainer.research_scout_alpha import run_research_scout_alpha


def test_visual_report_supports_research_alpha(tmp_path: Path):
    snapshot = load_historical_snapshot(
        Path("fixtures/2018-01-02/historical_snapshot.json")
    )
    result = run_research_scout_alpha(snapshot)
    output = tmp_path / "alpha_report.pdf"

    generate_scout_pdf_report(result, output)

    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 5000
    assert candidate_status(result["candidates"][0]) in {
        "SELECTED", "REJECTED", "BELOW THRESHOLD"
    }
