from pathlib import Path

from trainer.replay_engine import load_historical_snapshot
from trainer.report_generator import (
    candidate_status,
    generate_scout_markdown_report,
    generate_scout_pdf_report,
)
from trainer.research_scout_alpha import run_research_scout_alpha
from trainer.scout_engine import load_scout_config, run_scout


def test_visual_report_supports_research_alpha(tmp_path: Path):
    snapshot = load_historical_snapshot(
        Path("fixtures/2018-01-02/historical_snapshot.json"),
        config={"morning_freeze_time": "07:00:00"},
    )
    snapshot["freeze_timestamp"] = "2018-01-02T09:15:00-05:00"
    result = run_research_scout_alpha(snapshot)
    output = tmp_path / "alpha_report.pdf"

    generate_scout_pdf_report(result, output)

    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 5000
    from pypdf import PdfReader
    text = '\n'.join(page.extract_text() for page in PdfReader(output).pages)
    assert 'of 76 points' in text
    assert 'of 19 signals observed' in text
    assert 'Relative Strength Index 60 min' in text
    assert 'Premarket Dollar Volume Quality' in text
    assert 'Alpha12' in text
    assert candidate_status(result["candidates"][0]) in {
        "SELECTED", "REJECTED", "BELOW THRESHOLD", "NOT SCORABLE"
    }


def test_markdown_report_supports_not_scorable_candidate():
    config = load_scout_config()
    config["session"]["morning_freeze_time"] = "07:00:00"
    snapshot = load_historical_snapshot(
        Path("fixtures/2018-01-02/historical_snapshot.json"),
        config=config,
    )

    result = run_scout(snapshot, config=config)
    report = generate_scout_markdown_report(result)

    assert result["candidates"][0]["status"] == "NOT_SCORABLE"
    assert "NOT SCORABLE" in report
    assert "| N/A | N/A |" in report
