from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


def candidate_status(candidate: dict[str, Any]) -> str:
    if candidate["selected"]:
        return "SELECTED"

    if not candidate["eligible"]:
        return "REJECTED"

    return "BELOW THRESHOLD"


def hard_reject_reason(candidate: dict[str, Any]) -> str:
    reasons = candidate.get("rejection_reasons", [])

    if not reasons:
        return "-"

    return ", ".join(reasons)


def ordered_candidates(
    scout_result: dict[str, Any],
) -> list[dict[str, Any]]:
    return sorted(
        scout_result["candidates"],
        key=lambda candidate: (
            candidate["rank"] is None,
            candidate["rank"]
            if candidate["rank"] is not None
            else 999999,
            -candidate["score_pct"],
            candidate["ticker"],
        ),
    )


def generate_scout_markdown_report(
    scout_result: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append("# Scout Report")
    lines.append("")
    lines.append(f"Replay ID: `{scout_result['replay_id']}`")
    lines.append(f"Scout Version: `{scout_result['scout_version']}`")
    lines.append(
        f"Snapshot Time: `{scout_result['snapshot_timestamp']}`"
    )
    lines.append(
        f"Selection Threshold: "
        f"{scout_result['scoring_threshold_pct']:.1f}%"
    )
    lines.append(
        f"Eligible Universe Count: "
        f"{scout_result.get('eligible_universe_count', 0)}"
    )
    lines.append(
        f"Qualifying Candidate Count: "
        f"{scout_result.get('qualifying_candidate_count', 0)}"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Rank | Ticker | Final Decision | Score | Spread % | "
        "Liquidity | Spread Guardrail | Reject Reason |"
    )
    lines.append(
        "|---:|---|---|---:|---:|---|---|---|"
    )

    for candidate in ordered_candidates(scout_result):
        liquidity = candidate["guardrails"]["liquidity"]
        spread = candidate["guardrails"]["spread"]

        lines.append(
            f"| {fmt(candidate['rank'])} "
            f"| {candidate['ticker']} "
            f"| {candidate_status(candidate)} "
            f"| {candidate['score_pct']:.2f}% "
            f"| {fmt(candidate['raw_values'].get('spread_pct'))} "
            f"| {liquidity['action']} "
            f"| {spread['action']} "
            f"| {hard_reject_reason(candidate)} |"
        )

    lines.append("")
    lines.append("## Candidate Detail")
    lines.append("")

    for candidate in ordered_candidates(scout_result):
        lines.append(
            f"### {candidate['ticker']} - "
            f"{candidate_status(candidate)}"
        )
        lines.append("")

        lines.append(f"- Rank: {fmt(candidate['rank'])}")
        lines.append(
            f"- Score: {candidate['score_pct']:.2f}% "
            f"({candidate['total_score']:.2f} / "
            f"{candidate['maximum_possible_score']:.2f})"
        )
        lines.append(f"- Eligible: {candidate['eligible']}")
        lines.append(f"- Selected: {candidate['selected']}")

        lines.append("")
        lines.append("#### Component Scores")
        lines.append("")

        for name, component in candidate["component_scores"].items():
            lines.append(
                f"- {name}: "
                f"{component['score']:.2f} / "
                f"{component['maximum_score']:.2f}"
            )

        lines.append("")
        lines.append("#### Guardrails")
        lines.append("")

        for name, result in candidate["guardrails"].items():
            lines.append(
                f"- {name}: {result['action']} "
                f"({result.get('reason_code')})"
            )

        lines.append("")
        lines.append("#### Raw Values")
        lines.append("")

        for name, value in candidate["raw_values"].items():
            lines.append(f"- {name}: {fmt(value)}")

        if candidate["rejection_reasons"]:
            lines.append("")
            lines.append("#### Rejection Reasons")
            lines.append("")

            for reason in candidate["rejection_reasons"]:
                lines.append(f"- {reason}")

        lines.append("")

    return "\n".join(lines)


def generate_scout_pdf_report(
    scout_result: dict[str, Any],
    pdf_path: Path,
) -> None:
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ScoutTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=14,
    )

    small_style = ParagraphStyle(
        "ScoutSmall",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
    )

    candidate_style = ParagraphStyle(
        "ScoutCandidate",
        parent=styles["Heading2"],
        spaceBefore=12,
        spaceAfter=6,
    )

    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
    )

    story = []

    story.append(Paragraph("Scout Report", title_style))

    metadata = [
        ("Replay ID", scout_result["replay_id"]),
        ("Scout Version", scout_result["scout_version"]),
        ("Snapshot Time", scout_result["snapshot_timestamp"]),
        (
            "Selection Threshold",
            f"{scout_result['scoring_threshold_pct']:.1f}%",
        ),
        (
            "Eligible Universe Count",
            scout_result.get("eligible_universe_count", 0),
        ),
        (
            "Qualifying Candidate Count",
            scout_result.get("qualifying_candidate_count", 0),
        ),
    ]

    for label, value in metadata:
        story.append(
            Paragraph(
                f"<b>{label}:</b> {value}",
                styles["BodyText"],
            )
        )

    story.append(Spacer(1, 14))
    story.append(Paragraph("Summary", styles["Heading2"]))

    table_data = [
        [
            "Rank",
            "Ticker",
            "Final Decision",
            "Score",
            "Spread",
            "Liquidity",
            "Reject Reason",
        ]
    ]

    for candidate in ordered_candidates(scout_result):
        table_data.append(
            [
                fmt(candidate["rank"]),
                candidate["ticker"],
                candidate_status(candidate),
                f"{candidate['score_pct']:.2f}%",
                (
                    f"{fmt(candidate['raw_values'].get('spread_pct'))}%"
                ),
                candidate["guardrails"]["liquidity"]["action"],
                hard_reject_reason(candidate),
            ]
        )

    summary_table = Table(
        table_data,
        colWidths=[
            0.40 * inch,
            0.55 * inch,
            1.15 * inch,
            0.65 * inch,
            0.65 * inch,
            0.70 * inch,
            1.85 * inch,
        ],
        repeatRows=1,
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 16))
    story.append(Paragraph("Candidate Detail", styles["Heading2"]))

    for candidate in ordered_candidates(scout_result):
        story.append(
            Paragraph(
                f"{candidate['ticker']} - "
                f"{candidate_status(candidate)}",
                candidate_style,
            )
        )

        story.append(
            Paragraph(
                f"Rank: {fmt(candidate['rank'])}",
                small_style,
            )
        )

        story.append(
            Paragraph(
                f"Score: {candidate['score_pct']:.2f}% "
                f"({candidate['total_score']:.2f} / "
                f"{candidate['maximum_possible_score']:.2f})",
                small_style,
            )
        )

        story.append(
            Paragraph(
                f"Eligible: {candidate['eligible']}",
                small_style,
            )
        )

        story.append(
            Paragraph(
                f"Selected: {candidate['selected']}",
                small_style,
            )
        )

        story.append(
            Paragraph(
                "<b>Component Scores</b>",
                small_style,
            )
        )

        for name, component in candidate["component_scores"].items():
            story.append(
                Paragraph(
                    f"- {name}: "
                    f"{component['score']:.2f} / "
                    f"{component['maximum_score']:.2f}",
                    small_style,
                )
            )

        story.append(
            Paragraph(
                "<b>Guardrails</b>",
                small_style,
            )
        )

        for name, result in candidate["guardrails"].items():
            story.append(
                Paragraph(
                    f"- {name}: {result['action']} "
                    f"({result.get('reason_code')})",
                    small_style,
                )
            )

        story.append(
            Paragraph(
                "<b>Raw Values</b>",
                small_style,
            )
        )

        for name, value in candidate["raw_values"].items():
            story.append(
                Paragraph(
                    f"- {name}: {fmt(value)}",
                    small_style,
                )
            )

        if candidate["rejection_reasons"]:
            story.append(
                Paragraph(
                    "<b>Rejection Reasons</b>",
                    small_style,
                )
            )

            for reason in candidate["rejection_reasons"]:
                story.append(
                    Paragraph(
                        f"- {reason}",
                        small_style,
                    )
                )

        story.append(Spacer(1, 8))

    document.build(story)


def save_scout_reports(
    scout_result: dict[str, Any],
    trading_date: str,
) -> dict[str, Path]:
    report_dir = ROOT / "reports" / trading_date
    report_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = report_dir / "scout_report.md"
    pdf_path = report_dir / "scout_report.pdf"

    markdown_path.write_text(
        generate_scout_markdown_report(scout_result),
        encoding="utf-8",
    )

    generate_scout_pdf_report(
        scout_result,
        pdf_path,
    )

    return {
        "markdown": markdown_path,
        "pdf": pdf_path,
    }


def save_scout_report(
    scout_result: dict[str, Any],
    trading_date: str,
) -> Path:
    """
    Backward-compatible wrapper.

    Existing code calling save_scout_report() continues to work,
    but the PDF is generated at the same time.
    """
    paths = save_scout_reports(
        scout_result,
        trading_date,
    )

    return paths["markdown"]
