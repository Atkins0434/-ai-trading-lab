from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent


def _status(candidate: dict[str, Any]) -> str:
    if candidate["selected"]:
        return "SELECTED"

    if not candidate["eligible"]:
        return "REJECTED"

    return "BELOW THRESHOLD"


def _hash_result(scout_result: dict[str, Any]) -> str:
    """
    Produce a deterministic fingerprint of the Scout output.

    Same structured result = same hash.
    """
    canonical = json.dumps(
        scout_result,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def _ordered_candidates(
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


def generate_audit_json(
    scout_result: dict[str, Any],
) -> dict[str, Any]:
    candidates = []

    for candidate in _ordered_candidates(scout_result):
        candidates.append(
            {
                "ticker": candidate["ticker"],
                "final_decision": _status(candidate),
                "rank": candidate["rank"],
                "eligible": candidate["eligible"],
                "selected": candidate["selected"],
                "score_pct": candidate["score_pct"],
                "total_score": candidate["total_score"],
                "maximum_possible_score":
                    candidate["maximum_possible_score"],
                "threshold_points": candidate["threshold_points"],
                "component_scores":
                    candidate["component_scores"],
                "guardrails":
                    candidate["guardrails"],
                "raw_values":
                    candidate["raw_values"],
                "rejection_reasons":
                    candidate["rejection_reasons"],
            }
        )

    return {
        "audit_version": "scout_audit_v1.0",
        "replay_id": scout_result["replay_id"],
        "scout_version": scout_result["scout_version"],
        "snapshot_timestamp":
            scout_result["snapshot_timestamp"],
        "scoring_threshold_pct":
            scout_result["scoring_threshold_pct"],
        "eligible_universe_count":
            scout_result.get(
                "eligible_universe_count",
                0,
            ),
        "qualifying_candidate_count":
            scout_result.get(
                "qualifying_candidate_count",
                0,
            ),
        "determinism_hash":
            _hash_result(scout_result),
        "candidates": candidates,
    }


def generate_audit_pdf(
    audit: dict[str, Any],
    pdf_path: Path,
) -> None:
    styles = getSampleStyleSheet()

    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    story = []

    story.append(
        Paragraph(
            "Scout Audit",
            styles["Title"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Replay ID:</b> {audit['replay_id']}",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Scout Version:</b> "
            f"{audit['scout_version']}",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Snapshot Time:</b> "
            f"{audit['snapshot_timestamp']}",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Selection Threshold:</b> "
            f"{audit['scoring_threshold_pct']:.2f}%",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Determinism Hash:</b> "
            f"{audit['determinism_hash']}",
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 14))

    story.append(
        Paragraph(
            "Decision Summary",
            styles["Heading2"],
        )
    )

    summary_data = [
        [
            "Ticker",
            "Decision",
            "Score",
            "Eligible",
            "Selected",
        ]
    ]

    for candidate in audit["candidates"]:
        summary_data.append(
            [
                candidate["ticker"],
                candidate["final_decision"],
                f"{candidate['score_pct']:.2f}%",
                str(candidate["eligible"]),
                str(candidate["selected"]),
            ]
        )

    summary_table = Table(
        summary_data,
        colWidths=[
            0.9 * inch,
            1.6 * inch,
            0.9 * inch,
            0.9 * inch,
            0.9 * inch,
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
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 16))

    for candidate in audit["candidates"]:
        story.append(
            Paragraph(
                f"{candidate['ticker']} - "
                f"{candidate['final_decision']}",
                styles["Heading2"],
            )
        )

        story.append(
            Paragraph(
                "<b>Score Calculation</b>",
                styles["Heading3"],
            )
        )

        for name, component in (
            candidate["component_scores"].items()
        ):
            story.append(
                Paragraph(
                    f"{name}: "
                    f"{component['score'] if component['score'] is not None else 'MISSING'} / "
                    f"{component['maximum_score']} "
                    f"({component['status']})",
                    styles["BodyText"],
                )
            )

        story.append(
            Paragraph(
                f"<b>Total:</b> "
                f"{candidate['total_score']:.2f} / "
                f"{candidate['maximum_possible_score']:.2f} "
                f"= {candidate['score_pct']:.2f}%",
                styles["BodyText"],
            )
        )

        story.append(Spacer(1, 6))

        story.append(
            Paragraph(
                "<b>Guardrail Decisions</b>",
                styles["Heading3"],
            )
        )

        guardrail_data = [
            [
                "Guardrail",
                "Action",
                "Reason Code",
            ]
        ]

        for name, result in (
            candidate["guardrails"].items()
        ):
            guardrail_data.append(
                [
                    name,
                    result["action"],
                    result.get(
                        "reason_code",
                        "",
                    ),
                ]
            )

        guardrail_table = Table(
            guardrail_data,
            colWidths=[
                1.5 * inch,
                1.1 * inch,
                2.8 * inch,
            ],
            repeatRows=1,
        )

        guardrail_table.setStyle(
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
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                ]
            )
        )

        story.append(guardrail_table)
        story.append(Spacer(1, 6))

        story.append(
            Paragraph(
                "<b>Raw Inputs Seen by Scout</b>",
                styles["Heading3"],
            )
        )

        raw_data = [["Input", "Observed Value"]]

        for name, value in (
            candidate["raw_values"].items()
        ):
            raw_data.append(
                [
                    name,
                    str(value),
                ]
            )

        raw_table = Table(
            raw_data,
            colWidths=[
                2.7 * inch,
                2.7 * inch,
            ],
            repeatRows=1,
        )

        raw_table.setStyle(
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
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                ]
            )
        )

        story.append(raw_table)

        if candidate["rejection_reasons"]:
            story.append(Spacer(1, 6))
            story.append(
                Paragraph(
                    "<b>Rejection Reasons</b>",
                    styles["Heading3"],
                )
            )

            for reason in (
                candidate["rejection_reasons"]
            ):
                story.append(
                    Paragraph(
                        reason,
                        styles["BodyText"],
                    )
                )

        story.append(Spacer(1, 16))

    document.build(story)


def save_audit_reports(
    scout_result: dict[str, Any],
    trading_date: str,
) -> dict[str, Path]:
    report_dir = ROOT / "reports" / trading_date
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit = generate_audit_json(
        scout_result
    )

    json_path = (
        report_dir
        / "scout_audit.json"
    )

    pdf_path = (
        report_dir
        / "scout_audit.pdf"
    )

    json_path.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    generate_audit_pdf(
        audit,
        pdf_path,
    )

    return {
        "json": json_path,
        "pdf": pdf_path,
    }
