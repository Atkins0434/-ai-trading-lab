from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    PageBreak,
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
    selected = candidate.get("selected", candidate.get("research_selected", False))
    eligible = candidate.get("eligible", candidate.get("research_eligible", False))
    if selected:
        return "SELECTED"

    if not eligible:
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


def _candidate_selected(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("selected", candidate.get("research_selected", False)))


def _candidate_eligible(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("eligible", candidate.get("research_eligible", False)))


def _metric_label(metric_id: str) -> str:
    replacements = {"15m": "15 min", "30m": "30 min", "60m": "60 min"}
    label = metric_id.replace("_", " ").title()
    for source, target in replacements.items():
        label = label.replace(source.title(), target)
    return label


def _score_color(score: int | None):
    return {
        None: colors.HexColor("#E5E7EB"),
        0: colors.HexColor("#FECACA"),
        1: colors.HexColor("#FED7AA"),
        2: colors.HexColor("#FEF08A"),
        3: colors.HexColor("#BBF7D0"),
        4: colors.HexColor("#4ADE80"),
    }[score]


def _coverage(candidate: dict[str, Any]) -> tuple[int, int]:
    components = candidate["component_scores"].values()
    total = len(candidate["component_scores"])
    observed = sum(component["status"] == "OBSERVED" for component in components)
    return observed, total


def _draw_page(canvas, document) -> None:
    canvas.saveState()
    width, _ = landscape(letter)
    canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
    canvas.line(32, 24, width - 32, 24)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(32, 13, "AI Trading Lab - deterministic research report")
    canvas.drawRightString(width - 32, 13, f"Page {document.page}")
    canvas.restoreState()


def _score_bar(score_pct: float, width: float) -> Table:
    score_width = max(0.01, min(width - 0.01, score_pct / 100 * width))
    remaining = width - score_width
    bar = Table([["", ""]], colWidths=[score_width, remaining], rowHeights=[12])
    bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#2563EB")),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E5E7EB")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9CA3AF")),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return bar


def _raw_summary(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        parts = [f"{_metric_label(str(key))}: {_raw_summary(item)}" for key, item in list(value.items())[:3]]
        return "; ".join(parts)
    if isinstance(value, list):
        return f"{len(value)} values"
    return str(value)


def _metric_table(candidate: dict[str, Any], metric_ids: list[str]) -> Table:
    rows: list[list[Any]] = [["Signal", "Pts", "State"]]
    row_colors = []
    for row_index, metric_id in enumerate(metric_ids, start=1):
        component = candidate["component_scores"][metric_id]
        score = component["score"]
        rows.append([
            _metric_label(metric_id),
            "-" if score is None else f"{score}/4",
            component["status"],
        ])
        row_colors.append((row_index, _score_color(score)))
    table = Table(rows, colWidths=[2.55 * inch, 0.50 * inch, 0.85 * inch], repeatRows=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("LEADING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    commands.extend(("BACKGROUND", (1, row), (1, row), color) for row, color in row_colors)
    table.setStyle(TableStyle(commands))
    return table


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
                f"{fmt(component['score'])} / "
                f"{fmt(component['maximum_score'])} "
                f"({component['status']})"
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


def _generate_legacy_scout_pdf_report(
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
                    f"{fmt(component['score'])} / "
                    f"{fmt(component['maximum_score'])} "
                    f"({component['status']})",
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


def generate_scout_pdf_report(
    scout_result: dict[str, Any],
    pdf_path: Path,
) -> None:
    """Create a visual decision report while JSON remains the audit record."""
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#111827")
    blue = colors.HexColor("#2563EB")
    muted = colors.HexColor("#6B7280")
    pale_blue = colors.HexColor("#EFF6FF")
    pale_orange = colors.HexColor("#FFF7ED")
    selected_color = colors.HexColor("#DCFCE7")
    rejected_color = colors.HexColor("#FEE2E2")

    title_style = ParagraphStyle(
        "DashboardTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=25,
        textColor=navy,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DashboardSubtitle",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        textColor=muted,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "DashboardSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        textColor=navy,
        spaceBefore=9,
        spaceAfter=6,
    )
    detail_title_style = ParagraphStyle(
        "CandidateTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        textColor=navy,
        spaceAfter=4,
    )
    tiny_style = ParagraphStyle(
        "DashboardTiny",
        parent=styles["BodyText"],
        fontSize=7,
        leading=9,
        textColor=navy,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(letter),
        rightMargin=32,
        leftMargin=32,
        topMargin=30,
        bottomMargin=32,
        title="Scout Decision Report",
        author="AI Trading Lab",
    )
    candidates = ordered_candidates(scout_result)
    selected_count = sum(_candidate_selected(candidate) for candidate in candidates)
    rejected_count = sum(not _candidate_eligible(candidate) for candidate in candidates)
    observed_count = sum(
        component["status"] == "OBSERVED"
        for candidate in candidates
        for component in candidate["component_scores"].values()
    )
    metric_count = sum(len(candidate["component_scores"]) for candidate in candidates)
    coverage_pct = observed_count / metric_count * 100 if metric_count else 0.0
    mode = scout_result.get("mode", "PRODUCTION")
    research_only = mode == "RESEARCH_ONLY"

    story: list[Any] = [
        Paragraph("Scout Decision Report", title_style),
        Paragraph(
            f"{scout_result['snapshot_timestamp']}  |  {scout_result['scout_version']}  |  "
            f"Replay {scout_result['replay_id']}",
            subtitle_style,
        ),
    ]
    if research_only:
        banner = Table(
            [[Paragraph("RESEARCH ONLY - execution is disabled", tiny_style)]],
            colWidths=[10.1 * inch],
        )
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), pale_orange),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#FB923C")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([banner, Spacer(1, 7)])

    kpis = [
        ("UNIVERSE", len(candidates)),
        ("ELIGIBLE", scout_result.get("eligible_universe_count", 0)),
        ("SELECTED", selected_count),
        ("REJECTED", rejected_count),
        ("METRIC COVERAGE", f"{coverage_pct:.0f}%"),
        ("THRESHOLD", f"{scout_result['scoring_threshold_pct']:.0f}%"),
    ]
    kpi_table = Table(
        [[Paragraph(f"<b>{label}</b><br/><font size='15'>{value}</font>", tiny_style) for label, value in kpis]],
        colWidths=[10.1 * inch / len(kpis)] * len(kpis),
        rowHeights=[0.58 * inch],
    )
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pale_blue),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.extend([kpi_table, Paragraph("Ranked decision queue", section_style)])

    summary_rows: list[list[Any]] = [[
        "Rank", "Ticker", "Decision", "Score", "Points", "Coverage", "Guardrails", "Primary reason"
    ]]
    for candidate in candidates[:25]:
        observed, total = _coverage(candidate)
        actions = [result["action"] for result in candidate["guardrails"].values()]
        guardrail_summary = (
            "REJECT" if "REJECT" in actions else
            "NOT EVALUATED" if "NOT_EVALUATED" in actions else
            "PASS"
        )
        reason = next(iter(candidate.get("rejection_reasons", [])), "-")
        summary_rows.append([
            fmt(candidate["rank"]),
            candidate["ticker"],
            candidate_status(candidate),
            f"{candidate['score_pct']:.1f}%",
            f"{candidate['total_score']}/{candidate['maximum_possible_score']}",
            f"{observed}/{total}",
            guardrail_summary,
            Paragraph(reason.replace("_", " ").title(), tiny_style),
        ])
    summary_table = Table(
        summary_rows,
        colWidths=[0.42 * inch, 0.62 * inch, 1.05 * inch, 0.62 * inch, 0.72 * inch, 0.68 * inch, 1.02 * inch, 2.70 * inch],
        repeatRows=1,
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("LEADING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row, candidate in enumerate(candidates[:25], start=1):
        background = selected_color if _candidate_selected(candidate) else (
            rejected_color if not _candidate_eligible(candidate) else colors.white
        )
        commands.append(("BACKGROUND", (0, row), (-1, row), background))
    summary_table.setStyle(TableStyle(commands))
    story.append(summary_table)
    if len(candidates) > 25:
        story.append(Paragraph(
            f"Showing the top 25 of {len(candidates)} securities. The JSON audit retains every record.",
            subtitle_style,
        ))

    detail_candidates = sorted(
        candidates,
        key=lambda candidate: (
            not _candidate_selected(candidate),
            -candidate["score_pct"],
            candidate["ticker"],
        ),
    )[:10]
    for candidate in detail_candidates:
        story.append(PageBreak())
        observed, total = _coverage(candidate)
        story.append(Paragraph(
            f"{candidate['ticker']}  |  {candidate_status(candidate)}",
            detail_title_style,
        ))
        story.append(Paragraph(
            f"Rank {fmt(candidate['rank'])}  |  {candidate['total_score']} of "
            f"{candidate['maximum_possible_score']} points  |  {observed} of {total} signals observed",
            subtitle_style,
        ))
        story.append(_score_bar(candidate["score_pct"], 10.1 * inch))
        story.append(Paragraph(
            f"Score {candidate['score_pct']:.1f}%  |  Selection threshold "
            f"{scout_result['scoring_threshold_pct']:.1f}%",
            tiny_style,
        ))

        guardrail_cells = []
        guardrail_colors = []
        for column, (name, result) in enumerate(candidate["guardrails"].items()):
            action = result["action"]
            guardrail_cells.append(Paragraph(
                f"<b>{_metric_label(name)}</b><br/>{action.replace('_', ' ')}",
                tiny_style,
            ))
            guardrail_colors.append((column, {
                "PASS": selected_color,
                "REJECT": rejected_color,
                "PENALIZE": colors.HexColor("#FEF3C7"),
                "NOT_EVALUATED": colors.HexColor("#E5E7EB"),
            }.get(action, colors.white)))
        if guardrail_cells:
            guardrail_table = Table(
                [guardrail_cells],
                colWidths=[10.1 * inch / len(guardrail_cells)] * len(guardrail_cells),
                rowHeights=[0.45 * inch],
            )
            guardrail_commands = [
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#9CA3AF")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
            guardrail_commands.extend(
                ("BACKGROUND", (column, 0), (column, 0), color)
                for column, color in guardrail_colors
            )
            guardrail_table.setStyle(TableStyle(guardrail_commands))
            story.extend([Paragraph("Guardrails", section_style), guardrail_table])

        metric_ids = list(candidate["component_scores"])
        midpoint = (len(metric_ids) + 1) // 2
        left = _metric_table(candidate, metric_ids[:midpoint])
        right = _metric_table(candidate, metric_ids[midpoint:]) if metric_ids[midpoint:] else ""
        metric_layout = Table([[left, right]], colWidths=[4.95 * inch, 4.95 * inch])
        metric_layout.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (0, 0), 0),
            ("RIGHTPADDING", (0, 0), (0, 0), 5),
            ("LEFTPADDING", (1, 0), (1, 0), 5),
            ("RIGHTPADDING", (1, 0), (1, 0), 0),
        ]))
        story.extend([Paragraph("Signal map", section_style), metric_layout])

        strongest = sorted(
            (
                (metric_id, component)
                for metric_id, component in candidate["component_scores"].items()
                if component["status"] == "OBSERVED"
            ),
            key=lambda item: (-(item[1]["score"] or 0), item[0]),
        )[:5]
        if strongest:
            evidence_rows = [["Strongest observed signals", "Raw evidence", "Points"]]
            for metric_id, component in strongest:
                evidence_rows.append([
                    _metric_label(metric_id),
                    Paragraph(_raw_summary(component["raw_value"]), tiny_style),
                    f"{component['score']}/4",
                ])
            evidence_table = Table(
                evidence_rows,
                colWidths=[2.7 * inch, 6.65 * inch, 0.75 * inch],
                repeatRows=1,
            )
            evidence_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), pale_blue),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.extend([Spacer(1, 7), evidence_table])

        reasons = candidate.get("rejection_reasons", [])
        if reasons:
            story.append(Paragraph(
                "Decision notes: " + "; ".join(reason.replace("_", " ").title() for reason in reasons),
                subtitle_style,
            ))
        story.append(Paragraph(
            "Legend: green = stronger positive evidence; yellow/orange/red = 2/1/0 points; "
            "gray = missing. A zero is an observed result, not missing data.",
            tiny_style,
        ))

    document.build(story, onFirstPage=_draw_page, onLaterPages=_draw_page)


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
