from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle


NAVY = colors.HexColor("#12233F")
PALE_BLUE = colors.HexColor("#EAF2FF")
GREEN = colors.HexColor("#18865A")
RED = colors.HexColor("#B83A3A")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#667085")
GRID = colors.HexColor("#D7DDE8")
FONT = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"

pdfmetrics.registerFont(TTFont(FONT, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont(FONT_BOLD, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT_BOLD, italic=FONT, boldItalic=FONT_BOLD)


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.55 * inch, 0.35 * inch, "Scout Trainer - Research only - No automatic production changes")
    canvas.drawRightString(7.95 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _table(data, widths, *, header=True, font_size=8) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ]
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            style.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F8FAFC")))
    table.setStyle(TableStyle(style))
    return table


def generate_trainer_summary_pdf(state: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontName=FONT_BOLD, fontSize=22, leading=26, textColor=NAVY, alignment=TA_LEFT)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=14, leading=18, textColor=NAVY, spaceBefore=8, spaceAfter=8)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName=FONT, fontSize=9, leading=13, textColor=INK)
    small = ParagraphStyle("Small", parent=body, fontSize=7.5, leading=10)
    center = ParagraphStyle("Center", parent=body, alignment=TA_CENTER)

    doc = BaseDocTemplate(str(output_path), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.6 * inch, title="Scout Trainer Multi-Day Summary")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=_footer)])

    agg = state["aggregate_performance"]
    story = [
        Paragraph("Scout Trainer Multi-Day Summary", title),
        Paragraph(f"{state['run_id']} | {state['status']} | {state['mode']}", body),
        Paragraph(
            f"Selection policy: configured threshold plus top {state['selection_policy']['exploration_top_k']} guardrail-eligible names for research exploration",
            body,
        ),
        Spacer(1, 0.18 * inch),
    ]
    cards = [
        [Paragraph("DAYS", center), Paragraph("SCOUT P&L", center), Paragraph("BENCHMARK P&L", center), Paragraph("TOP-10 CAPTURE", center)],
        [Paragraph(f"<b>{agg['days_processed']}</b>", center), Paragraph(f"<b>${agg['scout_total_realized_pnl_usd']:,.2f}</b>", center), Paragraph(f"<b>${agg['benchmark_total_realized_pnl_usd']:,.2f}</b>", center), Paragraph(f"<b>{agg['average_top_10_capture_rate_pct']:.1f}%</b>", center)],
    ]
    card_table = Table(cards, colWidths=[1.83 * inch] * 4, rowHeights=[0.3 * inch, 0.52 * inch])
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), PALE_BLUE),
        ("GRID", (0, 0), (-1, -1), 1, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [card_table, Spacer(1, 0.2 * inch), Paragraph("Cumulative comparison", h2)]
    comparison = [
        ["Measure", "Scout", "Benchmark"],
        ["Average daily return", f"{agg['scout_average_daily_return_pct']:.3f}%", f"{agg['benchmark_average_daily_return_pct']:.3f}%"],
        ["Total realized P&L", f"${agg['scout_total_realized_pnl_usd']:,.2f}", f"${agg['benchmark_total_realized_pnl_usd']:,.2f}"],
        ["Daily result count", f"{agg['scout_wins']} wins / {agg['ties']} ties", f"{agg['misses']} Scout misses"],
        ["Top-10 representation", f"{agg['average_top_10_capture_rate_pct']:.1f}%", f"Target {agg['top_10_capture_target_pct']:.0f}%"],
    ]
    story += [_table(comparison, [2.8 * inch, 2.25 * inch, 2.25 * inch]), Spacer(1, 0.18 * inch)]
    target_color = GREEN if agg["top_10_capture_target_met"] else RED
    story += [Paragraph(f"<font color='{target_color.hexval()}'><b>Top-10 target {'met' if agg['top_10_capture_target_met'] else 'not yet met'}.</b></font> Evidence continues to accumulate without changing Production Scout.", body)]

    story += [PageBreak(), Paragraph("Daily replay ledger", title), Paragraph("Every row links one frozen historical decision to its persisted outcome, benchmark, and postmortem artifacts.", body), Spacer(1, 0.16 * inch)]
    day_rows = [["Date", "Status", "Scored", "Skipped", "Artifact directory"]]
    for item in state["days"]:
        day_rows.append([item["trading_date"], item["status"], str(item["scored_ticker_count"]), str(item["skipped_ticker_count"]), item["artifact_directory"]])
    story += [_table(day_rows, [1.05 * inch, 1.0 * inch, 0.7 * inch, 0.7 * inch, 3.85 * inch])]
    if state["failed_dates"]:
        story += [Spacer(1, 0.18 * inch), Paragraph("Failures requiring retry", h2)]
        for trading_date, error in sorted(state["failed_dates"].items()):
            story.append(Paragraph(f"<b>{trading_date}</b>: {error}", small))

    story += [PageBreak(), Paragraph("Cumulative hypotheses", title), Paragraph("Counts are independent date/ticker occurrences. Thirty observations unlock validation only; they do not authorize a model change.", body), Spacer(1, 0.16 * inch)]
    hypothesis_rows = [["Hypothesis", "Occurrences", "Status", "Production"]]
    for item in state["hypotheses"]:
        hypothesis_rows.append([Paragraph(item["hypothesis"], small), f"{item['independent_occurrence_count']} / {item['minimum_required_occurrences']}", item["status"], "LOCKED"])
    if len(hypothesis_rows) == 1:
        hypothesis_rows.append(["No missed-opportunity hypotheses recorded yet.", "0 / 30", "COLLECTING_EVIDENCE", "LOCKED"])
    story += [_table(hypothesis_rows, [3.6 * inch, 1.0 * inch, 1.55 * inch, 1.15 * inch], font_size=7.5), Spacer(1, 0.22 * inch), Paragraph("Safety boundary", h2)]
    controls = state["controls"]
    safeguards = [
        ["Control", "Enforced"],
        ["Real-money execution", "Disabled" if not controls["real_money_execution_allowed"] else "Enabled"],
        ["Production Scout mutation", "Disabled" if not controls["production_mutation_allowed"] else "Enabled"],
        ["Automatic promotion", "Disabled" if not controls["automatic_promotion_allowed"] else "Enabled"],
        ["Out-of-sample validation", "Required"],
        ["Blind holdout exam", "Required"],
        ["Final promotion", "Manual approval required"],
    ]
    story += [_table(safeguards, [3.65 * inch, 3.65 * inch])]
    doc.build(story)
    return output_path
