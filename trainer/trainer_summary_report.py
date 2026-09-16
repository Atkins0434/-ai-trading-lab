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


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}%"


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
    capture = agg["realized_pnl_capture_pct"]
    cards = [
        [Paragraph('<font color="#FFFFFF"><b>DAYS</b></font>', center), Paragraph('<font color="#FFFFFF"><b>QUALIFYING RETURN</b></font>', center), Paragraph('<font color="#FFFFFF"><b>MONEY CAPTURE</b></font>', center), Paragraph('<font color="#FFFFFF"><b>QUALIFYING DRAWDOWN</b></font>', center)],
        [Paragraph(f"<b>{agg['days_processed']}</b>", center), Paragraph(f"<b>{agg['scout_cumulative_return_pct']:.2f}%</b>", center), Paragraph(f"<b>{capture:.1f}%</b>" if capture is not None else "N/A", center), Paragraph(f"<b>{agg['scout_max_drawdown_pct']:.2f}%</b>", center)],
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
        ["Measure", "Qualifying", "Exploration", "Combined", "Baseline"],
        ["Avg daily return", f"{agg['scout_average_daily_return_pct']:.3f}%", f"{agg['exploration_average_daily_return_pct']:.3f}%", f"{agg['combined_average_daily_return_pct']:.3f}%", f"{agg['benchmark_average_daily_return_pct']:.3f}%"],
        ["Cumulative return", f"{agg['scout_cumulative_return_pct']:.3f}%", f"{agg['exploration_cumulative_return_pct']:.3f}%", f"{agg['combined_cumulative_return_pct']:.3f}%", f"{agg['benchmark_cumulative_return_pct']:.3f}%"],
        ["Total realized P&L", f"${agg['scout_total_realized_pnl_usd']:,.2f}", f"${agg['exploration_total_realized_pnl_usd']:,.2f}", f"${agg['combined_total_realized_pnl_usd']:,.2f}", f"${agg['benchmark_total_realized_pnl_usd']:,.2f}"],
        ["Maximum drawdown", f"{agg['scout_max_drawdown_pct']:.3f}%", f"{agg['exploration_max_drawdown_pct']:.3f}%", f"{agg['combined_max_drawdown_pct']:.3f}%", f"{agg['benchmark_max_drawdown_pct']:.3f}%"],
        ["Daily volatility", f"{agg['scout_daily_return_stddev_pct']:.3f}%", f"{agg['exploration_daily_return_stddev_pct']:.3f}%", f"{agg['combined_daily_return_stddev_pct']:.3f}%", f"{agg['benchmark_daily_return_stddev_pct']:.3f}%"],
        ["Positive-day rate", f"{agg['scout_positive_day_rate_pct']:.1f}%", f"{agg['exploration_positive_day_rate_pct']:.1f}%", f"{agg['combined_positive_day_rate_pct']:.1f}%", f"{agg['benchmark_positive_day_rate_pct']:.1f}%"],
        ["Verdict count", f"{agg['scout_wins']}W / {agg['ties']}T / {agg['misses']}M", "Not graded", "Reference only", "Random draw"],
        ["Top-10 diagnostic", f"{agg['average_top_10_capture_rate_pct']:.1f}%", "Reported daily", "Reported daily", "No verdict effect"],
        ["Unreachable movers", str(agg["unreachable_mover_count"]), "-", "-", f"{agg['unreachable_pct']:.1f}%"],
        ["Execution reviews", str(agg["execution_policy_review_count"]), "Tracked separately", "Reference only", "No hypotheses"],
    ]
    story += [_table(comparison, [1.75 * inch, 1.38 * inch, 1.38 * inch, 1.38 * inch, 1.38 * inch], font_size=6.8), Spacer(1, 0.18 * inch)]
    story += [Paragraph("Only QUALIFYING_THRESHOLD candidates determine Scout return, P&amp;L, win rate, and the daily WIN/TIE/MISS verdict. Exploration is simulated after qualifying positions and is reported separately; combined performance is reference-only. Top-10 representation never controls the verdict.", body)]

    story += [PageBreak(), Paragraph("Daily replay ledger", title), Paragraph("Every row links one frozen historical decision to its persisted outcome, benchmark, and postmortem artifacts.", body), Spacer(1, 0.16 * inch)]
    day_rows = [["Date", "Partition", "Status", "Scored", "Unreachable", "Execution review"]]
    for item in state["days"]:
        day_rows.append([item["trading_date"], item["partition"], item["status"], str(item["scored_ticker_count"]), f"{item['unreachable_mover_count']} ({item['unreachable_pct']:.1f}%)", str(item["execution_policy_review_count"])])
    story += [_table(day_rows, [0.95 * inch, 1.15 * inch, 0.85 * inch, 0.75 * inch, 1.65 * inch, 1.95 * inch], font_size=7.2)]
    if state["failed_dates"]:
        story += [Spacer(1, 0.18 * inch), Paragraph("Failures requiring retry", h2)]
        for trading_date, error in sorted(state["failed_dates"].items()):
            story.append(Paragraph(f"<b>{trading_date}</b>: {error}", small))

    story += [PageBreak(), Paragraph("Cumulative hypotheses", title), Paragraph("Only development date/ticker occurrences train a hypothesis. Thirty development observations unlock validation only; they do not authorize a model change.", body), Spacer(1, 0.16 * inch)]
    hypothesis_rows = [["Hypothesis", "Dev evidence", "Status", "Production"]]
    for item in state["hypotheses"]:
        hypothesis_rows.append([Paragraph(item["hypothesis"], small), f"{item['independent_occurrence_count']} / {item['minimum_required_occurrences']}", item["status"], "LOCKED"])
    if len(hypothesis_rows) == 1:
        hypothesis_rows.append(["No missed-opportunity hypotheses recorded yet.", "0 / 30", "COLLECTING_EVIDENCE", "LOCKED"])
    story += [_table(hypothesis_rows, [3.6 * inch, 1.0 * inch, 1.55 * inch, 1.15 * inch], font_size=7.5), Spacer(1, 0.18 * inch), Paragraph("Candidate threshold reviews", h2)]
    threshold_rows = [["Metric", "0/1 frequency", "Misses", "Observed raw-value range"]]
    for item in state["candidate_threshold_reviews"]:
        raw_range = f"{item['observed_raw_value_min']} to {item['observed_raw_value_max']}"
        threshold_rows.append([item["metric_id"], str(item["low_score_frequency"]), str(item["missed_ticker_count"]), Paragraph(raw_range, small)])
    if len(threshold_rows) == 1:
        threshold_rows.append(["No visible low-score evidence", "0", "0", "-"])
    story += [_table(threshold_rows, [2.0 * inch, 1.2 * inch, 0.8 * inch, 3.3 * inch], font_size=7.2), Spacer(1, 0.18 * inch), Paragraph("Execution-policy review", h2)]
    execution_rows = [["Date", "Ticker", "Rank", "Realized", "Capturable", "Exit"]]
    for item in state["execution_policy_review"]:
        execution_rows.append([item["trading_date"], item["ticker"], str(item["benchmark_rank"]), _pct(item["realized_return_pct"]), _pct(item["maximum_capturable_move_pct"]), item["exit_reason"] or "-"])
    if len(execution_rows) == 1:
        execution_rows.append(["No execution-policy losses", "-", "-", "-", "-", "-"])
    story += [_table(execution_rows, [1.15 * inch, 0.85 * inch, 0.55 * inch, 1.1 * inch, 1.1 * inch, 2.55 * inch], font_size=7.2), Spacer(1, 0.22 * inch), Paragraph("Safety boundary", h2)]
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

    story += [PageBreak(), Paragraph("Cross-day feature evidence", title), Paragraph("D / V / H separates development, validation, and holdout observations. Only development evidence advances a feature toward validation; holdout data remains isolated unless explicitly authorized.", body), Spacer(1, 0.16 * inch)]
    feature_rows = [["Feature", "D / V / H", "Dev observed", "Dev positive", "Avg dev pts", "Status"]]
    for item in state["feature_evidence"]:
        average = "MISSING" if item["average_points"] is None else f"{item['average_points']:.2f}"
        partitions = (
            f"{item['independent_occurrence_count']} / "
            f"{item['validation_occurrence_count']} / "
            f"{item['holdout_occurrence_count']}"
        )
        feature_rows.append([item["metric_id"], partitions, str(item["observed_count"]), str(item["positive_count"]), average, item["status"]])
    if len(feature_rows) == 1:
        feature_rows.append(["No shadow evidence recorded", "0", "0", "0", "MISSING", "COLLECTING_EVIDENCE"])
    story += [_table(feature_rows, [1.75 * inch, 0.9 * inch, 0.9 * inch, 0.85 * inch, 0.85 * inch, 2.05 * inch], font_size=7.2), Spacer(1, 0.2 * inch), Paragraph("Dataset isolation", h2)]
    policy = state["dataset_policy"]
    partition_counts = {
        name: sum(value == name for value in policy["date_partitions"].values())
        for name in ("DEVELOPMENT", "VALIDATION", "HOLDOUT")
    }
    isolation = [
        ["Partition", "Dates", "Access"],
        ["Development", str(partition_counts["DEVELOPMENT"]), "Enabled" if "DEVELOPMENT" in policy["allowed_partitions"] else "Locked"],
        ["Validation", str(partition_counts["VALIDATION"]), "Enabled" if "VALIDATION" in policy["allowed_partitions"] else "Locked"],
        ["Holdout", str(partition_counts["HOLDOUT"]), "Explicitly unlocked" if policy["holdout_unlocked"] else "Locked"],
    ]
    story += [_table(isolation, [2.5 * inch, 1.2 * inch, 3.6 * inch])]
    doc.build(story)
    return output_path
