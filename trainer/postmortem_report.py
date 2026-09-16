from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}%"


def generate_postmortem_pdf(
    benchmark: dict[str, Any], postmortem: dict[str, Any], output_path: Path
) -> None:
    """Render a bounded visual postmortem; JSON remains the full audit record."""
    navy = colors.HexColor("#111827")
    blue = colors.HexColor("#2563EB")
    green = colors.HexColor("#DCFCE7")
    red = colors.HexColor("#FEE2E2")
    pale = colors.HexColor("#EFF6FF")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("PostTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=24, textColor=navy, spaceAfter=5)
    small = ParagraphStyle("PostSmall", parent=styles["BodyText"], fontSize=7, leading=9, textColor=navy)
    section = ParagraphStyle("PostSection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=14, textColor=navy, spaceBefore=10, spaceAfter=6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=landscape(letter), leftMargin=32, rightMargin=32, topMargin=28, bottomMargin=28, title="Scout Trainer Postmortem", author="AI Trading Lab")
    comparison = benchmark["comparison"]
    scout = benchmark["scout_summary"]
    bench = postmortem["benchmark_performance"]
    baselines = benchmark["return_baselines"]
    result_color = green if postmortem["result"] == "WIN" else red if postmortem["result"] == "MISS" else pale
    story: list[Any] = [
        Paragraph("Scout Trainer Postmortem", title),
        Paragraph(f"{postmortem['trading_date']} | {postmortem['scout_version']} | RESEARCH ONLY - no automatic promotion", small),
        Spacer(1, 8),
    ]
    kpis = [
        ("RESULT", postmortem["result"]),
        ("SCOUT RETURN", _pct(scout["realized_return_pct"])),
        ("RANDOM BASELINE", _pct(bench["realized_return_pct"])),
        ("RETURN GAP", _pct(comparison["return_difference_pct"])),
        ("TOP-10 CAPTURE", _pct(comparison["top_10_capture_rate_pct"])),
        ("UNREACHABLE", _pct(postmortem["unreachable_pct"])),
    ]
    kpi = Table([[Paragraph(f"<b>{label}</b><br/><font size='14'>{value}</font>", small) for label, value in kpis]], colWidths=[10.1 * inch / 6] * 6, rowHeights=[0.60 * inch])
    kpi.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), result_color), ("BACKGROUND", (1, 0), (-1, -1), pale), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    comparison_rows = [
        ["Performance", "Scout", "Random baseline"],
        ["Selected / draw size", scout["candidate_count"], baselines["random_draw_size"]],
        ["Realized P&L", f"${scout['realized_pnl_usd']:.2f}", f"${bench['realized_pnl_usd']:.2f}"],
        ["Net realized return", _pct(scout["realized_return_pct"]), _pct(bench["realized_return_pct"])],
        ["Eligible-ticker mean", "-", _pct(baselines["eligible_ticker_mean_realized_return_pct"])],
        ["Random draws / seed", "-", f"{baselines['random_draw_count']} / {baselines['random_seed']}"],
    ]
    comparison_table = Table(comparison_rows, colWidths=[4.1*inch,3.0*inch,3.0*inch], repeatRows=1)
    comparison_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),navy),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F9FAFB")]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.extend([
        kpi,
        Paragraph("Execution comparison", section),
        comparison_table,
        Spacer(1, 8),
        Paragraph(
            "WIN/TIE/MISS compares Scout net return with 200 deterministic random same-universe baskets under the same execution policy. Top-10 capture is diagnostic only.",
            small,
        ),
        PageBreak(),
    ])
    rows = [
        ["Same-universe top movers", "", "", "", "", "", "", ""],
        ["Rank", "Ticker", "MFE", "Simulated", "Return", "P&L", "Capture", "Exit"],
    ]
    for item in benchmark["benchmark_candidates"]:
        rows.append([item["benchmark_rank"], item["ticker"], _pct(item["raw_move_pct"]), "YES" if item["fair_simulation_possible"] else "NO", _pct(item["realized_return_pct"]), "N/A" if item["realized_pnl_usd"] is None else f"${item['realized_pnl_usd']:.2f}", "N/A" if item["capture_ratio"] is None else f"{item['capture_ratio']:.2f}x", item["exit_reason"] or item["simulation_exclusion_reason"] or "-"])
    table = Table(rows, colWidths=[0.48*inch,0.65*inch,0.75*inch,0.80*inch,0.82*inch,0.90*inch,0.78*inch,2.65*inch], repeatRows=2)
    table.setStyle(TableStyle([
        ("SPAN", (0,0),(-1,0)),
        ("FONTNAME",(0,0),(-1,1),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,0),12),
        ("TEXTCOLOR",(0,0),(-1,0),navy),
        ("BACKGROUND", (0,1),(-1,1),navy),
        ("TEXTCOLOR",(0,1),(-1,1),colors.white),
        ("FONTSIZE",(0,1),(-1,-1),7),
        ("GRID",(0,1),(-1,-1),0.35,colors.HexColor("#D1D5DB")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUNDS",(0,2),(-1,-1),[colors.white,colors.HexColor("#F9FAFB")]),
    ]))
    story.append(table)
    if len(postmortem["missed_opportunities"]) > 5:
        story.extend([
            PageBreak(),
            Paragraph("Missed-opportunity diagnosis", title),
            Paragraph(
                "Benchmark opportunities not represented in the Scout selection basket",
                small,
            ),
            Spacer(1, 8),
        ])
    else:
        story.append(Paragraph("Missed-opportunity diagnosis", section))
    missed_rows = [["Rank", "Ticker", "Scout score", "Classification", "Reason"]]
    for item in postmortem["missed_opportunities"][:10]:
        reason = ", ".join(item.get("failure_reason_codes", [])) or "-"
        missed_rows.append([item["benchmark_rank"], item["ticker"], _pct(item["scout_score_pct"]), item["miss_classification"].replace("_", " "), Paragraph(reason.replace("_", " ").title(), small)])
    missed = Table(missed_rows, colWidths=[0.55*inch,0.75*inch,0.95*inch,1.85*inch,6.0*inch], repeatRows=1)
    missed.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),blue),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#D1D5DB")),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.extend([missed, Spacer(1, 8), Paragraph("Trainer boundary: diagnoses are hypotheses only. No weights, thresholds, guardrails, or Production Scout files were changed.", small)])
    doc.build(story)
