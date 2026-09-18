from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from trainer.universe_manifest import (
    DEFINITIVE,
    GAP,
    EXCLUSION_REASON_CLASSIFICATION,
)
from trainer.validate_contracts import load_json
from trainer.scorable_outcomes import concatenate_completed_outcomes


ROOT = Path(__file__).resolve().parents[1]
SCOUT_CONFIG_PATH = ROOT / "config" / "scout_alpha_v1.json"
EXECUTION_POLICY_PATH = ROOT / "config" / "execution_policy.json"
PAGE_WIDTH, _ = landscape(letter)
MARGIN = 24
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
NAVY = colors.HexColor("#111827")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
GRID = colors.HexColor("#9CA3AF")
MUTED = colors.HexColor("#4B5563")
GROSS_FOOTER = "GROSS — no execution costs modeled"
SECTIONS = (
    "1. Day summary",
    "2. Scout trades",
    "3. Execution policy comparison",
    "4. Top-10 movers of the day",
    "5. Scorability and universe",
    "6. Scored candidates",
    "7. Postmortem",
)
METRIC_IDS = (
    "relative_volume",
    "price_slope_15m",
    "price_slope_30m",
    "price_slope_60m",
    "volume_slope_15m",
    "volume_slope_30m",
    "volume_slope_60m",
    "premarket_gap_strength",
    "price_vs_premarket_vwap",
    "premarket_trend_consistency",
    "price_volume_confirmation",
    "premarket_range_expansion",
)


class ReplayReportError(Exception):
    """Raised when persisted replay artifacts cannot produce a report."""


def _read_optional(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.is_file() else None


def _required(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReplayReportError(f"Required replay artifact is missing: {path}")
    return load_json(path)


def load_day_artifacts(day_dir: Path) -> dict[str, Any]:
    """Load the immutable JSON inputs used by one daily report."""
    return {
        "universe": _required(day_dir / "daily_universe_manifest.json"),
        "snapshot": _required(day_dir / "historical_snapshot.json"),
        "scout": _required(day_dir / "research_alpha_output.json"),
        "outcome": _required(day_dir / "end_of_day_outcome.json"),
        "benchmark": _required(day_dir / "benchmark_result.json"),
        "postmortem": _read_optional(day_dir / "postmortem.json"),
    }


def _selected(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("research_selected", candidate.get("selected", False)))


def _qualifying(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get(
            "qualification_selected", candidate.get("selected", False)
        )
    )


def _fmt_number(value: Any, places: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.{places}f}"


def _fmt_pct(value: Any) -> str:
    return "—" if value is None else f"{float(value):,.2f}%"


def _fmt_money(value: Any) -> str:
    if value is None:
        return "—"
    amount = float(value)
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def _fmt_time(value: Any) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime(
            "%H:%M:%S"
        )
    except ValueError:
        return str(value)


def _fmt_raw(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return _fmt_number(value)
    if isinstance(value, dict):
        return ";".join(
            f"{key}={_fmt_raw(item)}" for key, item in list(value.items())[:2]
        )
    if isinstance(value, list):
        return f"[{len(value)}]"
    return str(value)


def _verdict(benchmark: dict[str, Any]) -> str:
    return {
        "SCOUT_OUTPERFORMED": "WIN",
        "SCOUT_TIED": "TIE",
        "SCOUT_UNDERPERFORMED": "MISS",
    }[benchmark["comparison"]["result_code"]]


def _premarket_count(security: dict[str, Any]) -> int:
    observation = security.get("market_data", {}).get(
        "real_bar_count_60m", {}
    )
    value = observation.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return 0


def _histogram(snapshot: dict[str, Any]) -> list[tuple[str, int]]:
    counts = Counter({label: 0 for label in ("0", "1-4", "5-9", "10-19", "20-29", "30+")})
    for security in snapshot.get("securities", []):
        count = _premarket_count(security)
        label = (
            "0" if count == 0 else
            "1-4" if count < 5 else
            "5-9" if count < 10 else
            "10-19" if count < 20 else
            "20-29" if count < 30 else
            "30+"
        )
        counts[label] += 1
    return [(label, counts[label]) for label in ("0", "1-4", "5-9", "10-19", "20-29", "30+")]


def _exclusion_counts(
    universe: dict[str, Any],
) -> tuple[Counter[str], Counter[str]]:
    definitive: Counter[str] = Counter()
    gaps: Counter[str] = Counter()
    for security in universe.get("securities", []):
        if security.get("inclusion"):
            continue
        deciding = security.get("deciding_definitive_reason")
        if deciding:
            definitive[deciding] += 1
            continue
        for reason in security.get("reason_codes", []):
            classification = EXCLUSION_REASON_CLASSIFICATION.get(reason)
            if classification == DEFINITIVE:
                definitive[reason] += 1
            elif classification == GAP:
                gaps[reason] += 1
    return definitive, gaps


def _zero_selection_reason(scout: dict[str, Any]) -> str:
    candidates = scout.get("candidates", [])
    if not any(candidate.get("status") == "SCORED" for candidate in candidates):
        return "No Scout trades were selected because there were no qualifying candidates."
    guardrail_rejections = sum(
        any(
            result.get("action") == "REJECT"
            for result in candidate.get("guardrails", {}).values()
        )
        for candidate in candidates
    )
    threshold_rejections = sum(
        "BELOW_RESEARCH_THRESHOLD" in candidate.get("rejection_reasons", [])
        for candidate in candidates
    )
    if guardrail_rejections and not threshold_rejections:
        return "No Scout trades were selected because every qualifying candidate failed a guardrail."
    if threshold_rejections and not guardrail_rejections:
        return "No Scout trades were selected because no candidate reached the configured threshold."
    return "No Scout trades were selected because candidates failed the threshold and/or guardrails."


def _policy_summary(
    executions: list[dict[str, Any]],
    realized_return_pct: float,
) -> dict[str, Any]:
    completed = [item for item in executions if item.get("trade_executed")]
    returns = [float(item["realized_return_pct"]) for item in completed]
    winners = [value for value in returns if value > 0]
    losers = [value for value in returns if value < 0]
    captures = [
        float(item["capture_ratio"])
        for item in completed
        if item.get("capture_ratio") is not None
    ]
    drawdowns = [
        float(item["maximum_position_drawdown_pct"])
        for item in completed
        if item.get("maximum_position_drawdown_pct") is not None
    ]
    normal = Counter(
        item.get("exit_reason")
        for item in executions
        if item.get("exit_reason") in {
            "TRAILING_STOP", "PROFIT_TARGET", "SESSION_END"
        }
    )
    rejected = Counter(
        item.get("entry_rejection_reason") or "UNSPECIFIED"
        for item in executions
        if item.get("exit_reason") == "ENTRY_REJECTED"
    )
    return {
        "net_realized_pnl_usd": sum(
            float(item.get("realized_pnl_usd") or 0) for item in completed
        ),
        "realized_return_pct": realized_return_pct,
        "win_rate_pct": (
            sum(value > 0 for value in returns) / len(returns) * 100
            if returns else None
        ),
        "average_winner_pct": mean(winners) if winners else None,
        "average_loser_pct": mean(losers) if losers else None,
        "average_capture_ratio": mean(captures) if captures else None,
        "max_drawdown_pct": min(drawdowns) if drawdowns else None,
        "exit_reason_counts": {
            "TRAILING_STOP": normal["TRAILING_STOP"],
            "PROFIT_TARGET": normal["PROFIT_TARGET"],
            "SESSION_END": normal["SESSION_END"],
            "ENTRY_REJECTED": dict(sorted(rejected.items())),
        },
    }


def _execution_matrix(
    outcome: dict[str, Any],
    benchmark: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_executions = [
        item["execution_result"]
        for item in outcome.get("outcomes", [])
        if item.get(
            "selected", item.get("execution_result", {}).get("trade_executed", False)
        )
    ]
    sample = next(iter(primary_executions), {})
    primary_config = load_json(EXECUTION_POLICY_PATH)
    primary_management = primary_config["position_management"]
    combined = benchmark.get("combined_summary", benchmark["scout_summary"])
    policies = [{
        "policy_id": sample.get(
            "policy_id", primary_config["policy_id"]
        ),
        "exit_mode": sample.get(
            "exit_mode", primary_management["exit_mode"]
        ),
        "sizing_mode": sample.get(
            "sizing_mode", primary_management["sizing_mode"]
        ),
        "summary": _policy_summary(
            primary_executions,
            float(combined["realized_return_pct"]),
        ),
    }]
    policies.extend(outcome.get("policy_comparisons", []))

    rows: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    primary_id = policies[0]["policy_id"]
    for item in outcome.get("outcomes", []):
        if item.get(
            "selected", item.get("execution_result", {}).get("trade_executed", False)
        ):
            key = ("SCOUT_SELECTION", item["ticker"], None)
            rows.setdefault(key, {"cohort": key[0], "ticker": key[1], "rank": None, "executions": {}})
            rows[key]["executions"][primary_id] = item["execution_result"]
    for item in benchmark.get("benchmark_candidates", []):
        key = ("TOP_10_MOVER", item["ticker"], item["benchmark_rank"])
        rows.setdefault(key, {"cohort": key[0], "ticker": key[1], "rank": key[2], "executions": {}})
        rows[key]["executions"][primary_id] = {
            "entry_timestamp": item.get("entry_timestamp"),
            "entry_price": item.get("entry_price"),
            "exit_timestamp": item.get("exit_timestamp"),
            "exit_price": item.get("exit_price"),
            "exit_reason": item.get("exit_reason") or item.get("simulation_exclusion_reason"),
            "realized_return_pct": item.get("realized_return_pct"),
            "capture_ratio": item.get("capture_ratio"),
        }
    for policy in outcome.get("policy_comparisons", []):
        for item in policy.get("executions", []):
            key = (item["cohort"], item["ticker"], item.get("benchmark_rank"))
            rows.setdefault(key, {"cohort": key[0], "ticker": key[1], "rank": key[2], "executions": {}})
            rows[key]["executions"][policy["policy_id"]] = item["execution_result"]
    ordered = sorted(
        rows.values(),
        key=lambda item: (
            0 if item["cohort"] == "SCOUT_SELECTION" else 1,
            item["rank"] is None,
            item["rank"] or 999,
            item["ticker"],
        ),
    )
    return policies, ordered


def build_daily_report_model(
    artifacts: dict[str, Any],
    *,
    day_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a renderer-neutral model, also used for deterministic tests."""
    universe = artifacts["universe"]
    snapshot = artifacts["snapshot"]
    scout = artifacts["scout"]
    outcome = artifacts["outcome"]
    benchmark = artifacts["benchmark"]
    postmortem = artifacts.get("postmortem")
    day_record = day_record or {}
    candidates = {item["ticker"]: item for item in scout.get("candidates", [])}
    outcomes = {item["ticker"]: item for item in outcome.get("outcomes", [])}
    selected = sorted(
        (item for item in scout.get("candidates", []) if _selected(item)),
        key=lambda item: (item.get("rank") is None, item.get("rank") or 999999, item["ticker"]),
    )
    trade_rows = []
    for candidate in selected:
        observed = outcomes.get(candidate["ticker"], {})
        execution = observed.get("execution_result", {})
        trade_rows.append({
            "ticker": candidate["ticker"],
            "rank": candidate.get("rank"),
            "score_pct": candidate.get("score_pct"),
            "entry_timestamp": execution.get("entry_timestamp"),
            "entry_price": execution.get("entry_price"),
            "shares": execution.get("position_size_shares"),
            "exit_timestamp": execution.get("exit_timestamp"),
            "exit_price": execution.get("exit_price"),
            "exit_reason": execution.get("exit_reason"),
            "pnl_usd": execution.get("realized_pnl_usd"),
            "return_pct": execution.get("realized_return_pct"),
            "mfe_pct": observed.get("mfe_pct"),
            "mae_pct": observed.get("mae_pct"),
            "capture_ratio": execution.get("capture_ratio"),
        })
    misses = {
        item["ticker"]: item
        for item in (postmortem or {}).get("missed_opportunities", [])
    }
    mover_rows = []
    for item in benchmark.get("benchmark_candidates", []):
        candidate = candidates.get(item["ticker"])
        mover_rows.append({
            **item,
            "scorable": bool(candidate and candidate.get("status") == "SCORED"),
            "miss_classification": misses.get(item["ticker"], {}).get(
                "miss_classification"
            ),
        })
    definitive, gaps = _exclusion_counts(universe)
    scorable = sorted(
        (
            item for item in scout.get("candidates", [])
            if item.get("status") == "SCORED"
        ),
        key=lambda item: (-float(item.get("score_pct", 0)), item["ticker"]),
    )
    miss_counts = Counter(
        item["miss_classification"]
        for item in (postmortem or {}).get("missed_opportunities", [])
    )
    threshold_reviews = sorted(
        (
            item for item in (postmortem or {}).get("missed_opportunities", [])
            if item["miss_classification"] == "VISIBLE_SCORED_LOW"
        ),
        key=lambda item: (-float(item["maximum_capturable_move_pct"]), item["ticker"]),
    )[:3]
    scorable_count = int(scout.get("scorable_candidate_count", len(scorable)))
    universe_count = int(universe.get("eligible_symbol_count", len(candidates)))
    minimum_real_bars_60m = int(
        load_json(SCOUT_CONFIG_PATH)["minimum_real_bars_60m"]
    )
    execution_policies, execution_comparison_rows = _execution_matrix(
        outcome, benchmark
    )
    return {
        "replay_id": snapshot["replay_id"],
        "trading_date": snapshot["trading_date"],
        "freeze_timestamp": snapshot["freeze_timestamp"],
        "plan": benchmark.get("massive_plan", {}).get("plan", "UNKNOWN"),
        "mode": "SMOKE" if day_record.get("smoke_mode") else "RESEARCH",
        "universe_manifest_hash": universe["manifest_hash"],
        "universe_count": universe_count,
        "scorable_count": scorable_count,
        "scorable_share": scorable_count / universe_count if universe_count else 0.0,
        "qualifying_count": sum(_qualifying(item) for item in candidates.values()),
        "selected_count": len(selected),
        "verdict": _verdict(benchmark),
        "scout_return_pct": benchmark["scout_summary"]["realized_return_pct"],
        "baseline_return_pct": benchmark["return_baselines"][
            "random_draw_mean_realized_return_pct"
        ],
        "phase_wall_time_seconds": day_record.get("phase_wall_time_seconds", {}),
        "trade_rows": trade_rows,
        "zero_selection_reason": _zero_selection_reason(scout),
        "mover_rows": mover_rows,
        "benchmark_summary": benchmark["benchmark_summary"],
        "histogram": _histogram(snapshot),
        "minimum_real_bars_60m": minimum_real_bars_60m,
        "definitive_exclusions": definitive,
        "coverage_gaps": gaps,
        "scored_candidates": scorable,
        "postmortem": postmortem,
        "miss_category_counts": miss_counts,
        "threshold_reviews": threshold_reviews,
        "execution_policies": execution_policies,
        "execution_comparison_rows": execution_comparison_rows,
    }


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReplayTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=20, leading=23, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "section": ParagraphStyle(
            "ReplaySection", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=16, leading=19, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subsection": ParagraphStyle(
            "ReplaySubsection", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=10, leading=12, textColor=NAVY, spaceBefore=7, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "ReplayBody", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8, leading=11, textColor=NAVY,
        ),
        "small": ParagraphStyle(
            "ReplaySmall", parent=base["BodyText"], fontName="Helvetica",
            fontSize=6.5, leading=8, textColor=MUTED,
        ),
    }


def _table(
    rows: list[list[Any]],
    widths: list[float],
    *,
    right_columns: Iterable[int] = (),
    font_size: float = 6.2,
) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Courier"),
        ("FONTNAME", (0, 0), (-1, 0), "Courier-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 1.5),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]
    for column in right_columns:
        commands.append(("ALIGN", (column, 1), (column, -1), "RIGHT"))
    table.setStyle(TableStyle(commands))
    return table


def _footer(replay_id: str, manifest_hash: str):
    def draw(canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(GRID)
        canvas.line(MARGIN, 25, PAGE_WIDTH - MARGIN, 25)
        canvas.setFont("Courier", 5.7)
        canvas.setFillColor(MUTED)
        left = f"{replay_id} | {manifest_hash}"
        canvas.drawString(MARGIN, 8, left)
        canvas.drawCentredString(PAGE_WIDTH / 2, 17, GROSS_FOOTER)
        canvas.drawRightString(PAGE_WIDTH - MARGIN, 8, f"Page {document.page}")
        canvas.restoreState()
    return draw


def _page(story: list[Any], heading: str, styles: dict[str, ParagraphStyle]) -> None:
    if story:
        story.append(PageBreak())
    story.append(Paragraph(heading, styles["section"]))


def _day_summary(story: list[Any], model: dict[str, Any], styles: dict[str, ParagraphStyle]) -> None:
    _page(story, SECTIONS[0], styles)
    rows = [
        ["Trading date", model["trading_date"], "Freeze time", _fmt_time(model["freeze_timestamp"])],
        ["Massive plan", model["plan"], "Mode", model["mode"]],
        ["Eligible universe", str(model["universe_count"]), "Scorable", f"{model['scorable_count']} ({model['scorable_share'] * 100:.2f}%)"],
        ["Qualifying", str(model["qualifying_count"]), "Selected", str(model["selected_count"])],
        ["Verdict", model["verdict"], "Gross return basis", "Strategy capital; no costs"],
        ["Scout gross return", _fmt_pct(model["scout_return_pct"]), "Random-draw baseline", _fmt_pct(model["baseline_return_pct"])],
    ]
    story.append(_table([["Measure", "Value", "Measure", "Value"], *rows], [1.45*inch, 2.2*inch, 1.45*inch, 2.2*inch], right_columns=(1, 3), font_size=7.2))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Wall time by phase", styles["subsection"]))
    phases = model["phase_wall_time_seconds"]
    phase_rows = [["Phase", "Seconds"]] + [
        [phase, _fmt_number(phases.get(phase))]
        for phase in ("download", "universe", "snapshot", "scoring", "grading", "benchmark", "postmortem", "report")
    ]
    story.append(_table(phase_rows, [2.4*inch, 1.1*inch], right_columns=(1,), font_size=7))


def _scout_trades(story: list[Any], model: dict[str, Any], styles: dict[str, ParagraphStyle]) -> None:
    _page(story, SECTIONS[1], styles)
    trades = model["trade_rows"]
    if not trades:
        story.append(Paragraph(model["zero_selection_reason"], styles["body"]))
        return
    rows = [["Rk", "Ticker", "Score", "Entry", "Entry $", "Shares", "Exit", "Exit $", "Reason", "P&L", "Return", "MFE", "MAE", "Capture"]]
    total_pnl = 0.0
    for item in trades:
        total_pnl += float(item["pnl_usd"] or 0)
        rows.append([
            str(item["rank"] or "—"), item["ticker"], _fmt_pct(item["score_pct"]),
            _fmt_time(item["entry_timestamp"]), _fmt_money(item["entry_price"]), _fmt_number(item["shares"], 0),
            _fmt_time(item["exit_timestamp"]), _fmt_money(item["exit_price"]), item["exit_reason"] or "—",
            _fmt_money(item["pnl_usd"]), _fmt_pct(item["return_pct"]), _fmt_pct(item["mfe_pct"]),
            _fmt_pct(item["mae_pct"]), _fmt_pct(None if item["capture_ratio"] is None else float(item["capture_ratio"]) * 100),
        ])
    rows.append(["", "TOTAL", "", "", "", "", "", "", "", _fmt_money(total_pnl), _fmt_pct(model["scout_return_pct"]), "", "", ""])
    widths = [0.3,0.55,0.62,0.68,0.65,0.55,0.68,0.65,1.0,0.72,0.67,0.58,0.58,0.66]
    story.append(_table(rows, [value*inch for value in widths], right_columns=(0,2,4,5,7,9,10,11,12,13), font_size=6.0))


def _exit_counts(value: dict[str, Any]) -> str:
    rejected = value.get("ENTRY_REJECTED", {})
    rejected_text = ",".join(
        f"{reason}:{count}" for reason, count in sorted(rejected.items())
    ) or "0"
    return (
        f"STOP:{value.get('TRAILING_STOP', 0)} "
        f"TARGET:{value.get('PROFIT_TARGET', 0)} "
        f"EOD:{value.get('SESSION_END', 0)} "
        f"REJECT:{rejected_text}"
    )


def _execution_policy_comparison(
    story: list[Any],
    model: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> None:
    _page(story, SECTIONS[2], styles)
    summary_rows = [[
        "Policy", "Exit", "Sizing", "Net P&L", "Return", "Win rate",
        "Avg winner", "Avg loser", "Avg capture", "Max DD", "Exit counts",
    ]]
    for policy in model["execution_policies"]:
        summary = policy["summary"]
        summary_rows.append([
            policy["policy_id"], policy["exit_mode"], policy["sizing_mode"],
            _fmt_money(summary["net_realized_pnl_usd"]),
            _fmt_pct(summary["realized_return_pct"]),
            _fmt_pct(summary["win_rate_pct"]),
            _fmt_pct(summary["average_winner_pct"]),
            _fmt_pct(summary["average_loser_pct"]),
            _fmt_pct(
                None if summary["average_capture_ratio"] is None
                else float(summary["average_capture_ratio"]) * 100
            ),
            _fmt_pct(summary["max_drawdown_pct"]),
            _exit_counts(summary["exit_reason_counts"]),
        ])
    story.append(_table(
        summary_rows,
        [1.35*inch,0.55*inch,0.9*inch,0.72*inch,0.62*inch,0.62*inch,
         0.68*inch,0.68*inch,0.75*inch,0.62*inch,2.25*inch],
        right_columns=(3,4,5,6,7,8,9),
        font_size=5.8,
    ))
    story.append(Paragraph(
        "Per-ticker Scout selections and top movers (policies shown side by side)",
        styles["subsection"],
    ))
    policies = model["execution_policies"]
    header = ["Cohort", "Rk", "Ticker"]
    widths = [0.82*inch, 0.3*inch, 0.55*inch]
    for policy in policies:
        short = policy["policy_id"].replace("execution_policy_", "")
        header.extend([
            f"{short} entry", f"{short} exit", f"{short} reason",
            f"{short} return", f"{short} capture",
        ])
        widths.extend([0.85*inch,0.85*inch,0.95*inch,0.58*inch,0.58*inch])
    rows = [header]
    for item in model["execution_comparison_rows"]:
        row = [
            "SCOUT" if item["cohort"] == "SCOUT_SELECTION" else "TOP-10",
            str(item["rank"] or "—"),
            item["ticker"],
        ]
        for policy in policies:
            execution = item["executions"].get(policy["policy_id"], {})
            row.extend([
                f"{_fmt_time(execution.get('entry_timestamp'))} @ {_fmt_money(execution.get('entry_price'))}",
                f"{_fmt_time(execution.get('exit_timestamp'))} @ {_fmt_money(execution.get('exit_price'))}",
                execution.get("entry_rejection_reason") or execution.get("exit_reason") or "—",
                _fmt_pct(execution.get("realized_return_pct")),
                _fmt_pct(
                    None if execution.get("capture_ratio") is None
                    else float(execution["capture_ratio"]) * 100
                ),
            ])
        rows.append(row)
    if len(rows) == 1:
        rows.append(["None", "—", "—"] + ["—"] * (5 * len(policies)))
    story.append(_table(
        rows, widths,
        right_columns=tuple(
            index for index, name in enumerate(header)
            if name.endswith("return") or name.endswith("capture") or name == "Rk"
        ),
        font_size=4.8 if len(policies) > 1 else 5.8,
    ))


def _movers(story: list[Any], model: dict[str, Any], styles: dict[str, ParagraphStyle]) -> None:
    _page(story, SECTIONS[3], styles)
    rows = [["Rk", "Ticker", "Max move", "Selected", "Scorable", "Sim return", "Exit reason", "Miss classification"]]
    for item in model["mover_rows"]:
        rows.append([
            str(item["benchmark_rank"]), item["ticker"], _fmt_pct(item["maximum_capturable_move_pct"]),
            "YES" if item["scout_selected"] else "NO", "YES" if item["scorable"] else "NO",
            _fmt_pct(item["realized_return_pct"]), item["exit_reason"] or item["simulation_exclusion_reason"] or "—",
            item["miss_classification"] or "NOT RUN",
        ])
    story.append(_table(rows, [0.45*inch,0.8*inch,1.0*inch,0.85*inch,0.85*inch,1.0*inch,1.55*inch,2.45*inch], right_columns=(0,2,5), font_size=6.5))
    summary = model["benchmark_summary"]
    story.append(Paragraph("Benchmark summary", styles["subsection"]))
    summary_rows = [["Trades", "Win rate", "Profit factor", "Net P&L", "Average capture"] , [
        str(summary["trades_executed"]), _fmt_pct(summary["win_rate_pct"]), _fmt_number(summary["profit_factor"]),
        _fmt_money(summary["realized_pnl_usd"]), _fmt_pct(None if summary["average_capture_ratio"] is None else float(summary["average_capture_ratio"]) * 100),
    ]]
    story.append(_table(summary_rows, [1.1*inch]*5, right_columns=(0,1,2,3,4), font_size=7))
    stopped = [item for item in model["mover_rows"] if item.get("exit_reason") == "TRAILING_STOP"]
    if stopped:
        moves = [float(item["maximum_capturable_move_pct"]) for item in stopped]
        returns = [float(item["realized_return_pct"]) for item in stopped]
        reading = (
            f"{len(stopped)} of {summary['trades_executed']} simulated top movers "
            f"were stopped out at {min(returns):.2f}% to {max(returns):.2f}% despite "
            f"{min(moves):.2f}-{max(moves):.2f}% available moves."
        )
    else:
        reading = f"{summary['trades_executed']} top movers were simulated; none exited by trailing stop."
    story.append(Spacer(1, 6))
    story.append(Paragraph(reading, styles["body"]))


def _scorability(story: list[Any], model: dict[str, Any], styles: dict[str, ParagraphStyle]) -> None:
    _page(story, SECTIONS[4], styles)
    story.append(Paragraph(f"Minimum real bars in the final 60 minutes: {model['minimum_real_bars_60m']}", styles["body"]))
    story.append(Spacer(1, 5))
    histogram_rows = [["Real bars (60m)", "Ticker count"]] + [[label, str(count)] for label, count in model["histogram"]]
    story.append(_table(histogram_rows, [1.8*inch,1.2*inch], right_columns=(1,), font_size=7))
    story.append(Paragraph("Definitive exclusions", styles["subsection"]))
    definitive = [["Reason", "Count"]] + [[reason, str(count)] for reason, count in sorted(model["definitive_exclusions"].items())]
    if len(definitive) == 1:
        definitive.append(["None", "0"])
    story.append(_table(definitive, [4.6*inch,0.8*inch], right_columns=(1,), font_size=6.8))
    story.append(Paragraph("Coverage gaps", styles["subsection"]))
    gaps = [["Reason", "Count"]] + [[reason, str(count)] for reason, count in sorted(model["coverage_gaps"].items())]
    if len(gaps) == 1:
        gaps.append(["None", "0"])
    story.append(_table(gaps, [4.6*inch,0.8*inch], right_columns=(1,), font_size=6.8))


def _candidate_block(candidate: dict[str, Any], styles: dict[str, ParagraphStyle]) -> KeepTogether:
    guards = ", ".join(
        f"{name}={result.get('action', '—')}"
        for name, result in sorted(candidate.get("guardrails", {}).items())
    ) or "None"
    reasons = ", ".join(candidate.get("rejection_reasons", [])) or "None"
    header = Paragraph(
        f"<b>{candidate['ticker']}</b> — {_fmt_pct(candidate.get('score_pct'))} "
        f"({candidate.get('total_score', 0)}/{candidate.get('maximum_possible_score', 48)}) "
        f"| Guardrails: {guards} | Rejections: {reasons}",
        styles["small"],
    )
    components = candidate.get("component_scores", {})
    metric_rows = [["Metric", "Pts", "Raw", "Metric", "Pts", "Raw", "Metric", "Pts", "Raw"]]
    for offset in range(0, len(METRIC_IDS), 3):
        row: list[str] = []
        for metric_id in METRIC_IDS[offset:offset+3]:
            component = components.get(metric_id, {})
            score = component.get("score")
            row.extend([metric_id, "—" if score is None else str(score), _fmt_raw(component.get("raw_value"))])
        metric_rows.append(row)
    widths = []
    for _ in range(3):
        widths.extend([2.15*inch,0.32*inch,0.72*inch])
    return KeepTogether([header, Spacer(1, 2), _table(metric_rows, widths, right_columns=(1,2,4,5,7,8), font_size=6.0), Spacer(1, 6)])


def _candidates(story: list[Any], model: dict[str, Any], styles: dict[str, ParagraphStyle]) -> None:
    _page(story, SECTIONS[5], styles)
    if not model["scored_candidates"]:
        story.append(Paragraph("No candidates were scorable.", styles["body"]))
        return
    for candidate in model["scored_candidates"]:
        story.append(_candidate_block(candidate, styles))


def _postmortem(story: list[Any], model: dict[str, Any], styles: dict[str, ParagraphStyle]) -> None:
    _page(story, SECTIONS[6], styles)
    postmortem = model["postmortem"]
    if postmortem is None:
        story.append(Paragraph("Postmortem was not produced for this day (smoke or non-research evidence mode).", styles["body"]))
        return
    counts = [["Miss category", "Count"]] + [[name, str(count)] for name, count in sorted(model["miss_category_counts"].items())]
    if len(counts) == 1:
        counts.append(["None", "0"])
    story.append(_table(counts, [3.2*inch,0.8*inch], right_columns=(1,), font_size=7))
    story.append(Paragraph("Missed opportunities and low metrics", styles["subsection"]))
    missed_rows = [["Rk", "Ticker", "Category", "Max move", "Low-scoring metrics"]]
    for item in postmortem.get("missed_opportunities", []):
        lows = ", ".join(
            f"{metric['metric_id']}={metric['score']}"
            for metric in item.get("component_scores", [])
            if metric.get("score") is not None and metric["score"] <= 1
        ) or "None"
        missed_rows.append([str(item["benchmark_rank"]), item["ticker"], item["miss_classification"], _fmt_pct(item["maximum_capturable_move_pct"]), lows])
    if len(missed_rows) == 1:
        missed_rows.append(["—", "—", "None", "—", "—"])
    story.append(_table(missed_rows, [0.35*inch,0.65*inch,1.7*inch,0.8*inch,4.0*inch], right_columns=(0,3), font_size=5.6))
    story.append(Paragraph("Execution policy review", styles["subsection"]))
    execution_rows = [["Policy", "Cohort", "Trades", "Return", "Avg capture", "Max DD"]]
    for item in postmortem.get("execution_policy_review", []):
        if "cohort_summaries" not in item:
            execution_rows.append([
                item.get("ticker", "—"), "legacy", "1",
                _fmt_pct(item.get("realized_return_pct")), "—", "—",
            ])
            continue
        for cohort, summary in item["cohort_summaries"].items():
            execution_rows.append([
                item["policy_id"], cohort, str(summary["trades_executed"]),
                _fmt_pct(summary["realized_return_pct"]),
                (
                    "—" if summary["average_capture_ratio"] is None
                    else f"{float(summary['average_capture_ratio']):.3f}x"
                ),
                _fmt_pct(summary["max_drawdown_pct"]),
            ])
    if len(execution_rows) == 1:
        execution_rows.append(["—", "—", "0", "—", "—", "—"])
    story.append(_table(execution_rows, [1.8*inch,1.25*inch,0.6*inch,0.8*inch,0.9*inch,0.8*inch], right_columns=(2,3,4,5), font_size=5.8))
    scorability = postmortem.get("universe_scorability")
    if scorability:
        story.append(Paragraph("Universe scorability diagnostic", styles["subsection"]))
        story.append(_table([
            ["Eligible", "Scorable", "Not scorable", "Zero premarket", "Zero final 60m", "Top-10 invisible"],
            [
                str(scorability["eligible_count"]),
                str(scorability["scorable_count"]),
                _fmt_pct(float(scorability["not_scorable_share"]) * 100),
                str(scorability["zero_premarket_bar_count"]),
                str(scorability["zero_premarket_60m_bar_count"]),
                str(scorability["top_10_invisible_count"]),
            ],
        ], [0.8*inch,0.8*inch,1.0*inch,1.1*inch,1.1*inch,1.1*inch], right_columns=(0,1,2,3,4,5), font_size=6))
    story.append(Paragraph("Top three candidate threshold reviews", styles["subsection"]))
    threshold_rows = [["Rk", "Ticker", "Score", "Max move", "Failure reasons"]]
    for item in model["threshold_reviews"]:
        threshold_rows.append([str(item["benchmark_rank"]), item["ticker"], _fmt_pct(item["scout_score_pct"]), _fmt_pct(item["maximum_capturable_move_pct"]), ", ".join(item["failure_reason_codes"])])
    if len(threshold_rows) == 1:
        threshold_rows.append(["—", "—", "—", "—", "None"])
    story.append(_table(threshold_rows, [0.35*inch,0.65*inch,0.75*inch,0.8*inch,4.8*inch], right_columns=(0,2,3), font_size=6))


def generate_daily_replay_report(
    day_dir: Path,
    output_path: Path | None = None,
    *,
    day_record: dict[str, Any] | None = None,
) -> Path:
    artifacts = load_day_artifacts(day_dir)
    model = build_daily_report_model(artifacts, day_record=day_record)
    output_path = output_path or day_dir / "replay_report.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story: list[Any] = []
    _day_summary(story, model, styles)
    _scout_trades(story, model, styles)
    _execution_policy_comparison(story, model, styles)
    _movers(story, model, styles)
    _scorability(story, model, styles)
    _candidates(story, model, styles)
    _postmortem(story, model, styles)
    document = SimpleDocTemplate(
        str(output_path), pagesize=landscape(letter), rightMargin=MARGIN,
        leftMargin=MARGIN, topMargin=25, bottomMargin=34,
        title=f"Replay Report {model['trading_date']}", author="AI Trading Lab",
    )
    draw = _footer(model["replay_id"], model["universe_manifest_hash"])
    document.build(story, onFirstPage=draw, onLaterPages=draw)
    return output_path


def _completed_day_models(output_root: Path) -> list[dict[str, Any]]:
    manifest = _required(output_root / "flatfile_replay_manifest.json")
    records = {item["trading_date"]: item for item in manifest.get("days", [])}
    models = []
    for trading_date in manifest.get("completed_dates", []):
        day_dir = output_root / "days" / trading_date
        models.append(
            build_daily_report_model(
                load_day_artifacts(day_dir), day_record=records.get(trading_date)
            )
        )
    return models


def generate_cumulative_replay_report(
    output_root: Path,
    output_path: Path | None = None,
) -> Path:
    manifest = _required(output_root / "flatfile_replay_manifest.json")
    concatenate_completed_outcomes(
        output_root, list(manifest.get("completed_dates", []))
    )
    models = _completed_day_models(output_root)
    output_path = output_path or output_root / "replay_summary.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story: list[Any] = [Paragraph("Cumulative Replay Summary", styles["title"])]
    rows = [["Date", "Universe", "Scorable", "Selected", "Scout return", "Baseline return", "Verdict"]]
    for model in models:
        rows.append([
            model["trading_date"], str(model["universe_count"]), str(model["scorable_count"]),
            str(model["selected_count"]), _fmt_pct(model["scout_return_pct"]),
            _fmt_pct(model["baseline_return_pct"]), model["verdict"],
        ])
    if len(rows) == 1:
        rows.append(["—", "0", "0", "0", "0.00%", "0.00%", "—"])
    story.append(_table(rows, [1.05*inch,0.85*inch,0.85*inch,0.8*inch,1.1*inch,1.2*inch,0.75*inch], right_columns=(1,2,3,4,5), font_size=7))
    totals = {
        "universe": sum(item["universe_count"] for item in models),
        "scorable": sum(item["scorable_count"] for item in models),
        "selected": sum(item["selected_count"] for item in models),
        "scout_return": sum(float(item["scout_return_pct"]) for item in models),
        "baseline_return": sum(float(item["baseline_return_pct"]) for item in models),
    }
    story.append(Paragraph("Cumulative totals", styles["subsection"]))
    story.append(_table([
        ["Completed days", "Universe observations", "Scorable", "Selected", "Scout return sum", "Baseline return sum"],
        [str(len(models)), str(totals["universe"]), str(totals["scorable"]), str(totals["selected"]), _fmt_pct(totals["scout_return"]), _fmt_pct(totals["baseline_return"])],
    ], [1.1*inch,1.35*inch,0.9*inch,0.9*inch,1.25*inch,1.35*inch], right_columns=(0,1,2,3,4,5), font_size=6.5))
    reachability: Counter[str] = Counter()
    for model in models:
        reachability.update(
            (model.get("postmortem") or {}).get("reachability", {})
        )
    story.append(Paragraph("Top-10 mover reachability", styles["subsection"]))
    story.append(_table([
        ["0 bars", "1-9 bars", "10-29 bars", "Visible low", "Reversal", "Guardrail rejected", "Picked", "Opening-range reachable"],
        [
            str(reachability["bars_0"]),
            str(reachability["bars_1_9"]),
            str(reachability["bars_10_29"]),
            str(reachability["visible_scored_low"]),
            str(reachability["visible_reversal_candidate"]),
            str(reachability["visible_guardrail_rejected"]),
            str(reachability["picked"]),
            str(reachability["opening_range_reachable_count"]),
        ],
    ], [0.65*inch,0.65*inch,0.75*inch,0.75*inch,0.75*inch,1.0*inch,0.55*inch,1.15*inch], right_columns=(0,1,2,3,4,5,6,7), font_size=5.5))
    reversal_rows = [["Date", "Candidates", "Selected", "Close > open", "Mean day MFE", "Policy returns"]]
    all_closed: list[bool] = []
    all_mfe: list[float] = []
    policy_returns: Counter[str] = Counter()
    for model in models:
        cohort = (model.get("postmortem") or {}).get("reversal_cohort", {})
        for candidate in cohort.get("candidates", []):
            if candidate.get("closed_above_open") is not None:
                all_closed.append(candidate["closed_above_open"])
            if candidate.get("day_mfe_pct") is not None:
                all_mfe.append(float(candidate["day_mfe_pct"]))
        policy_returns.update(cohort.get("policy_returns", {}))
        reversal_rows.append([
            model["trading_date"],
            str(cohort.get("candidate_count", 0)),
            str(cohort.get("selected_count", 0)),
            _fmt_number(cohort.get("close_above_open_share")),
            _fmt_pct(cohort.get("mean_day_mfe_pct")),
            ", ".join(
                f"{policy_id}={_fmt_pct(value)}"
                for policy_id, value in sorted(
                    cohort.get("policy_returns", {}).items()
                )
            ) or "—",
        ])
    reversal_rows.append([
        "Cumulative",
        str(sum(
            int(((model.get("postmortem") or {}).get("reversal_cohort", {})).get("candidate_count", 0))
            for model in models
        )),
        str(sum(
            int(((model.get("postmortem") or {}).get("reversal_cohort", {})).get("selected_count", 0))
            for model in models
        )),
        _fmt_number(sum(all_closed) / len(all_closed) if all_closed else None),
        _fmt_pct(mean(all_mfe) if all_mfe else None),
        ", ".join(
            f"{policy_id}={_fmt_pct(value)}"
            for policy_id, value in sorted(policy_returns.items())
        ) or "—",
    ])
    story.append(Paragraph("Reversal exploration base rates", styles["subsection"]))
    story.append(_table(
        reversal_rows,
        [0.9*inch,0.7*inch,0.65*inch,0.85*inch,0.9*inch,3.1*inch],
        right_columns=(1,2,3,4),
        font_size=6,
    ))
    story.append(PageBreak())
    story.append(Paragraph("Miss-category totals across completed days", styles["section"]))
    miss_totals: Counter[str] = Counter()
    for model in models:
        miss_totals.update(model["miss_category_counts"])
    miss_rows = [["Miss category", "Count"]] + [[name, str(count)] for name, count in sorted(miss_totals.items())]
    if len(miss_rows) == 1:
        miss_rows.append(["No postmortem miss categories", "0"])
    story.append(_table(miss_rows, [4.0*inch,0.9*inch], right_columns=(1,), font_size=7))
    document = SimpleDocTemplate(
        str(output_path), pagesize=landscape(letter), rightMargin=MARGIN,
        leftMargin=MARGIN, topMargin=25, bottomMargin=34,
        title="Cumulative Replay Summary", author="AI Trading Lab",
    )
    draw = _footer(manifest["run_id"], "MULTI-DAY")
    document.build(story, onFirstPage=draw, onLaterPages=draw)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cumulative PDF reporting for completed replay days."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = generate_cumulative_replay_report(args.output_root)
    print(path)


if __name__ == "__main__":
    main()
