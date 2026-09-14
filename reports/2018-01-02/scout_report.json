from __future__ import annotations

from pathlib import Path
from typing import Any


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
        "| Rank | Ticker | Status | Score | Spread % | "
        "Liquidity | Spread Guardrail |"
    )
    lines.append(
        "|---:|---|---|---:|---:|---|---|"
    )

    ordered = sorted(
        scout_result["candidates"],
        key=lambda c: (
            c["rank"] is None,
            c["rank"] if c["rank"] is not None else 999999,
            -c["score_pct"],
            c["ticker"],
        ),
    )

    for candidate in ordered:
        liquidity = candidate["guardrails"]["liquidity"]
        spread = candidate["guardrails"]["spread"]

        lines.append(
            f"| {fmt(candidate['rank'])} "
            f"| {candidate['ticker']} "
            f"| {candidate_status(candidate)} "
            f"| {candidate['score_pct']:.2f}% "
            f"| {fmt(candidate['raw_values'].get('spread_pct'))} "
            f"| {liquidity['action']} "
            f"| {spread['action']} |"
        )

    lines.append("")
    lines.append("## Candidate Detail")
    lines.append("")

    for candidate in ordered:
        lines.append(
            f"### {candidate['ticker']} "
            f"— {candidate_status(candidate)}"
        )
        lines.append("")

        lines.append(f"- Rank: {fmt(candidate['rank'])}")
        lines.append(
            f"- Score: {candidate['score_pct']:.2f}% "
            f"({candidate['total_score']:.2f} / "
            f"{candidate['maximum_possible_score']:.2f})"
        )
        lines.append(
            f"- Eligible: {candidate['eligible']}"
        )
        lines.append(
            f"- Selected: {candidate['selected']}"
        )

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


def save_scout_report(
    scout_result: dict[str, Any],
    trading_date: str,
) -> Path:
    report_dir = ROOT / "reports" / trading_date
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "scout_report.md"

    report_text = generate_scout_markdown_report(
        scout_result
    )

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    return report_path
