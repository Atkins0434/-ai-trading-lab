from __future__ import annotations

from pathlib import Path
from typing import Any

from trainer.validate_contracts import (
    ContractError,
    load_json,
    validate_contract,
)


ROOT = Path(__file__).resolve().parent.parent
SCOUT_CONFIG_PATH = ROOT / "config" / "scout_v1.json"


class ScoutError(Exception):
    """Raised when Scout cannot deterministically score a snapshot."""


def load_scout_config() -> dict[str, Any]:
    """Load the frozen Scout V1 configuration."""
    try:
        return load_json(SCOUT_CONFIG_PATH)
    except ContractError as exc:
        raise ScoutError(f"Unable to load Scout configuration: {exc}") from exc


def calculate_spread_pct(
    bid: float | None,
    ask: float | None,
) -> float | None:
    """Calculate bid/ask spread as a percentage of midpoint."""
    if bid is None or ask is None:
        return None

    if bid <= 0 or ask <= 0 or ask < bid:
        return None

    midpoint = (bid + ask) / 2

    if midpoint == 0:
        return None

    return ((ask - bid) / midpoint) * 100


def score_relative_volume(relative_volume: float | None) -> float:
    """
    Score relative volume on the Scout 0-4 scale.

    This is intentionally conservative and deterministic.
    """
    if relative_volume is None:
        return 0.0

    if relative_volume < 1.0:
        return 0.0
    if relative_volume < 1.25:
        return 1.0
    if relative_volume < 1.5:
        return 2.0
    if relative_volume < 2.0:
        return 3.0

    return 4.0


def score_catalyst(
    security: dict[str, Any],
) -> float:
    """
    Score currently available verified information events.

    SEC filings receive double treatment in the baseline concept.
    News receives normal treatment.

    This is a temporary V1 implementation until catalyst
    classification becomes more granular.
    """
    news_count = len(security.get("news", []))
    filing_count = len(security.get("filings", []))

    weighted_events = news_count + (filing_count * 2)

    if weighted_events <= 0:
        return 0.0
    if weighted_events == 1:
        return 1.0
    if weighted_events == 2:
        return 2.0
    if weighted_events == 3:
        return 3.0

    return 4.0


def evaluate_liquidity_guardrail(
    market_data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate Scout's deterministic liquidity floor."""
    rules = config["liquidity"]

    avg_dollar_volume = market_data.get(
        "average_daily_dollar_volume"
    )
    premarket_dollar_volume = market_data.get(
        "premarket_dollar_volume"
    )

    minimum_avg = rules[
        "minimum_average_daily_dollar_volume_usd"
    ]
    minimum_premarket = rules[
        "minimum_premarket_dollar_volume_usd"
    ]

    if avg_dollar_volume is None or premarket_dollar_volume is None:
        return {
            "passed": False,
            "action": "REJECT",
            "observed_value": "MISSING_DATA",
            "threshold": (
                f"ADV>={minimum_avg}; "
                f"PREMARKET>={minimum_premarket}"
            ),
            "reason_code": "LIQUIDITY_DATA_MISSING",
        }

    passed = (
        avg_dollar_volume >= minimum_avg
        and premarket_dollar_volume >= minimum_premarket
    )

    return {
        "passed": passed,
        "action": "PASS" if passed else "REJECT",
        "observed_value": (
            f"ADV={avg_dollar_volume}; "
            f"PREMARKET={premarket_dollar_volume}"
        ),
        "threshold": (
            f"ADV>={minimum_avg}; "
            f"PREMARKET>={minimum_premarket}"
        ),
        "reason_code": (
            "LIQUIDITY_OK"
            if passed
            else "LIQUIDITY_BELOW_MINIMUM"
        ),
    }


def evaluate_spread_guardrail(
    market_data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply Scout's 0.20% penalty and 0.40% rejection rules."""
    spread_pct = calculate_spread_pct(
        market_data.get("bid"),
        market_data.get("ask"),
    )

    penalty_start = config["spread"]["penalty_starts_pct"]
    reject_at = config["spread"]["hard_reject_pct"]

    if spread_pct is None:
        return {
            "passed": False,
            "action": "REJECT",
            "observed_value": None,
            "threshold": reject_at,
            "reason_code": "SPREAD_DATA_INVALID_OR_MISSING",
        }

    if spread_pct >= reject_at:
        return {
            "passed": False,
            "action": "REJECT",
            "observed_value": spread_pct,
            "threshold": reject_at,
            "reason_code": "SPREAD_HARD_REJECT",
        }

    if spread_pct >= penalty_start:
        return {
            "passed": True,
            "action": "PENALIZE",
            "observed_value": spread_pct,
            "threshold": penalty_start,
            "reason_code": "SPREAD_PENALTY",
        }

    return {
        "passed": True,
        "action": "PASS",
        "observed_value": spread_pct,
        "threshold": penalty_start,
        "reason_code": "SPREAD_OK",
    }


def score_security(
    security: dict[str, Any],
    config: dict[str, Any],
    threshold_pct: float,
    timestamp: str,
) -> dict[str, Any]:
    """Score one security using currently implemented V1 features."""
    ticker = security["ticker"]
    market_data = security["market_data"]

    rejection_reasons: list[str] = []
    reason_codes: list[str] = []

    guardrails = {
        "liquidity": evaluate_liquidity_guardrail(
            market_data,
            config,
        ),
        "spread": evaluate_spread_guardrail(
            market_data,
            config,
        ),
    }

    for result in guardrails.values():
        reason_code = result.get("reason_code")

        if reason_code:
            reason_codes.append(reason_code)

        if result["action"] == "REJECT":
            rejection_reasons.append(reason_code)

    component_scores = {
        "relative_volume": {
            "score": score_relative_volume(
                market_data.get("relative_volume")
            ),
            "maximum_score": 4.0,
            "reason_code": "RELATIVE_VOLUME_SCORE",
        },
        "catalyst": {
            "score": score_catalyst(security),
            "maximum_score": 4.0,
            "reason_code": "CATALYST_SCORE",
        },
    }

    # Spread between 0.20% and 0.40% receives a small,
    # deterministic scoring penalty without becoming a third
    # independent score component.
    spread_result = guardrails["spread"]

    if spread_result["action"] == "PENALIZE":
        component_scores["relative_volume"]["score"] = max(
            0.0,
            component_scores["relative_volume"]["score"] - 1.0,
        )

    total_score = sum(
        item["score"]
        for item in component_scores.values()
    )

    maximum_possible_score = sum(
        item["maximum_score"]
        for item in component_scores.values()
    )

    score_pct = (
        (total_score / maximum_possible_score) * 100
        if maximum_possible_score
        else 0.0
    )

    eligible = (
        security["eligible"]
        and not rejection_reasons
    )

    selected = eligible and score_pct >= threshold_pct

    if selected:
        reason_codes.append("SCOUT_SELECTED")
    elif eligible:
        rejection_reasons.append("BELOW_SELECTION_THRESHOLD")
    else:
        reason_codes.append("SCOUT_REJECTED")

    return {
        "ticker": ticker,
        "timestamp": timestamp,
        "eligible": eligible,
        "selected": selected,
        "rank": None,
        "total_score": total_score,
        "maximum_possible_score": maximum_possible_score,
        "score_pct": score_pct,
        "component_scores": component_scores,
        "raw_values": {
            "last_price": market_data.get("last_price"),
            "bid": market_data.get("bid"),
            "ask": market_data.get("ask"),
            "spread_pct": calculate_spread_pct(
                market_data.get("bid"),
                market_data.get("ask"),
            ),
            "relative_volume": market_data.get("relative_volume"),
            "premarket_dollar_volume": market_data.get(
                "premarket_dollar_volume"
            ),
            "average_daily_dollar_volume": market_data.get(
                "average_daily_dollar_volume"
            ),
        },
        "guardrails": guardrails,
        "catalysts": [],
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "rejection_reasons": list(
            dict.fromkeys(rejection_reasons)
        ),
    }


def rank_candidates(
    candidates: list[dict[str, Any]],
) -> None:
    """
    Rank only selected candidates.

    Ranking is deterministic:
    1. score percentage descending
    2. ticker ascending as deterministic tie-breaker
    """
    selected = [
        candidate
        for candidate in candidates
        if candidate["selected"]
    ]

    selected.sort(
        key=lambda item: (
            -item["score_pct"],
            item["ticker"],
        )
    )

    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank


def run_scout(
    snapshot: dict[str, Any],
    threshold_pct: float = 85.0,
) -> dict[str, Any]:
    """Run deterministic Scout V1 against one frozen snapshot."""
    try:
        validate_contract("historical_snapshot", snapshot)
    except ContractError as exc:
        raise ScoutError(
            f"Scout received invalid historical snapshot: {exc}"
        ) from exc

    config = load_scout_config()

    candidates = [
        score_security(
            security=security,
            config=config,
            threshold_pct=threshold_pct,
            timestamp=snapshot["freeze_timestamp"],
        )
        for security in snapshot["securities"]
    ]

    rank_candidates(candidates)

    result = {
        "replay_id": snapshot["replay_id"],
        "scout_version": snapshot["scout_version"],
        "snapshot_timestamp": snapshot["freeze_timestamp"],
        "scoring_threshold_pct": threshold_pct,
        "eligible_universe_count": sum(
            1 for item in candidates if item["eligible"]
        ),
        "qualifying_candidate_count": sum(
            1 for item in candidates if item["selected"]
        ),
        "candidates": candidates,
    }

    try:
        validate_contract("scout_output", result)
    except ContractError as exc:
        raise ScoutError(
            f"Scout produced invalid output: {exc}"
        ) from exc

    return result
