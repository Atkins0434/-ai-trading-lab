from __future__ import annotations

from pathlib import Path
from math import ceil
from typing import Any

from trainer.validate_contracts import (
    ContractError,
    load_json,
    validate_contract,
)
from trainer.price_volume import (
    PriceVolumeError,
    calculate_price_volume_metrics,
)


ROOT = Path(__file__).resolve().parent.parent
SCOUT_CONFIG_PATH = ROOT / "config" / "scout_v1.json"
FEATURE_REGISTRY_PATH = ROOT / "config" / "feature_registry_v1.json"
FIXED_MAXIMUM_POINTS = 120


class ScoutError(Exception):
    """Raised when Scout cannot deterministically score a snapshot."""


def load_scout_config() -> dict[str, Any]:
    """Load the frozen Scout V1 configuration."""
    try:
        return load_json(SCOUT_CONFIG_PATH)
    except ContractError as exc:
        raise ScoutError(f"Unable to load Scout configuration: {exc}") from exc


def load_feature_registry() -> dict[str, Any]:
    """Load and validate the immutable Scout V1 metric registry."""
    try:
        registry = load_json(FEATURE_REGISTRY_PATH)
    except ContractError as exc:
        raise ScoutError(f"Unable to load feature registry: {exc}") from exc

    metrics = registry.get("metrics", [])
    ids = [metric.get("id") for metric in metrics]
    numbers = [metric.get("number") for metric in metrics]

    if (
        registry.get("metric_count") != 30
        or registry.get("maximum_points") != FIXED_MAXIMUM_POINTS
        or len(metrics) != 30
        or len(set(ids)) != 30
        or numbers != list(range(1, 31))
    ):
        raise ScoutError(
            "Feature registry must contain exactly 30 uniquely named, "
            "ordered metrics and a fixed 120-point maximum."
        )

    return registry


def observed_value(observation: dict[str, Any] | None) -> float | None:
    """Return a point-in-time observation value without erasing provenance."""
    if observation is None:
        return None
    return observation.get("value")


def minimum_points_for_threshold(threshold_pct: float) -> int:
    """Convert a percentage threshold to attainable integer Scout points."""
    if threshold_pct < 0 or threshold_pct > 100:
        raise ScoutError("Selection threshold must be between 0 and 100.")
    return ceil((threshold_pct / 100) * FIXED_MAXIMUM_POINTS)


def missing_component(metric: dict[str, Any]) -> dict[str, Any]:
    """Represent missing data explicitly; missing is never observed zero."""
    return {
        "status": "MISSING",
        "score": None,
        "maximum_score": 4,
        "raw_value": None,
        "as_of_timestamp": None,
        "reason_code": "METRIC_NOT_IMPLEMENTED",
        "calculation_version": "unimplemented",
    }


def observed_component(
    score: int,
    raw_value: Any,
    as_of_timestamp: str | None,
    reason_code: str,
    calculation_version: str,
) -> dict[str, Any]:
    if score not in range(5):
        raise ScoutError("Component scores must be integers from 0 through 4.")
    return {
        "status": "OBSERVED",
        "score": score,
        "maximum_score": 4,
        "raw_value": raw_value,
        "as_of_timestamp": as_of_timestamp,
        "reason_code": reason_code,
        "calculation_version": calculation_version,
    }


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


def _missing_market_data_guardrail(
    policy: str,
    reason_code: str,
    threshold: float | str | None,
) -> dict[str, Any]:
    """Return the configured result for an unavailable market-data check."""
    if policy not in {"NOT_EVALUATED", "REJECT"}:
        raise ScoutError(
            "Missing market-data policy must be NOT_EVALUATED or REJECT."
        )

    rejected = policy == "REJECT"
    return {
        "passed": False if rejected else None,
        "action": "REJECT" if rejected else "NOT_EVALUATED",
        "observed_value": None,
        "threshold": threshold,
        "reason_code": reason_code,
    }


def score_relative_volume(relative_volume: float | None) -> int:
    """
    Score relative volume on the Scout 0-4 scale.

    This is intentionally conservative and deterministic.
    """
    if relative_volume is None:
        return 0

    if relative_volume < 1.0:
        return 0
    if relative_volume < 1.25:
        return 1
    if relative_volume < 1.5:
        return 2
    if relative_volume < 2.0:
        return 3

    return 4


def score_catalyst(
    security: dict[str, Any],
) -> int:
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
        return 0
    if weighted_events == 1:
        return 1
    if weighted_events == 2:
        return 2
    if weighted_events == 3:
        return 3

    return 4


def evaluate_liquidity_guardrail(
    market_data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate Scout's deterministic liquidity floor."""
    rules = config["liquidity"]

    avg_dollar_volume = observed_value(
        market_data.get("average_daily_dollar_volume")
    )
    premarket_dollar_volume = observed_value(
        market_data.get("premarket_dollar_volume")
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
    rules = config["spread"]
    bid = observed_value(market_data.get("bid"))
    ask = observed_value(market_data.get("ask"))
    reject_at = rules["hard_reject_pct"]

    if bid is None or ask is None:
        return _missing_market_data_guardrail(
            rules.get("missing_quotes_policy", "NOT_EVALUATED"),
            "SPREAD_QUOTES_UNAVAILABLE",
            reject_at,
        )

    spread_pct = calculate_spread_pct(bid, ask)
    penalty_start = rules["penalty_starts_pct"]

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


def evaluate_order_book_depth_guardrail(
    market_data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the minimum required order-book depth when it is available."""
    rules = config["order_book_depth"]
    reject_below = rules["hard_reject_below_multiple"]
    depth_multiple = observed_value(
        market_data.get("order_book_depth_multiple")
    )

    if depth_multiple is None:
        return _missing_market_data_guardrail(
            rules.get("missing_depth_policy", "NOT_EVALUATED"),
            "ORDER_BOOK_DEPTH_UNAVAILABLE",
            reject_below,
        )

    if (
        not isinstance(depth_multiple, (int, float))
        or isinstance(depth_multiple, bool)
        or depth_multiple < 0
    ):
        return {
            "passed": False,
            "action": "REJECT",
            "observed_value": depth_multiple,
            "threshold": reject_below,
            "reason_code": "ORDER_BOOK_DEPTH_INVALID",
        }

    passed = depth_multiple >= reject_below
    return {
        "passed": passed,
        "action": "PASS" if passed else "REJECT",
        "observed_value": depth_multiple,
        "threshold": reject_below,
        "reason_code": (
            "ORDER_BOOK_DEPTH_OK"
            if passed
            else "ORDER_BOOK_DEPTH_HARD_REJECT"
        ),
    }


def score_security(
    security: dict[str, Any],
    config: dict[str, Any],
    threshold_pct: float,
    timestamp: str,
    registry: dict[str, Any],
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
        "order_book_depth": evaluate_order_book_depth_guardrail(
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
        metric["id"]: missing_component(metric)
        for metric in registry["metrics"]
    }

    relative_volume_observation = market_data.get("relative_volume")
    relative_volume = observed_value(relative_volume_observation)
    if relative_volume_observation is not None and relative_volume is not None:
        component_scores["relative_volume"] = observed_component(
            score=score_relative_volume(relative_volume),
            raw_value=relative_volume,
            as_of_timestamp=relative_volume_observation["as_of_timestamp"],
            reason_code="RELATIVE_VOLUME_SCORE",
            calculation_version="relative_volume_v1.0",
        )

    try:
        price_volume_metrics = calculate_price_volume_metrics(
            security,
            config,
        )
    except PriceVolumeError as exc:
        raise ScoutError(
            f"Unable to calculate Price & Volume metrics for {ticker}: {exc}"
        ) from exc

    for metric_id, metric in price_volume_metrics.items():
        component_scores[metric_id] = observed_component(
            score=metric.score,
            raw_value=metric.raw_value,
            as_of_timestamp=metric.as_of_timestamp,
            reason_code=metric.reason_code,
            calculation_version="price_volume_v1.0",
        )

    catalyst_events = security.get("news", []) + security.get("filings", [])
    component_scores["catalyst_quality"] = observed_component(
        score=score_catalyst(security),
        raw_value={
            "news_count": len(security.get("news", [])),
            "filing_count": len(security.get("filings", [])),
        },
        as_of_timestamp=max(
            (event["published_timestamp"] for event in catalyst_events),
            default=timestamp,
        ),
        reason_code="CATALYST_QUALITY_SCORE",
        calculation_version="catalyst_quality_v1.0",
    )

    total_score = sum(
        item["score"] or 0
        for item in component_scores.values()
    )

    maximum_possible_score = FIXED_MAXIMUM_POINTS
    score_pct = (total_score / FIXED_MAXIMUM_POINTS) * 100
    threshold_points = minimum_points_for_threshold(threshold_pct)

    eligible = (
        security["eligible"]
        and not rejection_reasons
    )

    selected = eligible and total_score >= threshold_points

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
        "threshold_points": threshold_points,
        "component_scores": component_scores,
        "raw_values": {
            "last_price": observed_value(market_data.get("last_price")),
            "bid": observed_value(market_data.get("bid")),
            "ask": observed_value(market_data.get("ask")),
            "spread_pct": calculate_spread_pct(
                observed_value(market_data.get("bid")),
                observed_value(market_data.get("ask")),
            ),
            "relative_volume": relative_volume,
            "premarket_dollar_volume": observed_value(
                market_data.get("premarket_dollar_volume")
            ),
            "average_daily_dollar_volume": observed_value(
                market_data.get("average_daily_dollar_volume")
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
    registry = load_feature_registry()

    candidates = [
        score_security(
            security=security,
            config=config,
            threshold_pct=threshold_pct,
            timestamp=snapshot["freeze_timestamp"],
            registry=registry,
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
