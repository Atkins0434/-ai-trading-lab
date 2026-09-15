# Scout V1 Scoring Model

## Status

**Version:** Scout V1\
**Scoring architecture:** 30 scored metrics x 4 points = 120 maximum
points\
**Metric scale:** 0, 1, 2, 3, or 4 points\
**Purpose:** Production Scout candidate scoring and selection

Scout finds and scores opportunities. Scout Trainer proposes and
validates improvements. Trade Engine handles deterministic execution
after Scout selection. Audit explains what each layer did.

## Core Scoring Formula

``` text
30 metrics x 4 points = 120 maximum points
Scout Score % = Points Earned / 120 x 100
```

  Threshold     Points out of 120
  ----------- -------------------
  85%                       102.0
  90%                       108.0
  95%                       114.0
  96%                       115.2
  97%                       116.4
  98%                       117.6

Thresholds to validate historically and in shadow testing are 85%, 90%,
95%, 96%, 97%, and 98%. The long-term objective is to use the highest
threshold that still reliably produces approximately 5-10 viable
candidates per scan.

Hard guardrails override score.

# A. Price & Volume Dynamics

**Maximum: 48 points**

  \#   Metric                               Max
  ---- ---------------------------------- -----
  1    Relative volume                        4
  2    Price slope, 15 minute                 4
  3    Price slope, 30 minute                 4
  4    Price slope, 60 minute                 4
  5    Volume slope, 15 minute                4
  6    Volume slope, 30 minute                4
  7    Volume slope, 60 minute                4
  8    Premarket gap strength                 4
  9    Price position vs premarket VWAP       4
  10   Premarket trend consistency            4
  11   Price/volume confirmation              4
  12   Premarket range expansion              4

Price and volume slopes use ordinary least-squares linear regression
over 15, 30, and 60 minute windows. The 60-minute window provides
context and the 15-minute window provides the strongest recent trigger
information. Non-positive slopes are strongly disfavored.

Volume acceleration uses the same 15/30/60-minute framework and remains
independently auditable.

## Versioned V1 Baseline Thresholds

These are deterministic Trainer-baseline parameters, not promoted claims of
predictive value. Scout Trainer must challenge them against historical and
blind holdout results before Production promotion.

| Metric input | 1 point | 2 points | 3 points | 4 points |
|---|---:|---:|---:|---:|
| Price slope (%/minute) | 0.015 | 0.030 | 0.050 | 0.075 |
| Volume slope (%/minute) | 0.25 | 0.50 | 1.00 | 2.00 |
| Premarket gap (%) | 1.00 | 2.00 | 4.00 | 6.00 |
| Price above premarket VWAP (%) | 0.25 | 0.50 | 1.00 | 2.00 |
| Positive one-minute closes (%) | 50 | 60 | 70 | 80 |
| Premarket range / average daily range | 1.00x | 1.25x | 1.50x | 2.00x |

Price/volume confirmation scores the count of 15/30/60-minute windows in
which both normalized price and volume slopes are positive. All three aligned
windows score 3; all three plus the versioned acceleration override score 4.

Premarket VWAP uses the typical price `(high + low + close) / 3`, weighted by
each one-minute bar's volume. Trend consistency is the percentage of positive
close-to-close one-minute changes. Premarket range expansion compares the
observed premarket high-low range as a percentage of the prior close with the
historical average daily range percentage.

## Short-Term Acceleration Override

The override is a rule inside the price-momentum family, not a 31st
metric.

``` text
60-minute price slope > 0
AND 15-minute slope >= 2 x 60-minute slope
AND 30-minute slope >= 2 x 60-minute slope
```

The 2.0x multiple is a versioned parameter that Scout Trainer may later
challenge through historical validation.

# B. Relative Strength & Leadership

**Maximum: 24 points**

  \#   Metric                                                Max
  ---- --------------------------------------------------- -----
  13   Relative strength vs appropriate index, 15 minute       4
  14   Relative strength vs appropriate index, 30 minute       4
  15   Relative strength vs appropriate index, 60 minute       4
  16   Sector strength                                         4
  17   Stock leadership vs sector                              4
  18   Broad-market regime alignment                           4

Scout uses deterministic benchmark mapping. Sector strength and stock
leadership remain distinguishable so Scout can tell whether a stock is
merely riding sector movement or leading it.

# C. Catalyst & Attention

**Maximum: 24 points**

  \#   Metric                                     Max
  ---- ---------------------------------------- -----
  19   Catalyst quality                             4
  20   Catalyst verification confidence             4
  21   Catalyst freshness/relevance                 4
  22   Call-options confirmation                    4
  23   Analyst revision strength                    4
  24   Positive social-attention acceleration       4

Only verified, relevant news or filings count as catalysts. Syndicated
duplicates must not multiply evidence. SEC filings and earnings releases
receive the strongest treatment under the baseline design.

Call activity is confirmation only. Abnormal buy-side put activity is a
hard guardrail rather than a negative scored metric.

Social is confirmation-only:

``` text
Negative attention = 0
Neutral attention = 0
Positive accelerating attention = 1-4
```

# D. Volatility & Tradability Quality

**Maximum: 24 points**

  \#   Metric                                  Max
  ---- ------------------------------------- -----
  25   ATR % opportunity                         4
  26   Current range % opportunity               4
  27   Spread quality                            4
  28   Order-book depth quality                  4
  29   Average daily dollar-volume quality       4
  30   Premarket dollar-volume quality           4

Scout seeks opportunity and should not automatically punish volatility
merely because a stock is volatile. Some tradability characteristics are
both scored and guarded.

# Hard Guardrails

Hard guardrails sit outside the 120-point score. A hard rejection
overrides any score.

## Liquidity

Failure of required average daily dollar-volume or premarket
dollar-volume minimums causes rejection.

## Bid-Ask Spread

``` text
spread_pct = (ask - bid) / midpoint x 100
Penalty begins: 0.20%
Hard rejection: 0.40%
```

## Order-Book Depth

``` text
>= 4x intended order size = full credit
2x to <4x = partial credit
<2x = HARD REJECT
```

The same principle applies to entry and exit feasibility.

## Abnormal Put Volume

Use abnormal buy-side put volume, not generic put/call ratio.

``` text
>= 4x normal buy-side put volume
Before entry = REJECT
Already in position = IMMEDIATE FULL LIQUIDATION
```

## Dilution / Offering Risk

A qualifying active dilution or offering condition is a hard rejection.

## Halt / Distortion Risk

An unacceptable halt or market-distortion condition is a hard rejection.

## Tradability

Unavailable, restricted, ineligible, or otherwise non-executable
securities are rejected.

# Context-Only Variables

Float and short interest are retained for context and postmortem
analysis but are not scored in Scout V1.

# Missing Data Policy

Missing data and zero are different.

``` text
0 points = metric was observed and supplied no positive evidence
MISSING = metric could not be reliably calculated or observed
```

Every metric must define an explicit missing-data policy such as
REQUIRED, REJECT, or NO_SCORE_WITH_RENORMALIZATION. Missing values must
never silently become zero. Any renormalization must be deterministic,
versioned, auditable, and tested for score inflation.

# Candidate Selection

Production Scout should normally surface approximately 5-10 candidates
per scan. The downstream Trade Engine may reduce this further based on
capital, tradability, and position limits.

Scout never manufactures candidates to satisfy a target count. No
qualifying stocks means no selections.

# Scout Audit Requirements

Every replay should preserve enough information to reconstruct the
decision, including:

-   Replay ID and trading date
-   Freeze timestamp
-   Scout and feature-registry versions
-   Universe and data-feed versions
-   Selection threshold
-   Raw input values
-   All 30 component scores
-   Points earned and 120-point maximum
-   Normalized score percentage
-   Guardrail observations, thresholds, actions, and reason codes
-   Final candidate decision and rank
-   Determinism hash

Scout Audit explains why Scout selected, rejected, or ranked a security.
It remains separate from Trade Audit.

# Separation of Responsibilities

## Scout

Owns observations used for selection, feature calculations, 0-4
component scoring, the 120-point total, normalized score, selection
threshold, ranking, pre-entry guardrails, and candidate selection.

## Scout Trainer

Owns historical replay, grading, benchmark comparison, miss analysis,
feature and parameter proposals, Challenger creation, out-of-sample
validation, blind holdout testing, and promotion recommendations.
Trainer never directly mutates Production Scout.

## Trade Engine

Owns deterministic execution after Scout selection: buying-power
validation, sizing, entry feasibility, position limits, trailing stops,
profit targets, abnormal-put liquidation, mandatory session liquidation,
realized P&L, and execution state transitions. Trade Engine does not own
Scout scoring.

## Audit

Scout Audit explains Scout. Trade Audit explains Trade Engine. Audit
itself makes no decisions.

# Baseline Trade Engine Policy

``` text
Example strategy capital: $2,500
Maximum allocation per stock: 20%
Maximum fully allocated positions: 5
Trailing stop: 1%
Same-day profit take: 20%
Abnormal buy-side put volume: 4x normal -> immediate liquidation
Mandatory session-end liquidation: enabled
```

There is no forced minimum number of positions.

# Historical Replay Contract

Initial replay begins January 2, 2018 with a 07:00 America/New_York
point-in-time freeze. Scout cannot receive information generated or
published after the freeze during selection.

The first milestone is a manually reviewed single-day replay for January
2, 2018. After that succeeds, replay expands to days 2-10 and then
scales. Blind holdout dates eventually remain unseen during Trainer
development and feature discovery.

# Trainer Feature Promotion Requirements

A proposed feature requires at least:

-   30 independent historical occurrences
-   Multiple years and market regimes
-   Meaningful out-of-sample improvement
-   No material degradation elsewhere
-   No unacceptable redundancy or double-counting
-   Availability in the intended live environment
-   Deterministic definition
-   Complete feature card
-   Blind holdout pass
-   Explicit Champion/Challenger promotion decision

Feature interdependencies must be tracked.

# Versioning

Every replay should record Scout version, Trainer version, Trade Engine
policy version, feature-registry version, universe version, data
provider/feed version, replay timestamp, and repository commit hash when
available.

Changes to scoring definitions, thresholds, guardrails, missing-data
behavior, or feature formulas require versioned changes.

# Current Implementation Status

The repository contains a Trainer baseline used to validate architecture and
plumbing. The synthetic January 2, 2018 fixture and enriched metric tests
currently demonstrate:

-   Relative-volume scoring
-   Catalyst scoring
-   Liquidity and spread rejection
-   Candidate ranking and threshold behavior
-   Point-in-time replay enforcement
-   Scout PDF reporting and Scout Audit
-   Determinism hashing
-   15/30/60-minute price and volume slope calculations
-   Short-term acceleration override calculation
-   All twelve Price & Volume Dynamics calculations and component scores
-   Checksum-protected historical provider caching
-   A Scout-to-Trade-Engine single-day outcome path with MFE, MAE,
    maximum capturable move, and capture ratio

The existing synthetic `7 / 8 = 87.5%` score is a partial-development
score, not completed Production Scout V1.

Production Scout V1 is intended to use 30 metrics and 120 maximum
points.

# Architecture Summary

``` text
Historical / Live Market Data
            |
            v
          SCOUT
  Observe -> Calculate -> Score
       -> Apply Guardrails
       -> Rank -> Select
            |
            v
       TRADE ENGINE
   Deterministic Execution
            |
            v
          OUTCOME
            |
            v
      SCOUT TRAINER
 Replay -> Grade -> Diagnose
 -> Propose -> Validate
 -> Challenger -> Promotion Gate
```

**Governing principle:** Scout decides what is worth trading. Trade
Engine decides how a selected trade is executed. Trainer learns how
Scout can improve. Audit proves what happened.
