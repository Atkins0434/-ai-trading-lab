SCOUT TRAINER

Historical Learning, Validation & Promotion Specification

Status: V0.1 --- Implementation Baseline
Purpose: Train, test, and improve Production Scout using historical
market data without allowing the training system to alter Production
Scout directly.

1. Mission

Scout Trainer is the historical-learning subsystem for the agentic
trading platform.

Its job is to:

Replay historical trading days using only information available at
the historical decision time.

Run the current candidate Scout version against that frozen
information set.

Simulate the platform's deterministic execution policy against the
resulting selections.

Compare Scout's realized results against the best opportunities in
the same eligible market universe.

Identify misses without excusing them.

Propose new measurable features, parameters, interactions, or
threshold changes that could have improved selection.

Validate those proposals across independent historical observations
and market regimes.

Produce a candidate Scout version for champion/challenger testing.

Never modify Production Scout directly.

The Trainer learns. Production Scout trades from an approved, frozen
specification.

2. Core Architecture

Scout Trainer

Historical only. Performs replay, postmortem analysis, feature
discovery, validation, versioning, and candidate generation.

Production Scout

Forward-looking only. Uses the currently approved feature set and
deterministic scoring rules to identify current candidates.

Deterministic Execution Layer

Owns position sizing, entry eligibility, exits, and liquidation. It does
not improvise. Scout Trainer must know the exact version of this policy
so historical picks can be graded according to how the actual system
would have traded them.

Promotion Boundary

Trainer cannot edit Production Scout. Candidate versions must pass
validation and blind champion/challenger testing before promotion.

3. Historical Replay Protocol

Historical Start

Initial walk-forward dataset begins with the first regular U.S. trading
session of 2018: January 2, 2018.

Target historical depth: approximately eight years, subject to
point-in-time data availability.

Decision-Time Freeze

For the morning session:

07:00 America/New_York

At the moment a historical replay begins, the system must behave as
though nothing after the freeze timestamp exists.

No: - future prices, - future volume, - later news, - revised filings
unavailable at the time, - end-of-day rankings, - future social data, -
later analyst actions, - or derived features containing future
information.

This rule is absolute.

Midday Session

Trainer should also support testing a two-session strategy:

Morning Scout run

Deterministic liquidation around the defined midday boundary

New Scout scan using only information available at the second freeze

Afternoon positions

Mandatory end-of-day liquidation

The single-session and two-session approaches must be tested side by
side. The better approach is determined by realized results, turnover,
drawdown, and stability rather than assumption.

4. Eligible Market Universe

Trainer and benchmark must use the same eligible universe.

Initial universe:

U.S.-listed common stocks

Major exchanges

Exclude OTC

Exclude shells and clearly non-operating securities

Focus on small-cap through mid-cap opportunity

Final market-cap boundaries remain configurable and should be
validated historically

Smaller names remain eligible only when all liquidity, spread,
depth, dilution, and execution guardrails are satisfied

The system should not automatically exclude a stock solely because it is
volatile. Volatility is part of the opportunity being sought.

5. Baseline Scout V1 Inputs

Trainer begins with the agreed Scout V1 parameters. These are the
baseline, not eternal truths.

5.1 Premarket Price/Volume Composite

Price movement and volume are treated as interacting signals rather than
independent opportunities to double-count the same behavior.

Store the underlying price and volume components separately for
auditability, but combine them for scoring.

Volume includes: - actual volume threshold, - relative volume versus the
stock's normal comparable period, - dollar-volume/liquidity floors.

Relative-volume multiplier: - Below normal: 0× - At/around normal:
1× - At least 1.5× normal: 2×

The adjusted volume signal is blended with the price-movement score.

A large price move with inadequate volume can therefore collapse toward
zero. A smaller price move supported by exceptional volume can receive
meaningful credit.

All formulas must be deterministic.

5.2 Price Momentum / Slope

Use linear regression:

y = mx + b

Measure normalized price slope over: - 15 minutes - 30 minutes - 60
minutes

The 60-minute slope provides context. The 15-minute slope is the fast
trigger.

Conservative scoring is required.

Baseline principles: - Non-positive momentum is strongly disfavored. - A
sudden 15-minute spike after a prolonged decline must not automatically
be interpreted as durable momentum. - 30- and 60-minute context must
normally confirm the move. - A hard-coded acceleration override may
award the maximum momentum score when the 15- and 30-minute slopes are
dramatically stronger than the positive 60-minute slope. - The
acceleration override must remain formula-based. No LLM discretion.

5.3 Volume Acceleration

Apply the same 15/30/60 slope framework to volume.

The question is not merely whether volume exists, but whether
participation is increasing as the move develops.

5.4 Relative Market Strength

Benchmark a stock against the appropriate index rather than applying one
universal index.

Examples: - S&P 500 constituent → S&P 500 benchmark - Nasdaq-100
constituent → Nasdaq-100 benchmark - Otherwise use an explicitly mapped
appropriate broad-market benchmark

Mapping must be deterministic.

5.5 Sector Strength + Stock Leadership

Combine sector sympathy and relative sector strength into one metric to
prevent double counting.

Measure: 1. Whether the relevant sector/peer group is strengthening. 2.
Whether the stock is leading that movement.

5.6 Dollar Volume / Liquidity

Use deterministic minimum liquidity requirements. Thin names can be
rejected regardless of attractive percentage movement.

5.7 Bid-Ask Spread

Measure percentage spread:

Spread % = (Ask - Bid) / Midpoint × 100

Where:

Midpoint = (Ask + Bid) / 2

Rules: - Penalty begins at 0.20% - Hard rejection at 0.40%

5.8 Order-Book Depth

Evaluate executable depth relative to intended order size on both entry
and exit sides.

Rules: - ≥ 4× intended order size: full credit - 2× to <4×: 50%
credit - <2×: hard rejection

5.9 Catalyst / Verified Information

Catalyst scoring must rely on verifiable information, not narrative
intuition.

Rules: - One qualifying story per approved reputable source -
Duplicate/syndicated stories do not count multiple times - SEC filings
and earnings releases receive double weighting - All information must
have been available before the replay timestamp

5.10 Options Confirmation / Rejection

Options flow is not a general weighted score.

Calls: abnormal buy-side call activity is a confirmation signal.

Puts: abnormal buy-side put volume is asymmetric and defensive.

Hard rule: - ≥4× normal buy-side put volume: stock is disqualified
before entry - If already held and the threshold occurs: deterministic
execution layer liquidates the position

This uses absolute abnormal put activity, not a put/call ratio.

5.11 Volatility

Track: - ATR as a percentage of price - Current intraday high-low range
as a percentage of price

The goal is sufficient movement to create opportunity without blindly
rewarding disorder.

5.12 Price Continuity / Chop

Choppy price action receives only a small penalty.

Volatility and disorder must not be over-penalized because volatile
movement is part of the target strategy.

5.13 Analyst Revisions

Low-weight confirmation only: - Upward revisions: small positive - Flat:
neutral - Downward revisions: small negative

5.14 Dilution / Offerings

Active dilution risk, including relevant fresh offerings or ATM
activity, can operate as a hard rejection gate.

5.15 Social Attention

Social is confirmatory, not dominant.

Rules: - Negative sentiment: 0 - Neutral sentiment: 0 - Only
positive, accelerating attention may earn credit - Social cannot
overpower weak price/volume confirmation

Trainer is intentionally not told which specific social platforms or
social features must be important. It is allowed to discover
historically useful social features organically.

5.16 Halt / Distortion Risk

Repeated halts or clearly abnormal one-off prints can disqualify a
stock.

Extreme intraday range relative to normal volatility may receive a
penalty.

Context-Only Data

Examples such as float and short interest may be retained for
postmortem/context without necessarily contributing directly to the
baseline score.

6. Baseline Scoring

Scored metrics generally use:

0, 1, 2, 3, 4

where: - 0 = fails / unattractive - 4 = strongest qualifying condition

Composite metrics may combine underlying sub-scores mathematically and
normalize back to the expected range.

The denominator used for the final percentage must always be calculated
from the active scored feature set. If metrics are merged or removed,
the denominator changes automatically.

No hidden discretionary overrides.

Promotion Threshold Testing

Do not assume the ideal Scout threshold.

During historical testing, track candidate counts and results in
parallel at: - 85% - 90% - 95% - 96% - 97% - 98%

Long-term target: use the highest threshold that still reliably
surfaces approximately 5--10 viable candidates per scan, without
forcing candidates to exist.

7. Deterministic Execution Policy Used for Grading

Trainer must receive a versioned execution-policy object for every run.

Baseline rules currently include:

Maximum 20% of available strategy capital per stock

Unused allocation remains cash

No forced minimum number of positions

Maximum five fully allocated positions at 20% each

1% trailing stop per position

20% same-day profit take: immediate liquidation

4× abnormal buy-side put volume: immediate liquidation

Mandatory liquidation at the end of the applicable trading session

For a two-session test, liquidation occurs at the defined midday
boundary before the second Scout run

Pre-entry liquidity, spread, depth, tradability, and buying-power
checks are enforced

Execution must be a deterministic state machine. It acts first and
notifies afterward. It does not wait for human approval to protect a
position.

Any change to the execution policy creates a new execution-policy
version and requires comparable historical re-grading.

8. Path-Based Grading

Do not grade Scout using closing price alone.

A stock may rise substantially after selection, trigger a profitable
trailing exit, and later close negative. That can still be a successful
Scout selection.

Therefore, historical minute-level price path is a first-class object.

For each selection calculate at minimum: - Entry price/time - Maximum
Favorable Excursion (MFE) - Maximum Adverse Excursion (MAE) - Simulated
exit price/time - Exit reason - Realized return under deterministic
execution - Maximum theoretically capturable move after eligibility -
Capture ratio

This separates: - Scout quality: did it surface a tradable
opportunity? - Execution quality: how much of that opportunity did
the fixed execution policy capture?

Both use the same historical tape.

9. Daily Benchmark

At the end of each historical day, identify the top 10 movers inside
the exact same eligible universe used by Scout.

Do not compare small/mid-cap Scout against mega-cap names outside its
universe.

Run the same deterministic execution policy against the benchmark
opportunities where a fair simulation is possible.

Primary Win Condition

Scout is successful when its selected basket, under the fixed
deterministic execution policy, produces superior realized money capture
/ return relative to the benchmark basket under the same execution
assumptions.

The benchmark is not the objective. It is the measuring stick.

Scout is allowed to disagree with the day's eventual top movers.

If Scout's alternatives produce better realized results under the common
execution policy, Scout wins.

If Scout underperforms, it is a miss and triggers postmortem analysis.

No narrative excuse converts a miss into a win.

Secondary Metrics

Track: - Top-10 capture rate - Percentage of benchmark drivers
represented in Scout selections - Target driver capture: ≥70% -
Realized return - Drawdown - Win rate - Profit factor - Turnover -
MFE/MAE - Capture ratio - Number of qualifying candidates - Number
rejected by each guardrail

10. Postmortem Loop

After outcomes are revealed:

Grade Scout's selections.

Grade the benchmark under the same execution policy.

Identify benchmark opportunities Scout missed.

Determine which observable pre-decision features distinguished those
opportunities.

Determine whether existing parameters failed because of threshold,
interaction, redundancy, missing data, or missing feature.

Propose candidate improvements.

Do not promote them automatically.

A miss remains a miss. Postmortem reasoning exists to create better
future hypotheses, not to excuse historical errors.

11. Feature Discovery

Scout Trainer is allowed to discover features not present in V1.

Each proposed feature must be converted into a precise, machine-testable
definition.

No feature can be promoted because an LLM says it "looks useful."

Required Feature Card

Every candidate feature must contain:

Unique feature ID

Name

Description

Formula / deterministic definition

Required source data

First-seen date

Proposal date

Historical examples

Expected relationship to return/capture

Known dependencies

Known conflicts

Redundancy analysis

Missing-data behavior

Proposed score/weight/gate behavior

Validation status

Version history

All new parameters must be dated.

12. Feature Interdependency & Redundancy

Trainer must explicitly understand feature interactions.

For every proposed feature, evaluate: - Correlation with existing
features - Whether it is merely a transformation of an existing
feature - Whether two features double-count the same market behavior -
Whether one feature mathematically cancels or suppresses another -
Whether two related features should be blended into a composite -
Whether the interaction itself contains predictive information

Maintain a feature dependency/interaction matrix.

No silent double counting. No silent cancellation.

13. Feature Promotion Gate

A proposed feature or parameter change cannot enter a candidate
production model unless evidence exists in at least:

30 independent historical occurrences

Those occurrences must span multiple market conditions/regimes rather
than being concentrated in one event or short period.

Thirty is a minimum evidence count, not sufficient evidence by itself.

Promotion also requires: - Measurable improvement on relevant outcome
metrics - Out-of-sample improvement - No unacceptable degradation of
drawdown, liquidity, execution feasibility, or stability - No evidence
that the improvement is merely duplicated by an existing feature -
Point-in-time data availability in live production - Deterministic
definition - Complete feature card - Successful blind holdout exam

14. Champion / Challenger Promotion

Production Scout is the Champion.

Scout Trainer produces a Challenger candidate.

Before promotion: 1. Freeze both versions. 2. Run both on identical
unseen historical holdout days. 3. Use identical universe and
execution-policy versions. 4. Compare realized capture, return,
drawdown, stability, candidate availability, and benchmark performance.
5. Challenger must demonstrate a meaningful improvement rather than
winning through one outlier day. 6. Log the promotion decision.

Trainer cannot self-promote.

15. Blind Holdout Exam

Holdout dates must not be used during feature discovery or threshold
tuning.

For an acceptance test: - Select an unseen historical trading day -
Freeze the information set at the defined decision time - Run Champion
and Challenger - Record picks before revealing later market data -
Reveal the trading day - Simulate deterministic execution - Grade both
versions - Compare against the same-universe benchmark

The system should eventually support randomly selected blind holdout
days.

16. Versioning & Rollback

Every Scout specification is immutable after release.

Examples: - Scout V1.0 - Scout V1.1 candidate - Scout V1.1 promoted

Every run records: - Scout version - Trainer version - Execution-policy
version - Data-source version/feed - Feature registry version - Universe
definition - Historical timestamp - Code commit hash where available

Every promotion must be reversible.

17. Data Source Plan

Phase 1

Use Tiingo as the initial historical market-data source.

This phase is historical simulation only.

No live trading. No paper trading. No Alpaca execution dependency yet.

The first implementation should pull only a small historical sample to
prove the replay engine before attempting the full eight-year dataset.

Later Phases

After historical validation: 1. Blind holdouts 2. Alpaca paper trading
3. Compare paper results with historical expectations 4. Diagnose
slippage, latency, spread, fill, and data differences 5. Only then
consider live execution

18. First Implementation Milestone

Do not begin by processing eight years.

First milestone:

January 2, 2018

Run exactly one regular historical session through the complete
pipeline.

Validate: - Correct eligible universe - Correct 07:00 freeze - No future
leakage - Scout scores reproducibly - Picks are logged - Intraday tape
is available - Deterministic execution simulator behaves correctly -
Benchmark is calculated from the same universe - Postmortem runs -
Candidate feature proposals are recorded - No proposed feature modifies
Scout automatically

Only after this day passes review should the system advance
automatically through subsequent trading days.

19. Required Interface Contracts

All component handoffs should use strict structured schemas rather than
free-text handoffs.

Minimum contracts:

A. Historical Snapshot Input

Contains: - replay timestamp - universe version - point-in-time market
data - eligible news/filings/context - feature-registry version - Scout
version - execution-policy version

B. Scout Output

For each candidate: - ticker - timestamp - eligibility result - total
score - component scores - underlying raw values - guardrail results -
catalyst references - reason codes - rejection reasons where applicable

C. End-of-Day Outcome

Contains: - full intraday path - MFE - MAE - close - deterministic
simulated trade result - exit reason - realized return - capture ratio

D. Benchmark Result

Contains: - same-universe top movers - deterministic simulated result
for benchmark - benchmark realized return/capture - comparison against
Scout

E. Postmortem

Contains: - win/miss - missed opportunities - measurable
differentiators - existing-feature failures - candidate feature
proposals

F. Feature Proposal

Contains the complete feature-card schema.

G. Validation Result

Contains: - occurrence count - regime distribution - in-sample result -
out-of-sample result - dependency/redundancy results - risk degradation
checks - promotion eligibility

H. Promotion Decision

Contains: - Champion version - Challenger version - holdout set -
comparison metrics - decision - reason codes - timestamp

20. Build Order

Create repository structure.

Add this SCOUT_TRAINER.md.

Define strict JSON schemas for the interface contracts.

Encode baseline Scout V1 parameters.

Encode deterministic execution policy as a versioned state machine.

Connect Tiingo historical data.

Build point-in-time replay/freeze engine.

Build one-day fixture for January 2, 2018.

Generate Scout selections.

Replay intraday tape.

Simulate deterministic execution.

Build same-universe top-10 benchmark.

Grade Scout versus benchmark.

Run postmortem.

Generate feature proposals.

Validate that proposals cannot mutate Scout.

Review the entire one-day run manually.

Advance to a small multi-day batch.

Scale historical walk-forward testing.

Add blind holdouts.

Produce Champion/Challenger candidates.

Only after historical validation, connect Alpaca paper trading.

21. Design Principle

The Trainer is allowed to be curious.

Production Scout is not.

Trainer can propose, test, challenge, and learn. Production Scout
executes only an approved, versioned, deterministic specification.

Historical outcomes are the grader. Explanations create hypotheses.
Evidence earns promotion.
