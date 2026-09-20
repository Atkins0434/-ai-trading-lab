## Catalyst freshness correction

Metric 21 now gates relevance by configured event type and uses Massive insights sentiment only for the sentiment-dependent types. Keywords remain a CSV diagnostic hint and cannot change scores. Coverage adds under-12h/24h shares and pooled freshest-relevant-event age quartiles. All other metric calculations and the freshness ladder are unchanged; the backlog dataset version increases.

## Research Alpha v1.3 — scored news catalysts

Massive news is cached once per ticker/day for scorable and shadow-scored names, admitted with a documented 15-minute publication latency assumption, and scored through catalyst metrics 19–21. The reachable rubric is now 25 metrics / 100 points; alpha12_score_pct remains the original 12-metric comparison and score_pct_fixed120 remains points / 120. News coverage is reported by event, source, and price tier; the backlog dataset version increases. Existing 22 metrics, reversal, guardrails, execution, benchmark, and classification logic are unchanged. SEC filings remain out of scope.

## Research Alpha v1.2 — sector context and availability

Alpha adds metrics 16–18 using cached SIC major groups and flat-file SPY/QQQ/IWM bars. Selection now uses the reachable denominator (22 implemented metrics, 88 points), with missing ticker data still earning no points; score_pct_fixed120 remains a reference and alpha12_score_pct remains comparable across rubric versions. Sector labels and thresholds are research assumptions. The backlog dataset version is incremented; reversal retains its original 12 metrics.

# Changelog

## PR #35 — execution_costs_v1.0

Historical fills now carry additive commission, regulatory-fee, and tiered
slippage costs with net P&L, return, capture, and parallel net verdicts across
policies and cohorts. Gross fills and the default gross verdict are unchanged.
Reports and CSVs show costs alongside gross and net returns. Older results remain
readable with unavailable net values; dataset_version and rubric_version do not
change. Previously mislabeled policy-summary net P&L is now cost-adjusted, with
the original gross value retained under explicit gross fields.

## PR #34 — alpha_v1.1_19m

Research Scout Alpha v1.1 adds seven research-only metrics from already-loaded
snapshot data: three Cutler window RSI metrics, ATR percent, absolute premarket
range, and two dollar-volume quality metrics. Scores before this change are out
of 48 points; scores after it are out of 76 with missing metrics contributing no
points and the denominator remaining fixed. `alpha12_score_pct` is the comparable
figure across this boundary, with `alpha12_total_score` preserving the original
12-metric total. The original thresholds, 70% selection gate, guardrails, and
12-metric reversal rubric are unchanged. The backlog dataset version advances
to 2 so historical units rerun under the new rubric. RSI uses arithmetic mean
gains/losses within each real-bar window, recorded as `rsi_cutler_window_v1.0`.
