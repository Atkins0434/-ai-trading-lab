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
