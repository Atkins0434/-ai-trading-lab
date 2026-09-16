# Historical Replay Accelerator

PR #13 accelerates historical research replays. It does not add brokerage,
paper-order, or live-order capability.

## Execution model

The command accepts either explicit sessions or an inclusive date range:

```bash
python -m trainer.run_multi_day_trainer \
  --start-date 2024-09-16 \
  --end-date 2026-09-15 \
  --max-workers 2 \
  --exploration-top-k 3 \
  --universe-mode historical_research
```

`--dates-csv` remains available for targeted replays. Calendar generation
removes weekends, regular U.S. equity exchange holidays, and known
exchange-wide exceptional closures. Extra closures or sessions can be supplied
to the calendar function when a historical correction is required.

Workers are bounded to 1-8 concurrent trading dates. All provider requests
share one thread-safe adaptive limiter configured by `config/massive_plan.json`.
The Developer plan's null REST ceiling applies no proactive pacing. HTTP 429,
transient server responses, and timeout-class responses still use bounded
exponential retries and honor `Retry-After`.

The Massive Developer profile is:

- `--max-workers 2`
- `rest_calls_per_minute: null`
- ten years of history
- flat-file access available

Setting `rest_calls_per_minute` to a positive number restores the same shared
pacing used for finite plans. `--requests-per-minute` remains an optional local
override for controlled tests or a future plan downgrade.

`--exploration-top-k` adds only below-threshold, guardrail-eligible research
names. Qualifying-threshold candidates always consume simulated position slots
first. Benchmark artifacts persist three independent views:

- `scout_summary`: qualifying-threshold candidates only; this is the sole
  source of Scout return, P&L, win rate, and WIN/TIE/MISS;
- `exploration_summary`: exploration candidates only; and
- `combined_summary`: a reference view that is never used for the verdict.

The fixed-seed random baseline draw size equals the qualifying candidate count,
so changing `exploration_top_k` cannot change the qualifying Scout comparison.

For a deterministic non-research smoke test only, use
`--universe-mode ci_fixture --tickers ...`. Static symbols are rejected in
`historical_research` mode. See `UNIVERSE_INTEGRITY.md` for the full audit,
daily manifest contract, delisted-security convention, provider limitations,
and evidence gates.

## Resume and cache guarantees

The run persists an atomic date queue at `replay_queue.json`. Each day also
persists an atomic ticker queue and a completed result checkpoint per ticker.
Interrupted `IN_PROGRESS` tasks are returned to `PENDING` on restart; completed
work is reused. A partial day keeps its parent date task retryable so a later
run reuses completed ticker checkpoints and claims only the unfinished tickers.

Historical responses are stored through the checksum-protected cache. Cache
manifests record the source count, canonical record count, removed duplicate
count, provider/feed identity, and SHA-256 content digest. Loads revalidate the
digest and deduplication arithmetic before data reaches Scout.

## Historical news and sentiment

The accelerator backfills provider news per date/ticker, normalizes recognized
catalyst categories and ticker-specific sentiment, and writes:

- `historical_news_backfill.json`
- `catalyst_shadow_metrics.json`

Provider availability must be explicitly timestamped at or before the 07:00
ET replay freeze. Publication time alone is not treated as proof that an item
was available to the provider at the decision time. Catalyst results remain a
shadow layer and do not change Production Scout scoring.

## Dataset isolation

Dates are assigned chronologically to 60% development, 20% validation, and 20%
holdout partitions. Small probes of fewer than five sessions remain development
only. A saved experiment freezes its assignments; extending a date range in a
way that would reassign an existing date requires a new output directory.

- Development data may create and accumulate hypotheses.
- Validation data measures frozen hypotheses but cannot train them.
- Holdout data is excluded unless `--unlock-holdout` is explicitly supplied.
- Thirty independent development date/ticker observations unlock validation;
  they never authorize automatic promotion or Production Scout mutation.

## Output and reporting

`trainer_run_state.json` accumulates cross-day feature evidence and records
development, validation, and holdout observations separately. The generated
four-page `trainer_summary_report.pdf` presents:

- separately compounded qualifying, exploration, combined, and deterministic
  random-baseline return;
- separately reported qualifying, exploration, and combined realized P&L,
  with money-capture percentage based only on qualifying Scout P&L;
- maximum drawdown and daily-return volatility;
- positive-day rate and diagnostic-only top-ten capture;
- a partition-labeled daily replay ledger;
- unreachable-mover and execution-policy review sections;
- the top three low-scoring metric threshold reviews;
- cumulative hypotheses sourced only from visible low-score and guardrail misses;
- cross-day shadow-feature evidence; and
- the enforced safety and dataset-isolation boundaries.

All outputs remain research-only. Real-money execution, automatic production
mutation, and automatic promotion are disabled.
