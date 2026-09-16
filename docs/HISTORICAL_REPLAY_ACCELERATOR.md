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
  --requests-per-minute 5 \
  --exploration-top-k 3 \
  --tickers AA AAL BBAI CHWY CLOV ETSY FUBO JOBY LUNR UPST
```

`--dates-csv` remains available for targeted replays. Calendar generation
removes weekends, regular U.S. equity exchange holidays, and known
exchange-wide exceptional closures. Extra closures or sessions can be supplied
to the calendar function when a historical correction is required.

Workers are bounded to 1-8 concurrent trading dates. All provider requests
share one thread-safe adaptive limiter. HTTP 429, transient server responses,
and timeout-class responses use bounded exponential retries and honor
`Retry-After`; throttling slows the shared request pace and successful requests
gradually return it to the configured ceiling.

The Massive Free starting profile is deliberately conservative:

- `--max-workers 2`
- `--requests-per-minute 5`

Raising worker count improves local processing concurrency but does not bypass
the provider-wide request limit.

## Resume and cache guarantees

The run persists an atomic date queue at `replay_queue.json`. Each day also
persists an atomic ticker queue and a completed result checkpoint per ticker.
Interrupted `IN_PROGRESS` tasks are returned to `PENDING` on restart; completed
work is reused.

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

- compounded Scout and benchmark return;
- total realized P&L and money-capture percentage;
- maximum drawdown and daily-return volatility;
- positive-day rate and top-ten capture;
- a partition-labeled daily replay ledger;
- cumulative missed-opportunity hypotheses;
- cross-day shadow-feature evidence; and
- the enforced safety and dataset-isolation boundaries.

All outputs remain research-only. Real-money execution, automatic production
mutation, and automatic promotion are disabled.

