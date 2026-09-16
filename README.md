# AI Trading Lab

Deterministic historical research system for Scout, Trade Engine, Scout
Trainer, and Audit. Phase 1 is historical-only; paper trading is intentionally
out of scope until replay, benchmarking, blind holdouts, and manual promotion
gates pass.

## Current contract

- Scout V1 has exactly 30 metrics scored from 0 through 4.
- The denominator is always 120 points.
- `0` means an observed metric supplied no positive evidence.
- `MISSING` means the metric could not be calculated; it is never silently
  converted to observed zero.
- Hard guardrails override score.
- Every selection input must be timestamped and known by the 07:00
  `America/New_York` historical freeze.
- Trainer cannot mutate or promote Production Scout.

The synthetic January 2, 2018 fixture validates plumbing only. It does not
represent a completed historical replay or a promoted Production Scout.

## Local setup

```bash
python -m pip install -r requirements.txt
python -m pytest -v
```

## Tiingo setup

Copy `.env.example` to `.env`, add the local `TIINGO_API_TOKEN`, and never
commit the token.

```bash
python -m trainer.verify_tiingo --ticker SPY --date 2018-01-02
```

The Tiingo transport lives behind the provider-neutral
`MarketDataProvider` boundary. Provider responses must be converted to
timestamped observations and pass the replay freeze checks before Scout can
see them.

Fetch Tiingo's usable daily baseline and regular-session outcome history into
the checksum-protected local cache with:

```bash
python -m trainer.cache_tiingo --ticker SPY --date 2018-01-02
```

Every cache entry contains an immutable request manifest, provider/feed
identity, retrieval timestamp, record count, and a SHA-256 content digest.
`data/cache/` is intentionally ignored by Git; raw licensed provider data is
not committed to the repository.

Tiingo Free did not return historical 04:00-07:00 premarket rows in the live
capability check. It is therefore used only where the feed is sufficient:
daily baselines and regular-session outcome grading. Synthetic premarket bars
exercise the same provider-neutral contract until a consolidated historical
premarket provider passes its own capability check.

## Implemented replay path

- All twelve Price & Volume Dynamics metrics have deterministic V1 baseline
  calculations and 0-4 score mappings.
- Premarket bars must be timezone-aware, chronological, contiguous one-minute
  OHLCV observations and no later than the 07:00 freeze.
- Missing baseline inputs remain `MISSING`; they are never converted to zero.
- Regular-session paths are isolated from Scout and admitted only to outcome
  grading.
- `run_single_day_replay` connects Scout selection, position-limit enforcement,
  deterministic execution, MFE/MAE, maximum capturable move, and capture ratio.

Tiingo-ready does not mean the January 2 replay is complete. The next external
data gate is a provider check for genuine January 2018 premarket history.
After that passes, work proceeds through universe construction, the
same-universe top-ten benchmark, manual day-one review, and blind holdout
validation.

## Massive Free capability check

Massive is the provider candidate for consolidated U.S. premarket aggregates.
Store its credential as `MASSIVE_API_KEY`; never commit or print the key. The
free Stocks Basic plan is tested first against a recent completed trading day:
See the official
[Massive custom-bars documentation](https://massive.com/docs/rest/stocks/aggregates/custom-bars)
for the upstream aggregate contract.

```bash
python -m trainer.massive_capability_report \
  --ticker SPY \
  --date 2026-09-14
```

The check uses one full-day aggregate request, partitions it in
`America/New_York`, and requires at least 60 one-minute bars from 04:00 up to
but not including 07:00. The 07:00 bar is excluded because its completed value
would contain information occurring after the exact Scout freeze. The report
also checks a daily aggregate and writes only counts, timestamps, plan/feed
identity, and sanitized errors to `reports/massive/capability_report.json`.

Passing on a recent day proves the adapter and premarket feed shape. It does
not make January 2, 2018 available on the two-year free plan; that historical
milestone remains a later paid-data gate.

## Research Scout Alpha

Research Scout Alpha is isolated from Production Scout. It scores only the
twelve implemented Price & Volume Dynamics metrics against a fixed 48-point
denominator. It cannot authorize execution, and historical spread and order
book depth remain explicitly unevaluated on Massive Free.

Run the real-data pipeline probe with:

```bash
python -m trainer.run_massive_alpha --ticker SPY --date 2026-09-14
```

The universe collector supports dated common-stock discovery, point-in-time
market-cap filtering, free-plan request pacing, and an atomic checkpoint after
every ticker. A stopped collection resumes from the checkpoint:

```bash
python -m trainer.run_universe_collection \
  --date 2026-09-14 \
  --discover \
  --max-new 25
```

At five calls per minute, a full universe collection is intentionally slow.
The manifest reports remaining symbols and estimated minutes so a paid-plan
decision can be based on measured workload rather than guesswork.

## Historical Replay Accelerator

Multi-day research can run from an explicit session list or an exchange-aware
date range with bounded concurrent date workers, a shared adaptive provider
rate limit, resumable date/ticker queues, checksum-verified deduplicated cache
entries, and historical news/sentiment shadow backfills.

Chronological development/validation/holdout assignments are frozen per
experiment. Only development evidence can train a hypothesis; validation is
out of sample, and holdout dates require an explicit unlock. The cumulative PDF
reports compounded return, money capture, drawdown, stability, daily results,
cross-day hypotheses, feature evidence, and dataset isolation.

See `docs/HISTORICAL_REPLAY_ACCELERATOR.md` for operation and safety details.

Historical universe construction is a hard evidence boundary. Static symbols
are permitted only in explicit `ci_fixture` mode and can never feed Trainer,
cumulative performance, or promotion. Genuine research creates an immutable
daily point-in-time universe manifest, includes historically tradable
securities that later delisted, uses stable security identities, and carries
the same manifest hash through Scout and its benchmark. Incomplete provider
coverage fails closed. See `docs/UNIVERSE_INTEGRITY.md`.

## Authoritative specifications

- `SCOUT_TRAINER.md`
- `SCOUT_V1_SCORING_MODEL.md`
- `config/feature_registry_v1.json`
