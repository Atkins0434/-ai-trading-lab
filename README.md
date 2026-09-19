# AI Trading Lab

Deterministic historical research system for Scout, Trade Engine, Scout
Trainer, and Audit. Phase 1 is historical-only; paper trading is intentionally
out of scope until replay, benchmarking, blind holdouts, and manual promotion
gates pass.

## Current contract

Replay manifests include a top-level `provenance` object recording the Scout ID,
rubric version, feature registry ID/count, backlog dataset version, comparison
policy paths, execution-cost model ID, and checkout git SHA. Before resuming,
the engine and scheduled artifact restore compare every field against current
configuration. Missing or mismatched provenance archives the entire output root
as `<output_root>.stale-<UTC timestamp>/` and starts fresh; the log lists every
differing field. Any git SHA change also invalidates resume, even if unrelated
to scoring. Day/range output-cache keys include Scout ID, metric count, dataset
version, and a fingerprint of all provenance fields; data/reference cache keys
are unchanged. `--force-resume` is for debugging only: it records
`forced_resume=true` and the previous provenance. Such outputs require the debug
override again on subsequent resumes. No workflow sets this flag.

Historical execution reports gross fills and post-fill net results using
`config/execution_costs.json` (`execution_costs_v1.0`). Slippage uses each side's
fill-price tier; exits before 09:35 New York time use the opening tier, later
exits use the later tier, and `SESSION_END` liquidations use the session-end tier
(including early-close sessions). Commission minima apply independently to both
orders; regulatory fees apply only to shares sold. Costs never alter sizing,
stops, targets, or fill timestamps/prices. Net capture uses day MFE from the open.
`config/benchmark.json` keeps `verdict_basis` at `GROSS`; the parallel net verdict
is diagnostic. Changing the primary basis to `NET` requires a later reviewed PR.
Archived results without costs display unavailable net values; cumulative tables
identify costed days, and additive CSV cost columns remain empty for older days.
The previously mislabeled gross `net_realized_pnl_usd` in policy summaries now
holds net P&L; its old value is retained as `realized_pnl_usd` and
`gross_realized_pnl_usd`. No backlog dataset-version bump is required.

- Scout V1 has exactly 30 metrics scored from 0 through 4.
- The denominator is always 120 points.
- `0` means an observed metric supplied no positive evidence.
- `MISSING` means the metric could not be calculated; it is never silently
  converted to observed zero.
- Hard guardrails override score.
- Every selection input must be timestamped and known before the configured
  `America/New_York` historical freeze (09:15 by default).
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

Tiingo Free did not return historical premarket rows in the live
capability check. It is therefore used only where the feed is sufficient:
daily baselines and regular-session outcome grading. Synthetic premarket bars
remain limited to plumbing fixtures; research replay requires real Massive
trade-minute bars.

## Implemented replay path

- All twelve Price & Volume Dynamics metrics have deterministic V1 baseline
  calculations and 0-4 score mappings.
- Premarket bars must be real, timezone-aware, minute-aligned, strictly
  increasing OHLCV observations from 04:00 up to but not including the
  configured freeze. Sparse intervals are retained and never padded.
- Missing baseline inputs remain `MISSING`; they are never converted to zero.
- Regular-session paths are isolated from Scout and admitted only to outcome
  grading.
- `run_single_day_replay` connects Scout selection, position-limit enforcement,
  deterministic execution, MFE/MAE, maximum capturable move, and capture ratio.

Tiingo-ready does not mean the January 2 replay is complete. The next external
data gate is a provider check for genuine January 2018 premarket history.
After that passes, work proceeds through universe construction, deterministic
same-universe return baselines, top-ten missed-opportunity diagnostics, manual
day-one review, and blind holdout validation. WIN/TIE/MISS compares Scout's
qualifying-threshold gross return with the fixed-seed 200-draw baseline;
top-ten capture is diagnostic only. `EXPLORATION_TOP_K` names are simulated
only after every qualifying name receives execution priority. Their return,
P&L, and win rate are reported in a separate exploration summary and never
change Scout's verdict or qualifying performance.

## Massive Developer capability check

Massive is the provider candidate for consolidated U.S. premarket aggregates.
Store its credential as `MASSIVE_API_KEY`; never commit or print the key. The
active Stocks Developer capabilities are declared in
`config/massive_plan.json` and tested against a recent completed trading day:
See the official
[Massive custom-bars documentation](https://massive.com/docs/rest/stocks/aggregates/custom-bars)
for the upstream aggregate contract.

```bash
python -m trainer.massive_capability_report \
  --ticker SPY \
  --date 2026-09-14
```

The check uses one full-day aggregate request, partitions it in
`America/New_York`, and requires the configured minimum number of real bars in
the final 60 minutes before the morning freeze. A bar starting exactly at the
freeze is excluded because its completed value contains post-freeze
information. The report also checks a daily aggregate and writes only counts,
timestamps, plan/feed identity, and sanitized errors to
`reports/massive/capability_report.json`.

Passing on a recent day proves the adapter and premarket feed shape. The
Developer plan records ten years of history and flat-file access. Point-in-time
universe proof remains a separate provider-capability gate.

## Massive flat-file historical replay

Bulk replay uses Massive's S3-compatible flat files instead of requesting
minute aggregates one ticker at a time. Configure the REST credential used for
date-scoped reference data separately from the S3 credentials used for files:

```text
MASSIVE_API_KEY=...
MASSIVE_S3_ACCESS_KEY_ID=...
MASSIVE_S3_SECRET_ACCESS_KEY=...
```

The loader lists each dataset's year/month prefix and resolves the dated file
from the returned objects. It never assumes a fixed daily object filename.
Verified files are cached below `data/flatfiles/`, excluded from Git, and reused
when their S3 size and local checksum still match.

Run a resumable replay:

```bash
python -m trainer.flatfile_replay \
  --start 2018-01-02 \
  --end 2018-01-31
```

Report local coverage without downloading:

```bash
python -m trainer.flatfile_coverage \
  --start 2018-01-02 \
  --end 2018-01-31
```

The point-in-time universe starts with Massive's date-scoped reference endpoint.
Market capitalization is recomputed from a 120-calendar-day lagged shares-
outstanding query and the prior session close in `day_aggs_v1`; no present-day
active list or current market capitalization is substituted. The manifest
labels this policy `LAGGED_PROXY` and records the requested lagged date, actual
provider query date, and provider period date. Ticker Overview responses are
reused from `data/reference_cache/` within a calendar quarter only when the
cached date is no later than the requested lagged date. The flat-file replay
manifest records reference-cache hits/fetches alongside dataset keys, byte
sizes, SHA-256 checksums, universe sizes, real-bar counts, and the share of
eligible tickers that are not scorable. The day fails only when no ticker is
scorable.

## Research Scout Alpha

Research Scout Alpha is isolated from Production Scout. It scores only the
twelve implemented Price & Volume Dynamics metrics against a fixed 48-point
denominator. It cannot authorize execution, and historical spread and order
book depth remain explicitly unevaluated when historical quote/depth inputs are
unavailable. Missing quotes or
depth produce `NOT_EVALUATED` guardrails with explicit reason codes and do not
block selection. Each Scout config can restore fail-closed behavior by setting
the corresponding missing-data policy to `REJECT`; this changes the guardrail
action without concealing that the underlying market data was unavailable.

Run the real-data pipeline probe with:

```bash
python -m trainer.run_massive_alpha --ticker SPY --date 2026-09-14
```

The universe collector supports dated common-stock discovery, point-in-time
market-cap filtering, plan-aware request pacing, and an atomic checkpoint after
every ticker. A stopped collection resumes from the checkpoint:

```bash
python -m trainer.run_universe_collection \
  --date 2026-09-14 \
  --discover \
  --max-new 25
```

Developer runs apply no proactive REST pacing. If
`rest_calls_per_minute` is set to a positive number, the existing finite-tier
spacing is restored. Provider throttles and transient failures still use
bounded retry/backoff in either mode.

## Historical Replay Accelerator

Multi-day research can run from an explicit session list or an exchange-aware
date range with bounded concurrent date workers, plan-aware provider pacing,
resumable date/ticker queues, checksum-verified deduplicated cache
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
# Unattended replay backlog

`Flat-File Replay Scheduled` runs on main every six hours. It shares the day/range
workflow concurrency group, processes one half-month unit per job, and pauses
between days after 16,200 seconds. The manual day and range workflows are unchanged.
Scheduling becomes active when this workflow is merged to main; it uses the same
Massive secrets as the range workflow.

The seed is `config/replay_backlog.json`; live state is
`results/replay_backlog.json` on the isolated `replay-results` branch. Never merge
that branch into main. Add units with unique IDs, inclusive start/end calendar
dates, and `PENDING` status to the seed (new units are copied into live state), or
directly to the results branch. Non-session endpoints are allowed: the unchanged
engine enumerates US trading sessions inside the range.

- Bump `dataset_version` in the seed after a rubric/policy revision. On the next
  scheduled run, older unit results become PENDING and rerun from scratch. Keep
  versions monotonically increasing. Merely changing a code SHA does not requeue
  completed units.
- PAUSED units are preferred over PENDING units. COMPLETE and FAILED units are
  skipped automatically. A FAILED unit blocks nothing: the scheduler moves to the
  next PENDING unit. `IN_PROGRESS` is persisted before compute; if a runner is
  forcibly terminated, an operator must explicitly reset that unit.
- Reset a failed/interrupted unit by checking out `replay-results`, setting its
  status to PENDING in `results/replay_backlog.json`, and committing/pushing that
  change. Or dispatch the scheduled workflow from main with its `unit_id` to
  explicitly retry it. With code available, the equivalent state command is
  `python -m trainer.replay_backlog --path results/replay_backlog.json mark ID PENDING`.
- Inspect state with `python -m trainer.replay_backlog --path PATH status`; `next`
  emits one JSON unit or `null`, and `mark ID STATUS --fields '{"last_error":null}'`
  round-trips bookkeeping fields.

PAUSED recovery downloads the exact full artifact from `last_run_id` with
`actions/download-artifact@v4`. It must contain a matching manifest in
`PAUSED_WALL_BUDGET` state. Missing, expired, or invalid recovery logs a warning,
resets completed-day progress, and starts the unit cleanly. Full artifacts have
90-day retention (subject to repository retention limits); they are not cached.
Only the scheduled workflow grants `actions: read` alongside `contents: write`.

The results branch contains the backlog, `INDEX.md`, and an explicit compact
allowlist: run manifest/summary, cumulative CSVs, an optional run replay PDF, and
completed-day postmortem/benchmark JSON, CSVs, and replay PDF. Daily files retain
their dated names from #32. Snapshots, universe/scoring JSON, full outcome paths,
and stdout/stderr are uploaded only as run artifacts, never committed. The index
reports WIN/MISS/TIE, selection-cohort primary/ATR return sums, reachability, code
SHA, and run links. The optional root replay PDF is copied only if it exists;
the current engine produces daily replay PDFs and a run summary PDF.

Before saving the flat-file cache, only recognized data files and sidecars outside
the unit's range are deleted. Lookback data may be downloaded again next time.
Actions cache has a 10 GB repository budget and each run saves a new entry; pruning
bounds each entry to one unit, not the total retained cache usage. Existing push
workflows explicitly exclude `replay-results`; workflows without push triggers
cannot be activated by results commits.
