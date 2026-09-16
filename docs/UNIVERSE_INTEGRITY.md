# Point-in-Time Universe Integrity

PR #13 has two explicitly enforced universe modes:

| Mode | Symbols | Permitted use | Research evidence | Promotion input |
|---|---|---|---:|---:|
| `ci_fixture` | Explicit static list | Deterministic engineering smoke tests | No | No |
| `historical_research` | Daily point-in-time resolver | Replay, benchmark, Trainer, cumulative research | Only when coverage is complete | Only when evidence gates pass |

The static list `AA AAL BBAI CHWY CLOV ETSY FUBO JOBY LUNR UPST` is useful
for repeatable transport and scoring checks, but it is selected with present-day
knowledge. Treating it as a historical universe would introduce hindsight and
survivorship bias. Labels such as “research-only” do not make that population
valid research evidence.

## Hardcoded-symbol audit

| Entry point | Previous downstream path | Classification after this change |
|---|---|---|
| `.github/workflows/research-alpha-batch.yml` | CLI → overview collector → ticker queue → Scout → outcome → benchmark → postmortem | PR smoke only; `ci_fixture`; performance/Trainer artifacts suppressed |
| `.github/workflows/multi-day-trainer.yml` | CLI → daily batches → benchmark/postmortem → feature evidence → cumulative PDF | PR smoke is quarantined; schedule/dispatch defaults to `historical_research` |
| `.github/workflows/massive-universe-smoke.yml` (`AAPL MSFT NVDA`) | dated overview collector artifact | Engineering fixture with false evidence and promotion flags |
| Former `trainer/run_research_alpha_batch.py::DEFAULT_TICKERS` | default CLI population | Constant removed; symbols accepted only with explicit `--universe-mode ci_fixture` |
| `trainer/run_multi_day_trainer.py` | inherited Alpha default population | No longer a default; research mode rejects `--tickers` |
| `trainer/universe_collector.py` | current/date-scoped overview filtering | Explicitly fixture-only; cannot feed Trainer or promotion |
| tests and synthetic fixtures | contract, queue, scoring, and edge-case validation | Engineering fixtures unless a synthetic point-in-time provider proves complete coverage |

The complete traced path matters: prior static symbols reached Scout, the
same-universe benchmark, postmortem, feature evidence, and cumulative returns.
Those earlier artifacts are therefore quarantined, not merely relabeled at the
workflow entry point. `reports/LEGACY_RESEARCH_QUARANTINE.json` records the
repository and GitHub Actions artifact groups. The quarantine command can add a
non-destructive registry for downloaded legacy results:

```bash
python -m trainer.quarantine_legacy_results reports
```

## Daily construction rules

`us_listed_common_equity` version `1.0` is evaluated independently for every
trading date at 07:00 `America/New_York`.

- Include common equities listed on XNAS, XNYS, or XASE.
- Exclude ETFs, ETNs, preferred shares, warrants, rights, units, funds, OTC
  securities, test issues, and non-operating placeholders with deterministic
  reason codes.
- Enforce the configured $300 million–$15 billion point-in-time market-cap
  range only when the value and its provider-availability time are proven.
- Use historical listing/trading status, never today’s `active` flag.
- Preserve stable identities across ticker changes and distinguish later reuse
  of the same ticker text by a different security.
- Treat `delisting_effective_date` as the first ineligible session. When a
  provider supplies `last_trading_date`, that date remains inclusive.
- Do not reveal a later delisting or ticker event in an earlier manifest unless
  the provider proves it was known by the cutoff.
- Keep unavailable fields missing. Missing never means observed zero.

## Immutable manifest and resume behavior

The resolver writes `daily_universe_manifest.json` before ticker jobs are
created. Its canonical SHA-256 excludes only the retrieval timestamp and the
hash field itself; securities are sorted by stable identity and historical
ticker. The manifest records the replay/date/cutoff, versioned rules, provider
identity and query, coverage, every inclusion/exclusion decision, counts, and
evidence eligibility.

Resume verifies the stored hash, replay identity, date, mode, ruleset, provider,
feed version, and (for fixtures) exact symbol list. It reuses the manifest and
does not query a newer provider state. A mismatch is a hard error.

## Enforcement boundaries

The manifest’s hash and evidence metadata are copied into the historical
snapshot, Scout output, outcome, benchmark, postmortem, daily batch manifest,
and cumulative state. Benchmark construction rejects unequal hashes. Trainer
accepts only `historical_research` plus `complete` coverage and an affirmative
evidence flag. Promotion independently revalidates every input and rejects a
false promotion flag. Cumulative reporting aggregates only eligible days and
lists all other days under `quarantined_days`; it cannot mix fixture and
research performance.

## Provider limitation and failure behavior

Massive’s All Tickers API documents a historical `date` query, date-scoped
`active` status, stable FIGIs, venue/type, and delisting fields. That is useful
but is not enough for this ruleset. The current adapter cannot prove historical
market-cap availability at the 07:00 cutoff, and complete ticker-identity
interval reconstruction relies on an experimental ticker-events endpoint.
Ticker Overview also does not provide a safe filing-availability timestamp for
derived fundamentals.

Consequently the Massive adapter declares those capabilities unavailable. A
`historical_research` run writes an `incomplete` manifest and an `UNSUPPORTED`
batch without creating Scout performance, benchmark, Trainer, or promotion
evidence. A provider error behaves the same way. There is no fallback to the
current active list or the static CI list.

Provider plan depth is a separate constraint: the configured Massive Developer
plan exposes ten years of history and flat files. Those capabilities do not by
themselves resolve the availability-time and ticker-history proof gaps above.
Every universe and replay manifest records the active plan so evidence cannot
be confused with an earlier free-plan run.
