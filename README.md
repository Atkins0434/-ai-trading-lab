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

## Authoritative specifications

- `SCOUT_TRAINER.md`
- `SCOUT_V1_SCORING_MODEL.md`
- `config/feature_registry_v1.json`
