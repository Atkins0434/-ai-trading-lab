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

Tiingo-ready does not mean the January 2 replay is complete. The next
integration unit is the authenticated historical fetch/cache pipeline,
followed by universe construction, feature calculation, the same-universe
top-ten benchmark, outcome grading, and blind holdout validation.

## Authoritative specifications

- `SCOUT_TRAINER.md`
- `SCOUT_V1_SCORING_MODEL.md`
- `config/feature_registry_v1.json`
