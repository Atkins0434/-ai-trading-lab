# Point-in-Time Catalyst Shadow Layer

This research layer supplies Production Scout metrics 19-21 as shadow evidence:

1. Catalyst quality
2. Catalyst verification confidence
3. Catalyst freshness and relevance

It does not change Research Scout Alpha's 48-point score, Production Scout's
120-point contract, candidate qualification, or execution authority.

## Historical admission rule

An event is visible at the 07:00 `America/New_York` freeze only when both are
known and no later than the freeze:

- the publisher's `published_timestamp`; and
- the data provider's `provider_available_timestamp`.

Retrieving an old event after the replay date is allowed. Retrieval time is
recorded as `ingested_timestamp` but does not prove when the event originally
entered the provider feed. If historical provider availability is unknown, the
event is excluded rather than treating publication time as a substitute.

## Duplicate handling

Syndicated copies are retained for audit but count once. An upstream
`duplicate_group_id` is preferred. Otherwise, a deterministic key is derived
from ticker, event type, and normalized headline. The strongest verified
primary-source record becomes the admitted representative.

## Missing-data behavior

No admitted, positive, sufficiently relevant catalyst produces `MISSING` for
all three shadow metrics. It does not produce three observed zeroes. This keeps
missing evidence distinguishable from evidence that was observed and weak.

## Safety boundaries

- All metric results are labeled `SHADOW_ONLY`.
- `included_in_alpha_score` is always false.
- `production_score_changed` is always false.
- Offering or dilution evidence is surfaced as a guardrail candidate, but this
  layer cannot enforce a rejection.
- Promotion still requires 30 independent occurrences, out-of-sample
  improvement, blind holdout validation, and explicit approval.

## Provider adapter requirements

A news/catalyst adapter must return stable source references, timezone-aware
publication and provider-availability timestamps, ticker mappings, source
identity, and feed version. It must not infer provider availability from the
publisher timestamp. Raw licensed responses remain outside Git; normalized
audit records may be cached under the existing checksum-protected cache policy.
