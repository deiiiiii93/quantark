# Change: Update numerical theta handling for observation schedules

## Why
Observation schedules for barrier-like products now carry per-date barrier checks, but numerical theta bumps currently advance time without accounting for past observation entries. This can double-count or mis-handle barrier checks when time is bumped.

## What Changes
- Define numerical theta bump semantics for products with observation schedules: advance valuation by the bump, treat observation records before the bumped valuation as already observed, and do not shift future observation entries.
- Clarify handling for both observation_date and observation_time schedules so resolution reuses original entries while excluding past records.
- Preserve legacy behavior for products without schedules or with continuous/expiry monitoring.

## Impact
- Affected specs: equity-greeks (new)
- Affected code: asset/equity/riskmeasures/greeks_calculator.py, barrier-like pricing paths that consume ObservationSchedule

