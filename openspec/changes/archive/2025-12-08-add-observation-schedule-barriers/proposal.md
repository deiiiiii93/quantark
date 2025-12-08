# Change: Add ObservationSchedule for barrier-style options

## Why
Discrete barrier monitoring currently assumes a single barrier/rebate across all observation dates. We need date-specific barrier levels and payoff rules (including return-rate style payoffs) plus explicit aggregation modes across hits to cover structured barrier payoffs.

## What Changes
- Introduce `ObservationRecord` (observation time, barrier data, payoff terms) and `ObservationSchedule` (ordered list + aggregation mode: stop-first-hit, accumulate, best, worst).
- Extend existing barrier-like products (single barrier, double barrier, one-touch, double one-touch) to accept an optional `ObservationSchedule` instead of adding a separate discrete-only product.
- Define default/backward-compatible mapping so legacy `observation_type`/`observation_dates` continue to work unchanged when no schedule is provided.

## Impact
- Affected specs: `equity-barrier-products`
- Affected code: `asset/equity/product/option/{barrier_option.py,double_barrier_option.py,one_touch_option.py,double_one_touch_option.py}`, engines that price these products (analytical/PDE/MC) and their validation paths.

