# Design: KO-Reset Snowball Option (MC)

## Product Shape
Introduce a new product class `KnockOutResetSnowballOption` (equity/autocallable) that:
- Reuses Snowball-style payoff configs (rebate, participation, protection, accrual, airbag).
- Defines **two KO schedules**:
  - `pre_ko_schedule`: applied before KI is triggered.
  - `post_ko_schedule`: applied after KI is triggered.
- Defines a **post-KI schedule mode**:
  - `ABSOLUTE`: post-KI KO schedule uses fixed calendar times/dates; only observations after KI apply.
  - `REBASED`: post-KI KO schedule uses offsets from the KI event time.
- Keeps KI configuration consistent with Snowball (barrier + observation schedule).

## Schedule Resolution
- Both KO schedules are `ObservationSchedule` instances (records may include barrier and return_rate).
- KO payoff resolution follows Snowball logic (annualized accrual + principal inclusion).
- The product exposes:
  - `resolve_pre_ko_observations(pricing_env)`
  - `resolve_post_ko_observations(pricing_env)`
  - `get_ko_reset_profile(pricing_env)` returning both schedules + mode.

## MC Engine Strategy
### Absolute Mode
- Build a time grid that includes:
  - Pre-KI KO observation times
  - Post-KI KO observation times
  - KI observation times (or daily grid if continuous)
  - Final maturity = max(pre/post KO end)
- For each path:
  - Identify the first KI time (if any).
  - Evaluate KO using pre schedule before KI; after KI, switch to post schedule using only post observations after KI time.

### Rebasing Mode
- Post-KI KO schedule stores **offset times** (e.g., monthly fractions).
- MC builds a master grid that includes:
  - Pre-KI KO observation times
  - KI observation times
  - For each **possible** KI observation time, add (KI time + post offsets)
  - Final maturity = max(pre KO end, max(KI time + post offsets))
- For each path:
  - Identify the KI time index (first KI hit).
  - Select KO observation indices corresponding to (KI time + post offsets) and evaluate KO on those times only.
- Constraint: rebased mode requires **discrete KI monitoring** (continuous KI would explode the grid).

## Event Stats
- Extend MC event stats to attribute KO probabilities across pre/post schedules.
- Report KO probability for pre vs post and overall, along with V0/V1 probabilities.

## Backward Compatibility
- Existing `SnowballOption` behavior unchanged.
- New product is separate and explicitly priced by MC engine.
