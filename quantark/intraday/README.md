# Intraday pricing (`quantark.intraday`)

Values a contract at a **timezone-aware timestamp** instead of a date: 14:59:59, the
15:00 fixing itself, or thirty seconds after it with the fixing not yet published.
Variance accrues on a desk-declared intraday profile, carry and discounting run on
seconds-exact calendar time, today's fixing is an explicit before/after phase, and
fixings that are due but missing are replaced by the latest spot and **flagged**.
Design: `docs/superpowers/plans/2026-09-15-intraday-pricing-design.md`.

## The three clocks

| Clock | What runs on it | Where it lives |
|---|---|---|
| Calendar | carry, discounting, every engine time: `tau = seconds / (365·86400)` from the valuation instant | `timestamp.calendar_year_fraction` |
| Variance | total variance `w = σ_inner(K, u)²·u` with `u = to_trading(tau)` | `IntradayTimeMap` inside `TradingClockVolSurface` |
| Contractual | coupon accruals and cash amounts, fixed at inception on the contract's own day count | `ContractTimeline` (from the product's resolvers) |

A `VarianceProfile` splits one trading day's budget `1/D` over the close-to-close
window that ends at that day's close: overnight (weekends and holidays fold in), each
session, each break. Weights are explicit, versioned desk inputs summing to one; a zero
weight is an **exactly** zero-variance interval (the map returns the stored knot value,
so total variance differences to `0.0` bitwise). `VarianceProfile.uniform(...)` is a
constant calendar rate; `sessions_only(...)` puts all variance in the sessions. An early
close keeps the day's budget (`same_budget`).

## Inputs

- `pricing_env.valuation_date`: the valuation **timestamp**, timezone-aware (naive is
  rejected); `pricing_env.vol_surface` is the INNER trading-quoted surface (the resolver
  wraps it); `day_count_convention` must be `CALENDAR_DAYS`; `spot_quote.timestamp` is
  required whenever an assumption has to be recorded.
- `TradingSessionCalendar`: holidays (`Calendar`), timezone, sessions, early closes,
  payment time. Date-only observations resolve to the close; a DST gap or fold is an
  error, never a guess.
- `fixings`: `Fixing(timestamp, value)` at contract event instants.
- `lifecycle_state`: the authoritative date-based checkpoint (never mutated; events on or
  before its day are history).
- `event_phase`: `"before"` (an event exactly at the valuation instant is still open and is
  decided at spot by the engine) or `"after"` (it is history).
- `ObservationRecord.observation_timestamp` / `settlement_timestamp` give a record an
  explicit instant; `fixing_time_of_day` overrides the close convention per contract;
  float schedules need `schedule_origin`.

## What `price` means

`price = contingent_pv + pending_receivable_pv` per contract unit: the PV of the claim
that remains. Cash already paid is reported in `paid_cash` and is **not** in `price`;
`result.total_value(include_paid_cash=True)` states the other convention explicitly.
Every result carries `valuation_timestamp`, `phase`, `provisional`, the `assumptions`,
the reconstructed `lifecycle`, itemised `cashflows` with provenance
(`model`/`confirmed`/`provisional`), the profile and session identities, a
`market_snapshot_id` and a `context_identity` covering everything that changes the
economics (instant, phase, profile, calendar, market, contract, fixings, assumptions,
checkpoint).

## Provisional state: the 15:00 walkthrough

A monthly snowball fixes at 15:00 against KO barrier 103; the first five fixings are
confirmed. `example/intraday_snowball_fixing_demo.py` prints:

| Valuation | Phase | Fixing supplied | Spot 103.5 | Provisional |
|---|---|---|---|---|
| 14:59:59 | before | – | KO cash, still contingent | no |
| 15:00:00 | before | – | KO cash, decided at spot | no |
| 15:00:00 | after | 103.5 | 0 contingent; KO cash paid | no |
| 15:00:30 | after | – | 0 contingent; KO cash paid | **yes**: 103.5 assumed |

The last row replays history from the checkpoint with the latest spot standing in for
the missing fixing and records an `AssumedFixing`. When the real fixing arrives (say
102.5), the same call with it supplied reverses the assumed knock-out and restores the
live claim — nothing provisional was ever persisted. A continuously monitored KI
discloses a `ContinuousHistoryAssumption` (no unreported touch; the latest spot tests the
barrier with the contract's inclusive rule).

## Usage

```python
from datetime import datetime, time, timedelta, timezone
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.execution import PricingSession
from quantark.intraday import Fixing, TradingSession, TradingSessionCalendar, VarianceProfile
from quantark.util.calendar import CalendarType, create_calendar

shanghai = timezone(timedelta(hours=8))
sse = TradingSessionCalendar(
    name="SSE", tz=shanghai, calendar=create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028)),
    sessions=(TradingSession(time(9, 30), time(11, 30)), TradingSession(time(13, 0), time(15, 0))))
desk = VarianceProfile("desk", "1", days_per_year=244, overnight_weight=0.25,
                       session_weights=(0.35, 0.35), break_weights=(0.05,))

# snowball: a dated SnowballOption; env: PricingEnvironment with an aware valuation_date and spot timestamp
with PricingSession() as session:
    result = session.value_intraday(SnowballQuadEngineV2(), snowball, env,
                                    session_calendar=sse, variance_profile=desk,
                                    fixings=confirmed_fixings, event_phase="after")
print(result.price, result.provisional, [a.assumed_value for a in result.assumptions], result.lifecycle)
```

With a `PricingSession`, an engine-invoking route is re-dispatched through the
execution kernel and must agree with the direct route to 1e-12 (the manifest
fingerprint is appended to `result.records`).

## Capability matrix

Requests outside the matrix raise `CapabilityError` naming the limitation and the
intraday alternatives; an engine subclass does not inherit a row.

| Product | Engine | Monitoring | Profiles | Outputs | Status | Note |
|---|---|---|---|---|---|---|
| CashOrNothingDigitalOption | DigitalOptionAnalyticalEngine | terminal | any | price | supported | integrated carry/variance via TradingClockVolSurface; zero variance priced as the exact forward limit |
| SnowballOption | SnowballQuadEngineV2 | discrete | any | price | supported | exact Gaussian interval moments; a t=0 event under 'before' is decided at spot |
| SnowballOption | SnowballQuadEngineV2 | continuous | any | price | supported | continuous KI via the QUAD V2 survival kernel; touch history disclosed as an assumption |
| PhoenixOption | PhoenixQuadEngineV2 | discrete | any | price | supported | valuations before the first due coupon (coupon replay fails closed) |
| PhoenixOption | PhoenixQuadEngineV2 | continuous | any | price | supported | valuations before the first due coupon (coupon replay fails closed) |

`supported` means the semantics are implemented; `qualified` (accuracy evidence against an
independent reference on the time-to-fixing ladder) arrives with Gate C.

PDE routes report a resolution status with every price: `resolved` needs at least 4 grid
cells across the diffusion layer `sqrt(W)` to the first event, at least 16 time steps across
its variance, and a Crank–Nicolson grid-scale mode damped by e^-8 before the valuation; the
route refines points and steps per day on a clone to reach them. A barrier that enters the
grid by node overwrite (one-touch, discrete barriers) is `unqualified` unless a node sits on it.

## Greeks

Every Greek re-evaluates the SAME resolved price function: bumped cells share the numerical
twin, the ledger and every confirmed or assumed fixing, so a bump never re-decides an
observation.

- `greek_convention="desk_bump"` — the daily conventions (`bump_envs`): relative central
  spot bumps, one-sided raw vega per `vol_bump` of the trading-quoted surface, one-sided rho
  and dividend rho rescaled to +1%.
- `greek_convention="point"` — derivatives at the query spot from the route's own evidence:
  QUAD V2 kernel derivative, closed forms (digital; central difference of the barrier closed
  form), the PDE solver's stencil (`ok` only on a `resolved` grid), paired RQMC for MC
  (RANDOMIZED_QUASI engines with an RQMC session spec; others raise `CapabilityError`).
  Where the price function jumps at the query spot (an unfixed event at the valuation
  instant on its level, a continuous barrier hit there) delta and gamma are `undefined`.
  Point vega/rho/dividend rho are `unqualified` (no value) until a bump-limit ladder
  demonstrates them.

## Not yet covered

- Intraday theta, the roll-through-events scenario, batch valuation, spot curves and
  aggregation; qualification horizons from Gate C in the capability matrix.
- Phoenix coupon replay: the lifecycle tracker books a coupon as `principal·rate` while the
  engines pay `principal·rate·period fraction`; until that is settled a due Phoenix coupon
  not covered by the checkpoint fails closed.
- Dated autocallables without `initial_date` (their accrual would move with the valuation
  timestamp), time-based checkpoints, and two due fixings on one local date (the daily
  lifecycle tracker cannot separate them) fail closed.
