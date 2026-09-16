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
intraday alternatives; an engine subclass does not inherit a row. The full matrix —
QUAD V2, PDE, MC and closed-form routes for snowballs, phoenixes, KO-reset snowballs,
digitals, barriers and one-touches — is generated into
[`docs/execution/intraday-capability-matrix.md`](../../docs/execution/intraday-capability-matrix.md)
(`python -m quantark.intraday.publish`).

`supported` means the semantics are implemented and every price reports its numerical
status. `qualified` means every Gate C price cell of the row passed against the independent
Gaussian-transition reference (profiles uniform, desk and sessions-only; eleven spot offsets
on both sides of each barrier) at the qualified horizon and every longer horizon
(`quantark/intraday/evidence/gate_c_results.json`):

| Product | Engine | Monitoring | Qualified horizon |
|---|---|---|---|
| SnowballOption | SnowballQuadEngineV2 | discrete | 1 s |
| CashOrNothingDigitalOption | DigitalOptionAnalyticalEngine | terminal | 1 s |
| BarrierOption (zero carry) | BarrierAnalyticalEngine | continuous | 1 s |
| OneTouchOption (zero carry) | OneTouchAnalyticalEngine | continuous | 1 s |

PDE and MC routes miss the 1e-6-of-notional price budget at default settings while their
refinement converges — `supported`, with the per-price status saying so. Timestamp support
below the qualified horizon does not imply a Greek-accuracy certificate there.

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
  Point vega/rho/dividend rho are `unqualified` (no value) until a Gate C bump-limit ladder
  demonstrates them for the (product, route): demonstrated to one second before the fixing
  for QUAD V2 snowballs and analytical digitals (`evidence/gate_c_greeks.json`). MC point
  delta/gamma (a paired RQMC difference at the desk bump) stay `unqualified`.
- `"theta"` under either convention — a declared forward step (`theta_step`, default one
  hour; `theta_unit` second/minute/hour/day) on the frozen market (`roll_context`),
  including cash paid during the step. A step that would cross the next event is clamped to
  land on it (BEFORE) and reported (`result.numerical["theta_adjusted"]`, `theta_side`); at
  an event instant under BEFORE there is no step inside the segment and theta is
  `undefined`, never zero.

`roll_through_events(engine, request, to_timestamp, outcomes=...)` is a scenario, not a
derivative: every event crossed needs a `Fixing` outcome, and the contract is valued at
`to_timestamp` on the frozen market after them.

| Greek | `point` unit | `desk_bump` unit |
|---|---|---|
| delta | per unit spot (derivative) | per unit spot, central relative move `spot_bump` |
| gamma | per unit spot² (derivative) | per unit spot², central relative move `gamma_spot_bump` |
| vega | per unit trading-quoted vol (proxy, `unqualified` until demonstrated) | PnL per `+vol_bump` of the trading-quoted surface, one-sided |
| rho / dividend rho | per unit rate / yield (proxy, `unqualified` until demonstrated) | PnL per +1%, one-sided and rescaled |
| theta | PnL per `theta_unit` over the declared forward step (either convention, same number) | same |

Every `GreekValue` carries `status` (`ok`, `undefined`, `unqualified`, `failed`) and, when
not `ok`, a `reason` and no value. `example/intraday_greeks_demo.py` prints both conventions
side by side one hour and one second before a fixing, the undefined point Greeks at the
fixing, the zero delta and non-zero rho of an assumed knock-out paid later, and a
roll-through-events row.

## Batches, curves and books

- `value_intraday_many(items, collect_errors=...)` (or `PricingSession.value_intraday_many`)
  keeps caller order; a failed item is an `IntradayFailure`, never a silent `None`.
- `spot_curve(engine, request, spots)` resolves ONE context: its confirmed and assumed
  fixings come from the request's own spot and are shared by every point — a curve never
  re-decides an observation at a curve spot. QUAD V2 prepares its operator once over the
  spots and reads price, delta and gamma from it; other routes price each spot.
- `aggregate_intraday([(id, quantity, result), ...])` scales price, paid cash and Greeks by
  quantity; the book is provisional if any position is (and names them), and a Greek is
  summed only when every position reports it `ok` under one convention and unit.
- `spot_curve` returns a `SpotCurve`: a sequence of points that also carries the resolved
  context's provenance once — `provisional`, the assumptions and any
  `continuous_assumption`, the lifecycle, the profile/session/context identities and the
  engine. Each point carries its own price evidence (`numerical`, `method`), because a
  route that prices every spot separately can resolve one and not the next.

## Not covered — these fail closed

Every limit below raises rather than approximating. None of them is a silent fallback.

- **Continuous monitoring on QUAD V2.** Its exact continuous-curve classifier does not
  recognize `TradingClockVolSurface`, and its continuous grid builder does not carry the
  intraday map's session knots. Qualifying it needs an interval survival/crossing operator
  split at every clock and coefficient knot, with its own time-refinement and first-passage
  evidence. Route continuous barriers to PDE or MC.
- **A closed-form barrier whose coefficients are not provably piecewise.**
  `quantark.intraday.coefficients` admits only curve families that declare where their law
  changes and that it is affine between (flat and term-structure volatility, flat and
  linear/log-linear rates, flat and term-structure dividends, plus their parallel shifts).
  Anything else — a cubic-spline curve, a shifted term surface, a smile — is inadmissible:
  matching samples at two instants prove nothing about the interval between them.
- **A digital inside the daily engine's expiry tolerance.** Below `MIN_MATURITY` the daily
  engine switches to the intrinsic payoff, a CALENDAR-TIME shortcut. Intraday that window
  is reachable with variance still on the clock, so the route refuses rather than publish a
  price and Greeks that describe different functions. The `W == 0` limit is exact and is
  taken before this check.
- **Phoenix coupon memory with unequal periods.** Memory reaches the twin as a COUNT, which
  only reconstructs an amount when every period is worth the same (a declared
  `fixed_coupon_year_fraction`, or a per-period rate). Otherwise the twin fails closed.
- **KO-reset snowball replay.** Supply a checkpoint covering every fixing due before the
  valuation instant; the daily tracker observes only the pre-KI schedule.
- Dated autocallables without `initial_date` (their accrual would move with the valuation
  timestamp), time-based checkpoints, checkpoints that report the future, and two due
  fixings on one local date (the daily lifecycle tracker cannot separate them) fail closed.

## What a status does and does not claim

- A route's `numerical["resolution"]` is a RESOLUTION diagnostic — the mesh covered the
  diffusion layer — never an error budget. Greek status is bound separately to the Gate C
  demonstrated families in the capability matrix: product, route, measure, monitoring, the
  exact variance profile, and the horizon window the ladder actually swept. Outside any of
  those a Greek reports `unqualified` with no value, whatever the mesh did.
- A desk bump is exactly its repriced difference, so its only error is the prices' own: a
  bump cell the route could not resolve makes the desk Greek `unqualified` too.
- A local theta is a declared one-sided forward roll on the frozen market, clamped to the
  next event AND the next variance-clock or coefficient boundary. It is a finite roll, not
  a demonstrated `dV/dt`, and every value says so.
