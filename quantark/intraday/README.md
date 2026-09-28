# Intraday pricing (`quantark.intraday`)

Values a contract at a **timezone-aware timestamp** instead of a date: 14:59:59, the
15:00 fixing itself, or thirty seconds after it with the fixing not yet published.
Variance accrues on a desk-declared intraday profile, carry and discounting run on
seconds-exact calendar time, today's fixing is an explicit before/after phase, and
fixings that are due but missing are replaced by the latest spot and **flagged**.
Design: `docs/superpowers/specs/2026-09-15-intraday-pricing-design.md`.

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
- `lifecycle_state`: the authoritative date-based checkpoint (never mutated). Its
  `valuation_point.date` is an INSTANT: a date-only stamp (midnight, as the daily trackers
  write it) is the state after that day's close, and any other wall-clock time covers exactly
  the events at or before it. A checkpoint whose instant is after the valuation — including
  today's close stamped before that close — or whose contents (cashflows, observed events,
  hit dates) were determined after its instant is rejected; so is one AT the valuation
  instant under `before` that has decided an event there.
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
(`model`/`confirmed`/`provisional`; a provisional flow's `depends_on` names the assumed
events that can change whether it exists or what it pays — an assumed KO observation
reaches every later flow, an assumed coupon observation the memory of later coupons, an
assumed knock-in the maturity payoff and, where a knock-in changes the KO rule, every
later flow), the profile and session identities, a
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
diagnostics (PDE resolution, MC standard error, theta truncation estimate). Accuracy is not a
property of a row: it is measured offline (see the last section).

PDE routes report a resolution status with every price: `resolved` needs at least 4 grid
cells across the diffusion layer `sqrt(W)` to the first event, at least 16 time steps across
its variance, and a Crank–Nicolson grid-scale mode damped by e^-8 before the valuation; the
route refines points and steps per day on a clone to reach them. A barrier that enters the
grid by node overwrite (one-touch, discrete barriers) is `under_resolved` unless a node sits on it.
The verdict is a diagnostic reported with the price; it never withholds a number.

PDE spot readout uses a local cubic at the requested log spot. The former three-node
quadratic kept curvature at its nearest node, causing first-order phase oscillations
under refinement.

## Greeks

Every Greek re-evaluates the SAME resolved price function: bumped cells share the numerical
twin, the ledger and every confirmed or assumed fixing, so a bump never re-decides an
observation.

- `greek_convention="desk_bump"` — the daily conventions (`bump_envs`): relative central
  spot bumps, one-sided raw vega per `vol_bump` of the trading-quoted surface, one-sided rho
  and dividend rho rescaled to +1%.
- `greek_convention="point"` — derivatives at the query spot by the route's own method:
  QUAD V2 kernel derivative, closed forms (digital; central difference of the barrier closed
  form), the PDE solver's stencil (its resolution verdict travels as the reason), paired RQMC for MC
  (RANDOMIZED_QUASI engines with an RQMC session spec; others raise `CapabilityError`).
  Where the price function jumps at the query spot (an unfixed event at the valuation
  instant on its level, a continuous barrier hit there) delta and gamma are `undefined`.
  Every route returns the number it computes; the value's diagnostics (`reason`,
  `error_estimate`, `numerical`) travel with it, and its accuracy is measured offline.
- `"theta"` follows the convention. Under `desk_bump` it is the declared forward roll
  (`theta_step`, default one hour; `theta_unit` second/minute/hour/day) on the frozen market
  (`roll_context`), including cash paid during the step. Under `point` it is the time
  DERIVATIVE: a one-sided second-order stencil `(-3 V(0) + 4 V(h) - V(2h)) / 2h` with `h`
  one thousandth of the current segment (never below 1 ms, where price round-off dominates;
  `result.numerical["theta_point_step_s"]`). A roll or stencil that would cross the next
  event, variance-clock or coefficient boundary is clamped to land on it (BEFORE) and
  reported (`theta_adjusted`, `theta_side`); at an event instant under BEFORE there is no
  step inside the segment and theta is `undefined`, never zero.

`roll_through_events(engine, request, to_timestamp, outcomes=...)` is a scenario, not a
derivative: every event crossed needs a `Fixing` outcome, and the contract is valued at
`to_timestamp` on the frozen market after them.

| Greek | `point` unit | `desk_bump` unit |
|---|---|---|
| delta | per unit spot (derivative) | per unit spot, central relative move `spot_bump` |
| gamma | per unit spot² (derivative) | per unit spot², central relative move `gamma_spot_bump` |
| vega | per unit trading-quoted vol (finite-difference proxy, bump disclosed) | PnL per `+vol_bump` of the trading-quoted surface, one-sided |
| rho / dividend rho | per unit rate / yield (finite-difference proxy, bump disclosed) | PnL per +1%, one-sided and rescaled |
| theta | PnL per `theta_unit`, the time derivative (stencil) | PnL per `theta_unit` over the declared forward roll |

Every `GreekValue` carries `status` (`ok`, `undefined`, `failed`) and, when not `ok`, a
`reason` and no value. `example/intraday_greeks_demo.py` prints both conventions
side by side one hour and one second before a fixing, the undefined point Greeks at the
fixing, the zero delta and non-zero rho of an assumed knock-out paid later, and a
roll-through-events row.

## Batches, curves and books

- `value_intraday_many(items, collect_errors=...)` (or `PricingSession.value_intraday_many`)
  keeps caller order; a failed item is an `IntradayFailure`, never a silent `None`.
- `spot_curve(engine, request, spots)` resolves ONE context: its confirmed and assumed
  fixings come from the request's own spot and are shared by every point — a curve never
  re-decides an observation at a curve spot. QUAD V2 prepares its operator once over the
  spots and reads price, delta and gamma from it, each point with its own per-output
  `statuses`; other routes price each spot.
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
  intraday map's session knots. Supporting it needs an interval survival/crossing operator
  split at every clock and coefficient knot, with its own time-refinement and first-passage
  evidence. Route continuous barriers to PDE or MC.
- **A closed-form barrier whose coefficients are not provably piecewise.**
  `quantark.intraday.coefficients` admits only curve families that declare where their law
  changes and its polynomial degree between (flat and term-structure volatility, flat and
  linear/log-linear rates, flat and term-structure dividends, their parallel shifts, and the
  frozen-market roll wrappers a theta reprices). A single affine piece is flat by its law; a
  quadratic cumulative carry — a linearly interpolated zero rate or yield — is sampled at
  every piece's midpoint too, including the first piece from the valuation instant (pillars
  `(0, 1%)` and `(T, 10%)` are not a flat forward). Anything else — a cubic-spline curve, a
  shifted term surface, a smile, a subclass of an admitted family — is inadmissible:
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
  valuation instant; the daily tracker observes only the pre-KI schedule. For the same reason
  its maturity close under `BEFORE` is decided from the float-time twin
  (`KnockOutResetSnowballOption.decide_observations_at_valuation`), not through the
  `BEFORE` / `AFTER` lifecycle identity the snowball and the Phoenix resolve it with.
- **A KO-reset snowball past its pre-KI schedule.** A knock-in replaces the first knock-out
  schedule by the second and stays to the end, so a knocked-in contract is valued on the
  second schedule up to its final maturity. A contract NOT knocked in when the first schedule
  ends matured there: a checkpoint that still calls it alive is refused (`ValidationError`),
  and a knock-in is never tested after that schedule's last observation.
- Dated autocallables without `initial_date` (their accrual would move with the valuation
  timestamp), time-based checkpoints, checkpoints that report the future, and two due
  fixings on one local date (the daily lifecycle tracker cannot separate them) fail closed.

## What a status does and does not claim

- `ok`: a finite result was computed by the declared algorithm and convention. It is not a
  per-request accuracy claim. A route's `numerical["resolution"]` (`resolved` /
  `under_resolved` / `deterministic`), an MC `std_error` and a theta `error_estimate` are
  diagnostics that travel with the number.
- `undefined`: the requested derivative does not exist at this instant and spot (a payoff
  discontinuity of an unfixed event at the query spot; a theta at an event boundary).
- `failed`: the computation could not produce the result (a non-finite output, no admissible
  stencil step, a bump cell whose pricing raised). Too few RQMC batches for a standard error
  is not a failure: the estimate is reported `ok` and the reason says the error is unavailable.

Each requested output carries its own status: a finite delta is reported `ok` beside a gamma
that `failed`.

Accuracy is certified offline. The daily-KI snowball study
`example/modelvalidation/snowball_intraday_daily_ki_bsm.yaml` measures QUAD V2 and PDE on PV,
desk and point spot Greeks and desk theta against a deterministic reference (an
engine-independent Gaussian-transition solver with a declared uncertainty radius, qualified
case by case by paired RQMC, which cannot itself resolve intraday budgets). The banked
certificate is `docs/modelvalidation/certificates/snowball-intraday-daily-ki-bsm/2026-09-19/`:

- `SnowballQuadEngineV2` (order 8, 2 cells per standard deviation): **ADMITTED** on all six quantities, 23 cases.
- `SnowballPDESolver` (`accuracy: standard`): **REJECTED**; its errors and its own refinement envelope exceed the
  intraday budgets in most cells. It stays usable for research; a production release does not ship it for this study.

A production release ships an engine only for the studies that admit its shipped
configuration; nothing in this package reads that decision.
