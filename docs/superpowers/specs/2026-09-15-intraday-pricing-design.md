# Intraday pricing and observation-time Greeks

Status: **Approved design, 2026-09-15.** Scope, time/clock semantics, engine
integration, request/result flow, provisional behavior, Greek conventions,
numerical release gates, and delivery stages have been reviewed and accepted.
Implementation and numerical qualification remain future work. Concrete
acceptance budgets are established in the baseline/reference stage before
release candidates are tuned; no hard latency SLA has been set.

## Confirmed decisions

1. Support exchange-close observations by default, with explicit fixing times
   per contract. General monitoring windows and multiple-fixing product features
   are not an automatic addition to the first release.
2. Allocate variance through an explicit desk-selected profile covering trading
   sessions, lunch, and overnight. Discounting and carry continue in calendar
   time.
3. Cover BSM across applicable analytical, PDE, QUAD, and MC engines for
   digitals, barriers, and autocallables. Local volatility, Heston, and SLV are
   subsequent model extensions.
4. When a fixing time has passed and neither its fixing nor an authoritative
   resulting lifecycle state is available, return a **provisional scenario using
   the latest spot as the assumed fixing**, clearly flagged until the actual
   fixing arrives.
5. Greek bumps keep provisional assumed fixings fixed. Report sensitivities
   conditional on that provisional outcome; do not reconstruct assumed
   fixings from bumped market spots.
6. The time-and-clock contract is approved: timezone-aware intraday timestamps,
   calendar time as the common numerical axis, separate contractual accrual,
   explicit profile normalization, exact zero-variance intervals, timestamp
   ordering independent of numerical tolerances, and unchanged legacy daily
   interpretation outside explicit intraday mode.
7. The structured request/result design is approved: resolve history once,
   carry confirmed/provisional status and assumptions through price and risk,
   preserve explicit PV/quantity units, and reconstruct provisional outcomes
   when actual observations arrive.
8. For continuous barriers with incomplete touch history, the provisional
   scenario assumes no unreported earlier touch and checks the latest spot
   at valuation time. Disclose the uncovered history interval and any assumed
   hit time; replace these assumptions when authoritative history arrives.
9. Resolve a common valuation context before engine dispatch. Qualify
   analytical barrier/touch methods only where their first-passage and payment
   assumptions hold; explicit unsupported engine requests return capability
   errors, and any explicitly enabled routing reports its selected method.
10. Intraday Greek requests explicitly select point or desk finite-bump
    sensitivities. Preserve existing daily desk bump conventions; record units,
    actual bump sizes, and per-measure numerical status.
11. Delivery is governed by the semantic, provisional-state, numerical, and
    performance-evidence gates below. Establish independent references first,
    retain daily regression baselines, and write a separate implementation plan
    before production edits.

An assumed KO therefore has zero equity delta conditional on its fixing,
while the result remains provisional and can change when the actual fixing
arrives. This does not measure uncertainty about the missing fixing.

## Purpose and observable behavior

A trader should be able to request price and Greeks at distinct timestamps on
an observation date, including seconds before the fixing and after its result.
The request must identify which observations remain uncertain and which
outcomes have been applied. Timestamp precision does not itself certify Greek
accuracy at arbitrarily short horizons.

For a contract fixing at 15:00:

- At 14:59:59, the fixing remains a future event with its remaining integrated
  variance.
- At 15:00, an explicit before/after phase identifies the requested valuation.
- After fixing, confirmed or provisional outcomes select the remaining
  contract state. Fixed unpaid cashflows retain their settlement exposure.
- An assumed KO can disappear when the actual fixing replaces it. Confirmed
  history is never irreversibly changed by a provisional scenario.

## Architecture alternatives

| Approach | Consequence | Decision |
| --- | --- | --- |
| Shared timestamp, clock, and event resolution feeding existing engines | One timing contract across price, Greeks, lifecycle, and settlement | Selected |
| Fractional-maturity adapter around existing calls | Small initial change; daily lifecycle and bump rules still need separate fixes | Insufficient for the requested observation-time behavior |
| Full streaming event-processing framework | Adds data ingestion, scheduling, and operational lifecycle processing | Beyond the initial pricing-library scope |

The flow is: contract and market snapshot -> timestamp/clock
resolution -> confirmed history plus provisional assumptions -> resolved
valuation context -> engine and Greek evaluation -> structured result.

### Timestamp and clock contract

Use timezone-aware timestamps at the public intraday boundary. Resolve exchange
close, timezone, holidays, and shortened sessions through an explicit market
calendar/session definition. Contract fixing-time overrides take precedence.
Distinguish date-only inputs from explicitly timed inputs, including a real
midnight fixing; do not infer that distinction from a zero hour field.

Use calendar time as the common numerical axis in intraday mode.
Keep three concepts separate:

- Elapsed calendar time for discounting and carry.
- A cumulative variance clock derived from the selected profile.
- Contractual coupon accrual and payment conventions.

For an interval from `a` to `b`, the BSM transition uses integrated carry
`R(a,b) - Q(a,b)` and integrated variance `W(a,b)`:

```text
log(S_b / S_a) = R(a,b) - Q(a,b) - W(a,b)/2 + sqrt(W(a,b)) * Z
```

The profile declares its normalization intervals and annualization convention.
It preserves the variance budget of those reference intervals, is additive
across subintervals, and never renormalizes the remaining part of a day to a
full day's variance. Weekend/holiday allocation, early closes, and ownership
of overnight intervals must be specified by the profile.

Zero-weight intervals have exactly zero stochastic variance; calendar carry
can still move spot or change a cashflow PV. Deterministic-volatility term
structures require validated cumulative variance differences. A frozen-market
time roll must preserve the absolute future variance/carry schedule; a newly
supplied market snapshot is a separate market-data update.

Contractual event ordering uses timestamps and event identity. Numerical float
tolerances must not merge distinct fixings or turn a future fixing into a
valuation-time event. Events at the same timestamp retain contractual priority.

### Confirmed and provisional state

Begin each request from an authoritative lifecycle checkpoint. Merge supplied
fixings with the missing-fixing assumptions needed to reach the requested
valuation point, then process the merged event sequence in chronological
order. Do not apply a later supplied fixing before an earlier assumed one.
For each missing discrete fixing, record the scheduled timestamp, assumed
value, source spot timestamp, and affected event. A single latest spot used
for several missed fixings must be visible as several explicit assumptions.

Do not persist provisional KO/KI/coupon transitions into confirmed state or
confirmed cash ledgers. Rebuild the scenario when an actual fixing arrives;
this may change all downstream assumed outcomes and cashflows. Numerical
pricing itself must not mutate the input contract or confirmed history.

An intraday result must carry, alongside price and Greeks:

- Valuation timestamp and observation phase.
- Confirmed/provisional status and assumed-fixing details.
- Resolved lifecycle state and provisional cashflow provenance.
- Variance-profile identity, market snapshot identity, and Greek convention.
- Numerical method and relevant error/convergence information.

Provisional status and assumptions are economic inputs/results, not merely
operational diagnostics. Include them in normalized economics and cache
identity. Preserve them in batch/scenario results and aggregation; an aggregate
depending on a provisional component must disclose that dependency.

Continuous-monitoring history needs separate treatment: a latest spot cannot
establish whether a barrier was touched earlier. The agreed provisional rule
is to start from the last authoritative checkpoint, assume no unreported
earlier touch, and check the latest spot at the valuation timestamp. Mark the
uncovered history interval explicitly. If this scenario triggers a touch,
record valuation time as an assumed hit time, including its effect on
payment-at-hit cashflows. An authoritative current lifecycle state or supplied
touch history supersedes that scenario. This is a conditional scenario and
must not be described as an inference of the actual past path.

### Greeks and observation boundaries

All sensitivities use the same resolved price function and declared state.
Both confirmed historical fixings and provisional assumed fixings stay fixed
during market bumps. Resolve missing-fixing assumptions once from the base
market snapshot before constructing bump cells. A separately requested new
base valuation can refresh assumptions from its latest spot, so differences
between provisional snapshots can contain an assumption revision as well as
market risk; conditional Greeks alone do not explain that revision.

Before a fixing, large delta/gamma can be economically correct as remaining
variance shrinks. At a discontinuity of the unfixed event payoff, report an
undefined classical derivative explicitly. After an outcome is applied,
calculate sensitivities of the resulting state and remaining cashflows.
Do not replace zero time or zero variance with an undisclosed positive floor.

Introduce intraday time-bump durations and explicit theta units. Local theta
stays within the current event/clock segment, with a declared one-sided
convention at boundaries. A finite roll through an observation is a scenario
requiring outcomes and is distinct from a local derivative. Record the actual
bump duration and any adjustment. Existing daily theta semantics remain a
separate compatibility path.

### Engine integration and limits

Resolve the clock, observations, confirmed/provisional history, and settlement
timing once before engine dispatch. The engine consumes that resolved snapshot;
it must not independently decide whether an observation is completed or rebuild
assumed fixings for a bumped market. The Greek layer carries the same context
through each evaluation.

| Engine/method | Intraday integration | Qualification requirement |
| --- | --- | --- |
| Analytical European/digital | Integrated terminal carry, variance, and payment discounting | Correct zero-time/zero-variance limits and Greeks of the actual intraday price function |
| Analytical barrier/touch | Closed form only where its first-passage assumptions hold | An effective terminal volatility is not a general intraday extension |
| PDE | Exact event nodes, clock/coefficient breakpoints, short final propagation, common price/Greek readout | Resolve the shrinking diffusion layer and demonstrate temporal/spatial Greek convergence |
| Discrete-event QUAD | Exact Gaussian interval moments and event recursion | Intervals may span clock knots when only endpoints are observed and integration is exact |
| Continuous-monitoring QUAD | Split at coefficient changes and use a qualified survival/crossing operator | Separate time refinement and first-passage validation |
| MC | Integrated interval moments, exact discrete fixing nodes, shared random numbers for bumps | Qualified Greek estimator and uncertainty; continuous monitoring additionally requires crossing controls |

General intraday profiles do not automatically preserve closed-form
continuously monitored barrier/touch formulas. Matching terminal variance
alone does not match first-passage probabilities when carry and variance run
on different clocks. Existing BGK discrete-barrier approximations must not be
presented as exact near-fixing Greeks.

Here, "uniform wall-clock variance" means a constant instantaneous variance
rate in calendar time. Continuity of cumulative variance alone is insufficient:
a session/lunch profile can have continuous cumulative variance while its rate
changes. The usual flat-rate, flat-carry, constant-variance BSM case is the
standard analytical case.

Nonuniform variance is not universally incompatible with an analytical
barrier method. Writing `v(t)` for the calendar-time variance rate and changing
time to `u(t) = integral v(s) ds`, the log-price drift per unit variance time is
`(r(t) - q(t))/v(t) - 1/2` wherever `v(t) > 0`. A constant ratio permits the
standard fixed-barrier time change, provided zero-variance intervals introduce
no extra carry-driven motion and the payment discounting is also handled
correctly. For example, `r = q = 0` with deterministic nonuniform variance and
a fixed continuously monitored barrier has an exact time-change reduction.
Payment at first hit requires particular care when rates are nonzero.
Such cases must be qualified explicitly; this observation does not certify
the existing engines for arbitrary profiles. Numerical BSM barrier engines
remain the general route for the selected configurable profiles.

For an explicitly requested engine, an unsupported product/profile/method
combination raises a capability error naming the limitation and applicable
alternatives. An explicitly selected automatic engine mode can route to a
qualified numerical method and must return the engine/method actually used.
Do not silently substitute an approximation into an explicit analytical request.

The variance profile does not change contractual monitoring. A zero-variance
interval can still contain a deterministic barrier crossing caused by carry.
Continuous monitoring and discrete observation remain separate capabilities.

Reuse prepared calculations only when the full resolved timestamp, profile,
market, event phase, lifecycle, and provisional assumptions match. A spot bump
may reuse suitable numerical geometry and confirmed/assumed history; a time
roll or new base snapshot must not reuse stale event timing or assumptions.

This work does not expand each engine's existing contractual coverage by
implication. Supported products, monitoring types, clock profiles, and Greek
estimators need an explicit capability matrix.

### Compatibility and API contract

Keep current day-level calls and existing goldens as the baseline. Intraday
mode is explicit, including its timestamp and variance profile. Date-only
observations acquire the exchange-close convention within that mode; legacy
date interpretation remains unchanged outside it.

Reuse `PricingEnvironment`, observation schedules, lifecycle state/ledgers,
and the execution result framework where their semantics fit. The designed
entry point is `PricingSession.value_intraday(...)`, backed by a typed request,
one resolved context, and a structured result. The following illustrates the
approved API shape; these methods and types are not yet implemented:

```python
# env.valuation_date is the authoritative timezone-aware valuation timestamp.
result = session.value_intraday(
    engine=engine,
    product=product,
    pricing_env=env,
    variance_profile=desk_profile,
    lifecycle_state=checkpoint,
    fixings=observed_fixings,
    event_phase="after",
    greeks=("delta", "gamma", "vega", "rho", "theta"),
    greek_convention="point",
    theta_step=timedelta(seconds=30),
    theta_unit="hour",
)
```

The shared resolver creates an immutable `IntradayValuationContext` that is
passed through engine dispatch and all Greek bumps. It snapshots the input
objects and verifies the environment, curve anchors, calendar, profile, and
timestamp are consistent. `event_phase` governs events exactly at valuation;
past and future events retain their timestamp ordering. Reuse the existing
execution framework's request/normalization mechanisms instead of creating a
second independent cache/dispatch system. Existing direct `price()` and daily
Greek interfaces retain their prior return types and behavior outside
intraday mode.

`IntradayValuationResult` contains price, Greeks with units and
per-measure status, confirmed/provisional status, assumption records, cashflow
components, and engine/numerical information. `price` is the remaining claim
PV: contingent PV plus fixed pending receivables, in the declared existing
pricing units. Paid cash is reported separately and is added only by a
consumer requesting the corresponding total-value convention. Report units
and quantity scope; never mix a position ledger with per-contract sensitivities.

A bare floating-point result cannot communicate an assumed fixing. Preserve
the structured result through intraday batch and scenario interfaces and into
risk aggregation. Provisional status remains distinct from numerical status:
a numerically accurate provisional price still depends on assumed history.
An undefined derivative has an explicit status/reason and a missing numeric
value; a failure to demonstrate numerical convergence is reported separately.

Request processing is sequential in economic semantics:

1. Validate and snapshot contract, market, clock, and requested outputs.
2. Resolve contractual fixing/payment timestamps and exact event order.
3. Apply confirmed history and construct any provisional scenario from the
   base snapshot, without mutating the confirmed ledger.
4. Resolve common interval moments, eligible method, and numerical context.
5. Evaluate price and requested Greeks with historical assumptions frozen.
6. Return the state, assumptions, PV components, units, and numerical evidence.

Invalid timestamps/profile budgets, contradictory confirmed records, or
unsupported explicitly requested engine/output combinations produce typed
validation/capability errors. Missing fixing data follows the agreed
provisional policy. A new actual fixing triggers reconstruction from confirmed
history and replacement of downstream provisional outcomes, including assumed
cashflows; it is not appended on top of provisional realized events.

The profile object declares calendar/session identity, timezone,
annualization/quotation basis, reference-interval variance budgets, segment
weights, overnight/nontrading allocation, and shortened-session handling.
Require nonnegative finite weights and the declared budget normalization.
Contract-specific fixing overrides move events without renormalizing the
underlying's variance profile. Profiles are explicit, immutable, and versioned
inputs to the valuation identity.

Raw float schedules need an explicit origin and clock mapping in intraday
mode. Converting them to dates by rounding `time * 365` is not valid.
Settlement and accrual amounts must remain those of the same contract as the
valuation timestamp moves.

## Numerical release gates and delivery

### Greek convention

Expose point derivatives and desk finite-bump sensitivities as explicit,
separately labelled conventions. The intraday request includes
`greek_convention="point" | "desk_bump"`; it is required when Greeks are
requested through the intraday entry point. The result records the selected
convention and all actual bump sizes. Existing daily desk bumps keep their
meaning and size.

The point convention describes derivatives of the declared conditional price
function. A numerical estimator needs a demonstrated bump/mesh limit to claim
that convention. The desk convention evaluates its configured finite move;
it is not tested against a point derivative as though they were identical.
Confirmed and provisional fixings remain frozen under both conventions.

### Gate A: clock, event, and cashflow semantics

Check profile additivity, normalization, re-anchoring, zero weights,
session boundaries, overnight/weekend ownership, early closes, and timezone
conversion, including daylight-saving transitions for a calendar that uses
them. Unknown/ambiguous local-time inputs require explicit resolution rather
than a guessed UTC instant.

Test before/at/after fixing, exact barrier equality and contractual inclusivity,
shared fixing timestamps, coupon/KO/KI priority, expiry, settlement lags, and
no double event application. Compare contiguous time rolls with direct
valuation at the same final timestamp and confirmed state.

Reconcile contingent PV, fixed pending receivables, and paid cash in declared
units. Keep original coupon/accrual amounts unchanged as valuation advances.
Zero-variance intervals must retain deterministic carry, barrier crossings,
and discounting. Existing daily outputs and regression goldens form a separate
compatibility gate and must not be regenerated to hide intraday regressions.

### Gate B: provisional-state integrity

Test missing-fixing provenance, repeated pricing without input mutation,
actual-fixing replacement, reversal of assumed KO/KI, downstream coupon/ledger
reconstruction, continuous-history assumptions, aggregate status, and cache
separation. Verify the entire bump family uses assumptions resolved from the
unbumped base snapshot. A new provisional base snapshot may revise assumptions;
that is a distinct economic change and must retain its provenance.

Include an assumed KO with zero conditional equity delta but a nonzero rate
sensitivity on delayed payment, followed by an actual non-KO fixing restoring
the live claim. Include the opposite correction and coupon-memory cases.

### Gate C: independent numerical accuracy near fixing

Use a time-to-fixing ladder of one day, six hours, one hour, fifteen minutes,
five minutes, one minute, ten seconds, and one second. Test the exact event
boundary separately with explicit phase/state. Timestamp support below the
qualified horizon does not imply a Greek-accuracy certificate there.

For each positive-variance case, place spots on both sides of each relevant
barrier using fixed basis-point offsets and offsets scaled by remaining
standard deviation. Include drift-adjusted transition centers and exact
barrier/strike equality. Exercise uniform, session-only, weighted overnight,
lunch, holiday, and zero-variance profiles; cover representative late-life
Snowball/Phoenix/KO-reset states within each engine's existing contract scope.

Use independent analytical digital controls and an independently implemented
Gaussian-transition reference for discrete barriers/autocallables. Distinguish
agreement with another production discretization from external accuracy.
Continuous barriers need first-passage controls: analytical admissible cases,
the exact zero-carry/zero-rate time-change case, and independently qualified
nonuniform-clock cases with nonzero carry.

Refine at least three levels of the relevant spatial, temporal, quadrature,
and bump parameters, with domain/grid-placement checks where applicable.
Record actual delivered resolution and resource caps. The final Greek must
be stable under further justified refinement; a stable price alone does not
qualify its derivatives.

For each output, define absolute and relative error budgets in explicit
price/Greek units before tuning release candidates. Count reference uncertainty
in the budget and retain an absolute criterion near a zero-valued Greek.
Pilot/reference work establishes the measurable budgets and qualified envelope;
candidate results must not determine relaxed tolerances. Finite desk bumps
are compared with the same finite-move operation on the independent reference.

MC records path counts, seeds, estimator, uncertainty, and timestep/bump
refinement. Confidence intervals quantify sampling uncertainty and do not
replace checks on discretization or estimator bias. Results outside a
demonstrated envelope carry an explicit unqualified/inconclusive numerical
status. Mathematically undefined derivatives are a separate status, never
replaced by zero or by an undocumented smoothing width.

### Gate D: performance and adoption evidence

Benchmark single-trade price plus requested Greeks, a 101-point spot-risk
curve, and a 100-trade mixed-product batch. Record cold preparation and warm
evaluation separately, together with memory, hardware, threads, engine
settings, and the accuracy achieved. Existing scenario/batch services should
preserve the new context and result provenance.

There is no agreed hard latency SLA. Report measurements at the same declared
accuracy; a faster setting with degraded or unreported risk accuracy is not
an improvement. Product/engine/profile/output capabilities are published only
for combinations that clear the applicable gates.

### Delivery stages

1. Establish the current baseline, concrete capability inventory, independent
   reference cases, and fixed numerical acceptance budgets.
2. Implement the shared timestamp/profile resolver, immutable valuation
   context, lifecycle/provisional reconstruction, and structured result.
3. Integrate analytical digitals/admissible analytical barriers, then PDE,
   QUAD, and MC through the common contract; qualify each supported output.
4. Complete intraday Greek/time-roll handling, existing execution/batch
   integration, end-to-end examples, regression checks, and benchmarks.

These are dependency stages, not a reduced first-release scope. The agreed
release still covers applicable BSM analytical, PDE, QUAD, and MC paths.
Create a separate file-level implementation plan before production edits.

## Current-code evidence inspected during design

- `quantark/util/calendar/day_counter.py`: whole-day year fractions.
- `quantark/util/calendar/trading_clock.py`: daily-knot variance/carry maps.
- `quantark/priceenv/pricing_environment.py`: date-based environment and
  existing clock configuration.
- `quantark/asset/equity/product/option/observation_schedule.py`: shared
  observation resolution.
- `quantark/asset/equity/lifecycle/autocallable.py` and `barrier.py`: daily
  observation/aging and lifecycle state.
- `quantark/asset/equity/engine/quad/v2/engine.py`: explicit before/after event
  phases already exist for QUAD V2.
- `quantark/asset/equity/engine/pde/grid/request.py` and `time.py`: event-aligned
  grids with floating-time deduplication and daily density settings.
- `quantark/asset/equity/riskmeasures/greeks/numerical.py`: daily theta can
  return zero when the bump reaches maturity or drops all observations.
- `quantark/asset/equity/engine/analytical/barrier_analytical_engine.py`:
  effective scalar inputs and BGK approximation for regular discrete grids.
- `quantark/execution/contracts.py` and `diagnostics.py`: economic payloads
  and operational diagnostics have distinct identity semantics.

## Implementation handoff

The next artifact is a separate file-level implementation plan covering the
delivery stages, component ownership, dependencies, commands, focused tests,
and qualification gates. The plan must inventory the active checkout before
editing: existing clock, lifecycle, and QUAD work may have advanced since the
code inspection recorded here. Preserve unrelated working changes.

Desk profile weights and conventions are explicit deployment inputs rather
than guessed library defaults. Numerical error budgets and the measured
qualification envelope are outputs of the baseline/reference stage, fixed
before candidate tuning. These outstanding implementation artifacts do not
alter the approved product behavior or authorize a claim of numerical support.
