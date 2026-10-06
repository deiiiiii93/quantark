# The 1D PDE core (0.10.0)

Every 1D PDE route prices through one path: European, American, the barrier family
(single, double, one-touch, double one-touch, knock-in by composition), Snowball,
Phoenix, KO-reset, DCN, and the local-vol vanilla, barrier, DCN, Snowball and Phoenix.
The pieces live in `quantark/asset/equity/engine/pde/core/`. This page explains how they
fit together, how to plug in a custom component, and what the core keeps from 0.6.0 on
purpose. The design is `docs/superpowers/specs/2026-09-30-pde-1d-one-core-design.md`;
0.8.0's numerical policy (damping form, projection, grid day, cap, far-field rows) is
`docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-design.md`; 0.9.0's event
statistics are `docs/superpowers/specs/2026-10-03-pde-1d-event-statistics-design.md`;
0.10.0's remaining routes and base-family repairs are
`docs/superpowers/specs/2026-10-04-pde-1d-remaining-routes-design.md`.

Only the 2D Heston / SLV solvers are not on the core. The vanilla, barrier and DCN ones
take ADI grid sizes of their own (`n_x`, `n_v`, ...). The Snowball and Phoenix ones read
the built-in stepper's settings (`theta`, `event_theta`, the damping counts) from
`PDEParams.stepper` and damp in whole steps: an explicit `damping="half_step"` raises
(`ROUTE_RULES["2D"]`, the one row left in `route_applicability`). `explain` does not
cover the 2D solvers. Below the vol-of-vol
floor `_SIGMA_MIN` (1e-4) the Heston and SLV barrier solvers do not run their 2D grid:
they price the deterministic-variance limit on the core, `LocalVolBarrierPDESolver` on
`quantark.volmodels.heston.deterministic_limit.DeterministicVarianceLocalVol`, whose local
vol is `L(S, t) sqrt(v(t))` with `v(t) = theta + (v0 - theta) e^(-kappa t)` (`L = 1` under
Heston, the leverage surface under SLV).

## Three stages

```
engine.price / calculate_greeks / calculate_event_stats / price_with_events
explain / prepared session / bump context / intraday route / life surface
    -> prepare(solver, product, env, request, frozen=None, session=None) -> PreparedProblem
    -> march(prepared) -> MarchResult
    -> read(prepared, result) -> price / greeks / event statistics / surfaces
```

- **`OutputRequest`** (frozen) names what one call wants: the `operation`, `price`,
  `greeks`, `event_stats` (with optional `streams`), `grid`, `life_surface` with its
  `extra_times`, an `external_pv` (the 2D solvers' statistics-only march), and
  `short_circuit`.
- **`prepare`** does, in order: the per-operation terminal outcome (expired, knocked out
  at valuation: no grid, no march), route applicability, the adapter's solve state, the
  market snapshot, the grid request, the `Layout` (fresh, frozen for a bump context, or
  the session's stored one), the observation maps, the adapter's ordered events, the
  market samples and the coefficients, the θ schedule, the column groups, and the
  identities.
- **`PreparedProblem`** (frozen) is everything one march consumes: the adapter, solve
  state, layout, coefficients, stepper and θ schedule, column groups, factorization
  cache, readout, events, and `identity`. With `terminal` set, the geometry is `None` and
  there is nothing to march.
- **`march`** is one loop over the steps, last to first. For each column group it takes
  the boundaries, calls `stepper.step`, validates the result, and runs the group's
  after-solve hook (the American obstacle). Then the step's events run in the adapter's
  order. The loop has no product branches.
- **`read`** turns the retained columns into the request's outputs through the readout.

A knock-in barrier prepares a `CompositeProblem` of two parts, marched independently: the
vanilla, and the knock-out leg carrying the knock-in's rebate `R` paid at the terminal
payment time. They compose as

```
knock-in = participation × vanilla + R × DF(0, T_pay) − leg
```

for the price; the Greeks are `participation × vanilla − leg` (the constant is in the
price only). The double barrier's vanilla carries no participation. The identity holds
path by path only when the first breach ends the leg, so a knock-in under any aggregation
other than `STOP_FIRST_HIT` raises `ValidationError` (continuous and EXPIRY monitoring are
first-hit), as does a schedule record whose payoff differs from the product's rebate.
Knocked in at valuation, the composite is `participation × vanilla` and the rebate is
forfeited. Each part is in contract units (both copy `contract_multiplier`). A bump
context freezes one layout per part at the base market (`_frozen_parts`); a part with no
base-market solve (the leg of a product knocked in at the base spot) binds fresh under
the bumped market.

## Column groups and policies

| Family | Value groups | Backend |
|---|---|---|
| European, knock-out barriers, touches, American | one column | sparse LU |
| Local-vol European and knock-out barrier | one column | banded (one coefficient set per step defeats the factorization cache) |
| Knock-in barriers | two prepared problems (vanilla, knock-out); knocked in: the vanilla alone | each part its own family's |
| DCN, LV DCN | `alive` (never knocked in) and `ki` as one two-column group | banded |
| Snowball, KO-reset, LV snowball | alive and knocked-in as one two-column group | banded, one `solve_banded` per step; LU with two one-column groups when the interior has two nodes or fewer |
| Phoenix, LV phoenix | one group per (memory state, surface), plus the two termination surfaces under EXPIRY coupons, each one column | banded or LU per column |
| Event statistics | one rolling group of `2·n_cols` columns (both surfaces hstacked) | banded only; all-zero columns skipped |

| Policy | Boundary | θ | After events |
|---|---|---|---|
| value | the adapter's Dirichlet values, set before the solve | the schedule; θ=1 on zero-diffusion steps (every value group, Phoenix included since 0.8.0) | none |

Since 0.9.0 the statistics group takes the value policy too (`statistics-legacy`, the
carried rows and the zero-slope copy are gone).

## Event statistics (0.9.0)

Event statistics are a column group of the price march: `calculate_event_stats` marches
the value and statistics groups together and does not re-price. The package is
`core/statistics/`:

- **`columns.py`, `ColumnLayout`** — the block per surface: knock-out units `KO_k` (the
  pre schedule on KO-reset), the KO-reset post-schedule units, the Phoenix triggers
  `P_k`, the knock-in pair `KIS` / `KIE`, then the maturity leg `M` (KO-reset: `M_pre`,
  `M_post` and the maturity units `U_pre`, `U_post`), the Phoenix per-date coupon cash
  per memory state, the EXPIRY payment buckets per state and the termination unit.
  `streams` prunes columns; pruned fields read 0.
- **`rows.py`, row allocation** — the value's Dirichlet rows are produced one code branch
  per cash term (`RowTerm`); the statistics send each term to the column whose readout
  multiplies it (`M`, a `KO` unit, a termination unit). The cash columns' rows therefore
  sum to the value's rows bit for bit. Rows are allocated from the value, never designed.
- **`breach.py`, the stages** — the adapter's event functions (knock-out, knock-in, the
  Phoenix coupon / joint / termination stages, the KO-reset seed) take an optional
  `breach`. The value passes `None` and writes its scalar cash; a statistics group passes
  a `Breach` that returns the block's values where the event fires. The same function,
  the same order (the tie rules included), the same cell-average projections — so the
  Phoenix coupon-barrier smoothing reaches every column.
- **`snowball.py` / `phoenix.py` / `ko_reset.py`** — the families' `RollingStatistics`
  (`allocate`, `terminal`, `rows`, `step_events`, `store`, `read`); θ is the value
  group's schedule (θ = 1 on zero-diffusion steps).
- **`readout.py`** — interpolates the columns at the spot, applies the valuation-date
  capture (a coupon decided today reads state 0 or, on a memory miss, state 1), and
  assembles the dataclass and its ledger with the marched maturity leg as the terminal
  row. `reconciliation_error = pv − Σ expected_discounted_cashflows` is round-off on
  every native route (the suite bounds it at 1e-10).
- **`terminal.py`** — a product knocked out at valuation (the family's
  `decide_observations_at_valuation` with the carried knock-in state) is a terminal
  outcome: deterministic statistics from the decision, no layout, no march.

The statistics-only path (`external_pv`) prepares its own stages and termination tables
and is the 2D solvers' surrogate. The 2D Heston / SLV solvers refuse statistics unless
built with `statistics="flat_vol_surrogate"`; the result's `provenance` names both
models.

**The reference fixture.** `test/pde_statistics/reference.py` is an exact two-state
(not knocked in / knocked in) Gauss–Legendre recursion for a flat-GBM Snowball or
memory-free Phoenix. It uses no engine code. It refuses what it cannot represent:
non-flat curves, non-constant barriers, EXPIRY coupons, memory, reverse products and
an observation at the valuation date. It also checks the integration domain, mass
conservation, settlement and terminal payment times, and finiteness. Its values are
banked in `reference_values.json` with the quadrature error, the conservation defect
and the Monte Carlo qualifier's z-scores. Regenerating the JSON (`build_reference.py`)
is a deliberate act with provenance, as for the goldens. KO-reset is gated against a
32,001-point QUAD in `ko_reset_reference.json`; the memory and EXPIRY Phoenix against
QUAD V2.

## Adapters

One adapter per family (`core/adapters/`): `EuropeanAdapter`, `AmericanAdapter`,
`BarrierAdapter` (with `DoubleBarrierAdapter`, `OneTouchAdapter`,
`DoubleOneTouchAdapter`), `SnowballAdapter`, `PhoenixAdapter`, `KOResetAdapter`,
`DCNAdapter`, and the local-vol adapters below. An adapter supplies the solve state, the
grid request, terminal fills, boundary values, the ordered events, captures, the column
groups for a request, and the readout branch. The adapters delegate to the product
helpers the solvers kept (the 2D solvers inherit them), so per-solve state still lives on
the solver. The march itself holds none.

## The four routes moved in 0.10.0

| Solver | Adapter | Parent |
|---|---|---|
| `LocalVolPDESolver(params, local_vol_surface)` | `LocalVolEuropeanAdapter` | `EuropeanPDESolver` / `EuropeanAdapter` |
| `LocalVolBarrierPDESolver(params, local_vol_surface)` | `LocalVolBarrierAdapter` | `BarrierPDESolver` / `BarrierAdapter` |
| `DCNPDEEngine(params)` | `DCNAdapter` (`core/adapters/dcn.py`) | `BasePDESolver` / `FamilyAdapter` |
| `LocalVolDCNPDEEngine(local_vol_surface, params)` | `LocalVolDCNAdapter` | `DCNPDEEngine` / `DCNAdapter` |

Class names and constructor signatures are unchanged. Each takes the core's defaults:
200 / 400 / 800 concentrated points and 4 / 4 / 8 steps per day, the core damping rows,
the cell-average projection, the parent family's Dirichlet rows (the DCN's below) and the
shared domain rule, and every `PDEParams` field is honoured.

**Local vol.** The three local-vol adapters (`core/adapters/local_vol.py`) are the parent
family's adapter with `LocalVolAdapterMixin` (in `core/adapters/snowball.py`, shared with
the local-vol Snowball and Phoenix): `NODAL` samples, one local-vol row per step at the
interval midpoint on the solve's S grid (`sample_local_vol`). The local-vol European and
barrier adapters skip the constant-vol smile guard and march their one value column on
the banded backend. On the solver side `LocalVolSurfaceMixin` (`pde/local_vol_surface.py`)
holds the surface for all five local-vol solvers: `_build_surface` takes a prebuilt
`LocalVolSurface` or builds the Dupire surface from the environment's `GridVolSurface`;
`_with_surface` builds it once around a whole call (price, Greeks, session outputs, spot
curve; the barrier's bump context too), so every solve of that call, and both parts of a
knock-in, read one surface object; and `_flat_exact_step_coefficients` returns the
per-step sets unchanged (no flat-exact replacement). A local-vol
knock-in's vanilla part is `LocalVolPDESolver` on that surface
(`LocalVolBarrierPDESolver._vanilla_solver`; `EuropeanPDESolver` on the flat route).

- `LocalVolPDESolver.calculate_greeks` returns `price, delta, gamma` from the grid of one
  solve (pointwise, where 0.9.0 bumped 1 %) and `rho`, `theta` re-solved on a frozen
  layout with the surface held fixed (`localvol_greeks.local_vol_rho_theta`: `rho` per
  1 % from a central shift of `BumpConfig.rate_bump`, `theta` per day from a roll of
  `BumpConfig.time_bump_days`). At maturity or in a terminal lifecycle state it returns
  the three grid keys.
- `LocalVolBarrierPDESolver.calculate_greeks` is the barrier family's: `price, delta,
  gamma` from the grid, composed for a knock-in.
- The kernel's vanilla half stays in `volmodels/localvol/pde_kernel.py` for
  `FxLocalVolPDESolver`; its barrier half (`price_barrier_lv_pde`) left the library and
  is the test tree's independent oracle (`test/pde_routes/lv_barrier_kernel.py`).

**DCN.** `DCNAdapter` marches two surfaces, `alive` (never knocked in) and `ki` (knocked
in), as one banded group. The contract comes from `build_dcn_grid_context(product)`.

- *Grid request.* The horizon is the final observation date; every interior daily date is
  an event node; the critical prices are the spot and the knock-in, knock-out and coupon
  barriers; the bound anchors are the spot and the initial price.
- *Terminal.* `ki` is the loss put `−(notional / initial_price) × participation ×
  max(k_loss − S, 0)` discounted from its payment time to the horizon, `alive` zero; then
  the last observation's events.
- *Events at a date*, coupon, knock-out, knock-in, as one projection per surface: on
  `alive`, `project_piecewise` over the thresholds present that day (`ki` below the
  knock-in barrier, the knock-out cash at or above the knock-out barrier, which has
  priority over the coupon, `alive + coupon` at or above the coupon barrier); on `ki`,
  `project_between` at the knock-out barrier on knock-out dates.
  `event_projection=NODAL` applies the legacy masks (`apply_dcn_events`) instead.
- *Far-field rows.* Lower, both surfaces: the knocked-in loss on the edge carried forward
  to the horizon by the solver's forward factors. Upper: the next knock-out's cash
  discounted to the row's time, plus, on `alive`, the coupons due before it.
- *Readout.* The `ki` column when the product is flagged knocked in at valuation or the
  spot is at or below the knock-in barrier, else `alive`; times `direction_sign`.
  `calculate_greeks` returns `price, delta, gamma` from that column.
  `price_detailed` returns `DCNPDEResult`, whose `num_time_steps` is the delivered step
  count. A session grid or spot curve is a `BranchSolution` (below).

`LocalVolDCNPDEEngine` adds `LocalVolSurfaceMixin` and samples through
`LocalVolAdapterMixin`. `apply_dcn_events` and `DCNPDEResult` stay in `dcn_pde_solver.py`
for `HestonDCNPDESolver` (2D, not on the core).

## The three components

`PDEParams` has three component slots. `None` means the built-in:

| Slot | Built-in | Protocol |
|---|---|---|
| `stepper` | `CrankNicolson()` | `schedule(layout) -> θ per step`; `step(state, s, dt, theta, boundaries) -> advanced columns`; `evidence() -> dict` |
| `operator` | `CentralDifference()` | `assemble(samples, layout) -> StepCoefficients`; `supports: frozenset[SampleKind]`; `evidence() -> dict` |
| `readout` | `StandardReadout()` | `price(column, spot, layout)`; `delta_gamma(column, spot, layout)`; `evidence() -> dict` |

`CrankNicolson` carries the scheme's settings: `theta` (0.5), `event_theta` (1.0),
`damping` (`"full_step"` or `"half_step"`), `terminal_damping_steps` and
`event_damping_steps` (`None` = the route's row of `grid.config.DAMPING_DEFAULTS`; 0
disables). The core has two rows chosen by the contract alone
(`core.adapters.base.continuous_knock_in(product)`): half-step (1, 1) for every product without
continuous knock-in, full-step (1, 2) for one with it. Every 1D route takes these two,
DCN and the local-vol vanilla and barrier included (0.10.0 deleted their own rows); only
the 2D solvers have a row of their own, full-step (1, 2). Its schedule is 0.6.0's rule: `theta` everywhere,
`event_theta` on the damped intervals before each interior event node, 1.0 on the terminal
intervals, terminal wins on overlap.

**Time refinement.** Half-step damping is a property of the time grid, not of the
schedule: `CrankNicolson.time_refinement()` returns a `TimeRefinement(terminal_intervals,
event_intervals)` that `prepare` puts on the `GridRequest`, and `build_time` splits each
of those intervals below the terminal node and below every event node at its midpoint,
after the fill. The midpoint is an ordinary node and each half is half the interval's
constructed `dt`. `time_fill(request, config)` counts the delivered steps without
building, and is what `build_time` checks against `max_steps`: a grid over the cap raises
`ValidationError`, it is never scaled. The stepper's `validate_schedule` refuses a layout
whose refinement does not match its own (a frozen or shared layout built for the other
form), and `validate_external_layout` compares `refinement` and `day_count`. The grid day
is the request's `day_count`, the pricing environment's `bus_days_in_year`.

A worked custom stepper, fully implicit everywhere through the built-in arithmetic (the
test fixture `test/pde_core/components_fixtures.py`):

```python
from dataclasses import dataclass

import numpy as np

from quantark.asset.equity.engine.pde import EuropeanPDESolver
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.param import PDEParams


@dataclass(frozen=True)
class BackwardEuler:
    def schedule(self, layout):
        return np.ones(layout.time.actual_steps)

    def step(self, state, s, dt, theta, boundaries):
        return CrankNicolson().step(state, s, dt, 1.0, boundaries)

    def evidence(self):
        return {"kind": "backward_euler"}


solver = EuropeanPDESolver(PDEParams(stepper=BackwardEuler()))
```

It prices the same as `CrankNicolson(theta=1.0, terminal_damping_steps=0,
event_damping_steps=0)` to round-off.

**Validation, never fallback.** The core checks every component result and raises
`NumericalError` naming the component: the schedule (length, finite, in [0, 1]), the
coefficients (shapes, finite, a valid `set_index`), the advanced columns (the group's
shape, finite), and readout values (finite). The state a stepper is handed is read-only.
A stepper that writes into `state.v_next` raises `ValueError: assignment destination is
read-only`, and must allocate its own scratch space.

**Where custom components run.** On every 1D route: since 0.10.0 the local-vol vanilla
and barrier solvers and DCN are core routes and take them as the others do. An operator
must list the route's `SampleKind` in `supports`: `SCALAR` on the Black-Scholes routes
(the DCN included), `NODAL` on the local-vol ones. A custom operator's step sets are
marched as assembled, never collapsed to the scalar stencil. The built-in's flat-exact
replacement applies to the built-in only. Nodal (local-vol) sets are never collapsed
either, even with one step (`FamilyAdapter.coefficients` keys the collapse on the sample
kind as well as on the number of unique sets), so a one-step local-vol solve no longer
returns the flat-vol price.

## Identities and the fingerprint rule

A component's identity is its value fingerprint (`try_fingerprint(component)`). A frozen
dataclass of plain values fingerprints itself; an arbitrary object has none.

- **Layout fingerprint:** the arrays the march consumes (`x`, `dx`, `t`, `dt`), the event
  and extra node indices, and the config key.
- **Coefficient key:** the layout fingerprint plus every sampler input (rate curve,
  dividend yield, vol or local-vol surface, spot, reference strike) and the operator.
- **Factorization key:** the coefficient key, the stepper, the groups' backends, and
  their effective θ schedules.
- **Problem identity:** the factorization key, the readout, and the complete economic
  state (product token, solve state, event table, valuation date and day count, the
  effective settlement calendar, `event_projection`, the output request).

A component without a fingerprint makes the problem unfingerprintable. `price` works,
no session factorization is stored or reused, and `explain` reports `problem: None`.

**The fingerprint covers configuration and class, not code.** Reassigning a method
leaves the fingerprint unchanged. That is enough for session artefacts, which live in one
process's memory, and not enough for certification. So **only the built-in components
are certifiable**: a study declares its PDE candidate through `PDEParams.from_config`,
which cannot name a custom component, and a candidate built any other way is refused at
load.

## Discrete events of the base family

The barrier, double-barrier, one-touch and double-one-touch solvers state each
observation as data: `EventRecord(lower, upper, value, op)` (breached where `s >= upper` or
`s <= lower`; `op` is `SET` or `ADD`) and `DiscreteEvent(records)`, in the order the solver
applies them (`_step_event`, `_terminal_event`). `grid.apply_discrete_event(nodes, column,
event, projection)` applies one observation:

- `NODAL`: for each record in order, `column[breached] = value` or `+= value` (the 0.7.0
  masks; `fold_nodal` runs the same fold in place on a grid column on the per-step path);
- `CELL_AVERAGE` (the default since 0.8.0): the same fold as a function of log-price,
  projected **once** per time node: the breakpoints are the records' distinct thresholds,
  and every dual cell a threshold crosses gets the exact cell average of the whole event.
  Projecting record by record is not equivalent when two thresholds share a cell.

Continuous monitoring (the barrier is the domain edge) stays nodal in every mode. The
valuation-date observation is not marched at all: it is the decision of the next section,
applied where a value is read. `explain`'s `events` section lists each observation's
records and treatment (`BaseFamilyEvents`), the valuation-date one as `"readout"`.

EXPIRY monitoring on the single barrier is one observation at the maturity node (the
terminal event, cell-average projection), its rebate paid at the terminal payment time.
The double barrier and the touch products have no EXPIRY far-field row; before maturity
their adapters return a composite with no marched part (`expiry_replication`): the
terminal value is a function of the terminal spot alone, priced as European vanillas and
cash digitals on `BlackScholesEngine` and `DigitalOptionAnalyticalEngine`. The adapter's
`replication` hook makes `request_geometry` return `None`, so no grid is requested or bound
(bump contexts, session artefacts). Greeks are `GreeksCalculator`'s own of the legs
(`BaseEngine.static_replication`); the facade's are the calculator on the solver's params,
and a session GRID reads them at each requested level (it must name its levels).
Under DISCRETE or EXPIRY monitoring with first-hit aggregation, a single knock-out's (and
so a knock-in's leg's) far-field row on the barrier's side is the certain knock-out at the
next observation:
when an observation lies at or after the row's time and the edge is beyond that
observation's level, the row is that record's cash discounted from its payment time.
With no observation ahead, or the edge inside the level, it is the vanilla asymptote of
0.9.0. The opposite row, the `ACCUMULATE` rows and the double-barrier and touch rows did
not change.

## The decision at the valuation instant and at maturity

The barrier family (single, double, one-touch, double one-touch; `_BarrierFamilyAdapter`
in `core/adapters/barrier.py`) decides a hit at valuation by the product's own records,
the counterpart of the autocallables' `_valuation_decision`.

- **The event.** `valuation_event(product, env)` returns the valuation instant's
  `DiscreteEvent` as the march would build it. Under continuous monitoring it is one `SET`
  record at the barrier or barriers whose value is the hit outcome (the knock-out's
  rebate, the touch value). Under a schedule (DISCRETE; EXPIRY is the schedule with one
  record at maturity) it is the records whose observation time is the valuation instant,
  built by the helper the march's `_step_event` uses (`_event_for_records`): each with
  its own level, its cash discounted from its own payment time, and its operation. Otherwise it is `None`, and the product is
  solved from its spot even when that spot lies beyond a barrier.
- **The continuation.** The march does not apply the valuation-date observation: the
  surface at t = 0 is the value an instant after that observation, as if it had not
  fired.
- **The fold.** `decision_at` folds the event at a spot by the march's nodal rule
  (`grid.events.fold_decision`): for each record in order, `SET` replaces the price by the
  record's value and the delta and gamma by zero; `ADD` adds the record's value to the
  price and leaves the delta and gamma. What a record does is its operation, never a rule
  inferred from the aggregation mode (a no-touch's maturity record is `SET 0` under either
  mode). When a `SET` record fires at the base spot the outcome is terminal
  (`"barrier_hit"`) and nothing is marched; otherwise `read` folds the event at the spot
  into the marched price, the Greeks and a session's `npv`. A knock-in asks the same
  function whether it is knocked in.
- **At maturity** (`tau ≤ 0`), before any composite is built and identically for
  `PRICE`, `GREEKS` and `SESSION_OUTPUTS`, a value is read in the same two steps
  (`maturity_value`). The continuation is the solver's terminal payoff as the march
  starts from it (`_terminal_payoff`): the payoff × the participation × the terminal
  payment discount `DF(T_pay) / DF(T)` × the contract multiplier, with its slope by the
  same factors as delta and gamma zero; for a no-touch the rebate × that discount, for a
  one-touch zero. The decision is the maturity event (`_maturity_event`): under
  continuous monitoring the hit record; under a schedule the maturity records as
  `_terminal_event` builds them, always present under EXPIRY and in a DISCRETE schedule
  only if it has one. A knock-in at maturity is the composite's identity on terminal
  values, with no part product built: `participation × payoff × that discount` (delta
  likewise) when a `SET` maturity record fires, else `R × DF(0, T_pay)` with delta zero.
  A session's outputs at maturity carry that value as `npv` and no solution.

The other base-family adapters give one expired value under every operation too:
`EuropeanAdapter` answers `SESSION_OUTPUTS` at expiry as it answers `PRICE` (with the
terminal payment discount and a lifecycle state's receivable), and `AmericanAdapter`'s
Greeks and session outputs at expiry take its price's exercise payment discount.

## Barrier-family grids: the per-level rule, `CompositeSolution`, `BranchSolution`

Two consumers read one seam: a prepared session asked for `GRID`, and the public
`calculate_spot_greeks_curve` (on `PDEEngine` and every solver). Both run
`OutputRequest(SESSION_OUTPUTS, grid=True)` and hand the solution to
`BasePDESolver._grid_projection_from_solution`; neither branches on the product. Under
`SESSION_OUTPUTS` the barrier family applies the product check, the valuation-date
decision and the composite test as `PRICE` does.

- **A single surface** is a `PDESolutionResult` carrying `decision` (the valuation event)
  and `cash` (a lifecycle state's spot-independent receivable on the European). Each row
  is read in two steps: the price is interpolated and delta and gamma differentiated on
  the solved surface at the level, then the decision is folded at that level by the rule
  above and the cash added. At a level where a knock-out or touch product is hit the row
  is the record's value with delta and gamma zero, not the edge node's one-sided
  gradients.
- **A knock-in** returns a `CompositeSolution(vanilla, leg, participation, constant,
  decision)`: each part's solved surface, the participation, the constant
  `R × DF(0, T_pay)` and the knock-in's own valuation event. Its `rows(levels)` projects
  each part on its own grid and composes level by level:

  ```
  level not a hit:   price        = participation × vanilla + constant − leg
                     delta, gamma = participation × vanilla − leg
  level a hit:       price, delta, gamma = participation × vanilla
  ```

  "A hit" is a `SET` record of the knock-in's decision firing at the level: at or beyond a
  continuously monitored barrier, or at or beyond the level of a record on the valuation
  date. The leg is not read at a hit level. With no levels requested the rows are at the
  vanilla part's nodes. The session's `npv` is the composed price.
- **The DCN** returns a `BranchSolution(alive, ki, x_vec, s_vec, ki_barrier,
  knocked_in)`: each row is read from the branch the readout takes at that level (`ki` at
  or below the knock-in barrier, or when the contract is flagged knocked in; `alive`
  otherwise) and differentiated within that surface.
- **Decided at the base spot.** A product on which a `SET` record fires at the spot, or a
  knock-in knocked in there, has no base-market solve to project (a knocked-in knock-in
  has no leg). Its session outputs carry `solution=None`: a session's `GRID` request fails
  closed with `CapabilityError` and `PV` alone is served; the public curve falls back to
  point Greeks, one `calculate_greeks` per level, rows marked `reprice`. At maturity there
  is no surface either, and the public curve likewise returns the terminal value level by
  level.

A level outside a part's solved domain is read from that part's edge node (`np.interp`
clamps), for every family.

## The knock-in-side far-field row (0.10.0)

Under discretely observed knock-in, European included, the not-knocked-in surface's row on
the knock-in side takes the knocked-in surface's row and its statistics allocation:

- a standard product: the lower row, where the lower edge is below every knock-in level;
- a reverse product: the upper row, where the upper edge is above every knock-in level;

in both cases only at a row time with a knock-in observation at or after it. The rule is
keyed on the contract (the product has a knock-in barrier and `continuous_knock_in` is
false) and reads the product's resolved knock-in observation times (the cached knock-in
profile), never a solver's index set, so a bump or a lifecycle flag cannot change it and
the 2D solvers, which never fill that index set, are covered. After the last knock-in
date, on an exhausted schedule, or for a KO-reset after its first schedule has ended, no
knock-in can be observed and the row is the 0.9.0 one. It lives once in the shared row
builder, `SnowballPDESolver._knock_in_side_row` (called from `_boundary_rows_v0`, taking
the row and its `RowTerm`s from `_boundary_rows_v1`), so the Snowball, Phoenix and
KO-reset solvers, their local-vol variants and the 2D solvers' inherited rows all follow
it.

Beyond every knock-in level a knock-in at the next observation is certain, so the
not-knocked-in value at that edge is the knocked-in value there. Under monthly or daily
knock-in the next projection already resets the edge, so those prices move at round-off;
under European knock-in nothing does, and the 0.9.0 row ("principal plus rebate to
maturity") left every European-knock-in Snowball about +4.4e-3 contracts high at any
preset. The row is still a constant: with `bounds` set close to the knock-in barrier the
right row is a knock-in probability, and an explicit bound belongs well beyond the
barrier.

## Mesh kernels

`space._legacy_concentrated_mesh` integrates the concentrated mesh's ODE with RK4, about
460,000 evaluations per layout, which was most of a cold solve. `grid.mesh_kernels` runs the
same IEEE operations in the same order on Python floats, or compiled by Numba when the
`accel` extra is installed (`mesh_kernels.backend()` says which); both are bit-identical to
the NumPy reference on every input the builders produce. **Deferral rule:** a kernel never
handles an unusual value. A non-finite or zero denominator, a non-finite sum or node, or
more than eight critical points raises `Deferred`, and the call re-runs on the NumPy
reference under the caller's `np.errstate`, so every exception and the fallback mesh happen
exactly as before. CI's four Python jobs require the Python kernel
(`QUANTARK_REQUIRE_MESH_KERNEL=python`); the `accel` job requires Numba and runs the grid,
core, anchor, banked-certificate and execution tests.

## The evidence record

Certificates hash `{declared params, engine, grid: <resolved record>}`. The candidate
builders construct the engine they run from the mapping they record
(`declared_pde_params` → `PDEParams.from_config`), and the record is
`resolved_grid_evidence(params, day_count=...)`. That is the eight resolved `GridConfig`
fields, `day_count` (on the density basis, after `max_steps`, where 0.7.0 had it: the
environment's business days, `environment_day_count`), the continuous-knock-in row's two
damping counts under their 0.6.0 keys (`terminal_damping_steps`, `event_damping_steps`),
`damping` only when that row's form is half-step, and `discrete_monitoring_damping`
(`[form, terminal, event]`) only when the core row resolves differently, which is the
0.8.0 default. A full-step (1, 2) configuration on a 252-day environment with
`max_steps=5000` serializes to the 0.7.0 record byte for byte. The record is the numerical
grid plus the scheme's damping, not a serialization of `GridConfig`.

`PDEParams.evidence()` (what `engine_config(params)` serializes) records `accuracy`, the
`grid` overlay, the stepper (the built-in's settings; a custom stepper's fingerprint or
`"uncertifiable"`), `operator` and `readout` only when not the built-in, `event_projection`,
`cache_enabled`, and the inherited `EngineParams` fields.

## `explain`

`solver.explain(product, env)` (and `PDEEngine.explain`) runs `prepare` and skips the
march. The returned `PDEPlan` has sections `route`, `settings` (requested / effective /
source / applicability for every `PDEParams` field, every `stepper.*` setting and every
`grid.*` field), `space`, `time`, `events`, `scheme`, `outputs`, `identity`, and
`terminal` (per operation: marches, composes, or its terminal reason). Node lists live in
`to_dict()`, not in `str(plan)`. A request that cannot price raises the same error `price`
would. Every 1D route has `explain` and reports every field honoured (a 1D adapter
declares no route rule). The 2D Heston / SLV
Snowball and Phoenix solvers' `explain` raises `PricingError`.

## Kept on purpose

The core reproduces these 0.6.0 behaviours bit for bit. Each is decided in a later
sub-project, under re-certification, not here. (0.8.0 decided three former entries: Snowball
`NODAL` masks interior events nodally, Phoenix takes θ = 1 on zero-diffusion steps, and under
continuous KI the knocked-in boundary rows are written before the alive rows that read them.
0.9.0 decided four more: the statistics conventions, the Heston/SLV statistics (now refused
unless the surrogate is requested), the KO-reset statistics from QUAD, and the knocked-out
`price_with_events` distribution. 0.10.0 decided four more: a knock-in's bump context
freezes one layout per part, a single-step local-vol layout keeps its per-step sets, the
American's expired Greeks carry the exercise payment discount, and the double barrier's
knocked-out Greeks equal its price.)

- The life-surface pricer's node Greeks use its own three-point stencil.
- American Greeks skip the valuation-date exercise max that `price` applies.
- KO-reset never composes its pre-event valuation-date readout: price and Greeks
  interpolate the post-event column 0.
- Interior KO cash is discounted from `t_obs` on the snowball's schedule path but from
  `total_tau − tau_remaining` on the inline paths; interior projections use `layout.x` on
  the schedule path and `log(s)` inline.
- `cache_enabled=False` differs from `True` at round-off: a cached run shares one
  factorization across intervals whose `dt` differ by an ULP (the factorization key
  rounds `dt` to 12 places, as every banked number was computed; θ is exact).
