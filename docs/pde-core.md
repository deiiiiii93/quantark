# The 1D PDE core (0.9.0)

Every shared-grid 1D PDE route prices through one path: European, American, the
barrier family (single, double, one-touch, double one-touch, knock-in by composition),
Snowball, Phoenix, KO-reset, and the local-vol Snowball and Phoenix. The pieces live in
`quantark/asset/equity/engine/pde/core/`. This page explains how they fit together,
how to plug in a custom component, and what the core keeps from 0.6.0 on purpose.
The design is `docs/superpowers/specs/2026-09-30-pde-1d-one-core-design.md`; 0.8.0's
numerical policy (damping form, projection, grid day, cap, far-field rows) is
`docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-design.md`; 0.9.0's event
statistics are `docs/superpowers/specs/2026-10-03-pde-1d-event-statistics-design.md`.

DCN, the local-vol vanilla and barrier solvers, and the 2D Heston/SLV solvers are not
on the core yet (sub-project 3). They take their damping counts and weights from
`PDEParams.stepper` and reject custom components.

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

A knock-in barrier prepares a `CompositeProblem`: the vanilla and the knock-out, marched
independently and composed as `vanilla − knock-out` for the price and for each Greek.

## Column groups and policies

| Family | Value groups | Backend |
|---|---|---|
| European, knock-out barriers, touches, American | one column | sparse LU |
| Knock-in barriers | two prepared problems (vanilla, knock-out); knocked in: the vanilla alone | sparse LU each |
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
`BarrierAdapter`, `SnowballAdapter` (also the local-vol snowball, through a mixin that
samples a local-vol row per step), `PhoenixAdapter`, `KOResetAdapter`. An adapter
supplies the solve state, the grid request, terminal fills, boundary values, the ordered
events, captures, the column groups for a request, and the readout branch. The adapters
delegate to the product helpers the solvers kept (the 2D solvers inherit them), so
per-solve state still lives on the solver. The march itself holds none.

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
continuous knock-in, full-step (1, 2) for one with it; DCN, the local-vol kernels and the
2D solvers have their own rows. Its schedule is 0.6.0's rule: `theta` everywhere,
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

**Where custom components run.** On the core routes only. The local-vol vanilla and
barrier solvers and DCN reject a stepper that is not a `CrankNicolson`, and a non-built-in
operator or readout, with "no core seam until sub-project 3". An operator must list the
route's `SampleKind` in `supports`: `SCALAR` on the Black-Scholes routes, `NODAL` on the
local-vol ones. A custom operator's step sets are marched as assembled. The built-in's
flat-exact replacement applies to the built-in only.

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

Continuous monitoring (the barrier is the domain edge) and the valuation-date observation
(a terminal outcome) stay nodal in every mode. `explain`'s `events` section lists each
observation's records and treatment (`BaseFamilyEvents`).

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
would. The routes outside the core have no `explain`; the 2D solvers raise.

## Kept on purpose

The core reproduces these 0.6.0 behaviours bit for bit. Each is decided in a later
sub-project, under re-certification, not here. (0.8.0 decided three former entries: Snowball
`NODAL` masks interior events nodally, Phoenix takes θ = 1 on zero-diffusion steps, and under
continuous KI the knocked-in boundary rows are written before the alive rows that read them.
0.9.0 decided four more: the statistics conventions, the Heston/SLV statistics (now refused
unless the surrogate is requested), the KO-reset statistics from QUAD, and the knocked-out
`price_with_events` distribution.)

- The life-surface pricer's node Greeks use its own three-point stencil.
- A knock-in barrier's bump context freezes the knock-out layout but rebinds the vanilla
  on every bumped solve.
- A single-step local-vol layout marches on the scalar Black-Scholes stencil (one unique
  set selects the flat stencil). No preset reaches it.
- American Greeks skip the valuation-date exercise max that `price` applies, and the
  expired-Greeks short-circuit omits the settlement delay factor.
- The double barrier's knocked-out Greeks return the undiscounted rebate; `price`
  discounts it.
- KO-reset never composes its pre-event valuation-date readout: price and Greeks
  interpolate the post-event column 0.
- Interior KO cash is discounted from `t_obs` on the snowball's schedule path but from
  `total_tau − tau_remaining` on the inline paths; interior projections use `layout.x` on
  the schedule path and `log(s)` inline.
- `cache_enabled=False` differs from `True` at round-off: a cached run shares one
  factorization across intervals whose `dt` differ by an ULP (the factorization key
  rounds `dt` to 12 places, as every banked number was computed; θ is exact).
