# The 1D PDE core (0.7.0)

Every shared-grid 1D PDE route prices through one path: European, American, the
barrier family (single, double, one-touch, double one-touch, knock-in by composition),
Snowball, Phoenix, KO-reset, and the local-vol Snowball and Phoenix. The pieces live in
`quantark/asset/equity/engine/pde/core/`. This page explains how they fit together,
how to plug in a custom component, and what the core keeps from 0.6.0 on purpose.
The design is `docs/superpowers/specs/2026-09-30-pde-1d-one-core-design.md`.

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
| Event statistics | one group of `2·n_cols` indicator columns | banded only |

| Policy | Boundary | θ | After events |
|---|---|---|---|
| value | the adapter's Dirichlet values, set before the solve | the schedule; θ=1 on zero-diffusion steps for the base family, American, Snowball, KO-reset; no override for Phoenix | none |
| statistics-legacy | the previous row carried into the right-hand side | the schedule, no zero-diffusion override | zero-slope copy (`v[0] = v[1]`, `v[-1] = v[-2]`) |

Event statistics are a column group of the price march: `calculate_event_stats` marches
the value and statistics groups together and does not re-price. The statistics group
keeps the thirteen 0.6.0 conventions listed in the spec (section 6) bit for bit; moving
it onto the value policy is sub-project 3's "Level 2".

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

`CrankNicolson` carries the scheme's four settings: `theta` (0.5), `event_theta` (1.0),
`terminal_damping_steps` and `event_damping_steps` (`None` = the route's defaults: 1 and 2
on the core routes, 2 and 2 on DCN; 0 disables). Its schedule is 0.6.0's rule: `theta`
everywhere, `event_theta` on the damped steps before each interior event node, 1.0 on the
terminal steps, terminal wins on overlap.

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

## The evidence record

Certificates hash `{declared params, engine, grid: <resolved record>}`. The candidate
builders construct the engine they run from the mapping they record
(`declared_pde_params` → `PDEParams.from_config`), and the record is
`resolved_grid_evidence(params)`. That is the nine resolved `GridConfig` fields plus the
built-in stepper's two resolved damping counts, under their 0.6.0 keys
(`terminal_damping_steps`, `event_damping_steps`). The record is the numerical grid plus
the scheme's damping, not a serialization of `GridConfig`, and it is byte-identical for
every banked configuration.

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

The core reproduces these 0.6.0 behaviours bit for bit. Each is decided in sub-project 3,
under re-certification, not here:

- Snowball interior discrete events cell-average through the schedule even under
  `event_projection=NODAL`; terminal, valuation-date and statistics events go nodal.
- The thirteen statistics conventions (spec section 6).
- Phoenix's value sweep has no zero-diffusion θ override; the other families do.
- The alive column's lower boundary under continuous KI is zero before the coupling.
- Heston/SLV statistics march in 1D on implied vol with the 2D price as residual.
- KO-reset statistics come from QUAD, with the price difference in the maturity leg.
- The life-surface pricer's node Greeks use its own three-point stencil.
- On a product knocked out at valuation, `price_with_events` returns the trivial
  distribution while `calculate_event_stats` marches and reports KO probability one.
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
