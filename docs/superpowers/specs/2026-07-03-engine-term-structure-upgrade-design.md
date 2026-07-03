# Engine Term-Structure Upgrade (r, q, vol) — Design

**Date:** 2026-07-03
**Status:** design draft
**Area:** `quantark/asset/equity/engine/`, `quantark/param/`, `quantark/priceenv/`
**Depends on:** none
**Depended on by:** `2026-07-03-equity-implied-futures-carry-risk-design.md`
  (that spec's futures-tenor delta buckets are degenerate until at least
  Phase 1 of this spec lands)

## Summary

Make every deterministic equity pricing engine (MC, PDE, QUAD) consume
term-structured risk-free rates, dividend/carry yields, and volatilities,
instead of sampling three scalars at product maturity. Enhance the existing
engines in place through one shared term-sampling layer; do **not** create
parallel term-structure engine variants.

Two governing principles:

1. **Identity on flat inputs.** The forward rate of a flat curve is the flat
   rate; the forward carry of a flat yield is the flat yield; the step vol of
   a flat surface is the flat vol. Every upgraded engine must produce results
   identical to today's on flat inputs. Backward compatibility is guaranteed
   by this mathematical identity — enforced by the existing test suite passing
   unchanged — not by feature flags.
2. **Term-in-time only, no smile.** Vol is sampled at each engine's existing
   reference strike and converted to per-step vols via total variance.
   Smile-consistent dynamics remain the exclusive job of the Local Vol / SLV /
   Heston engines. This upgrade must never be described or used as a
   local-vol substitute.

## Motivation

All constant-vol equity engines currently collapse the market term structure
to three scalars at product maturity:

- every constant-vol MC engine: `q = pricing_env.get_div_yield(T)` once
  (e.g. `snowball_mc_engine.py:212`, `phoenix_mc_engine.py:141`);
- every BSM PDE solver: `r`, `q`, `sigma` extracted once at the top of
  `_solve()` with `tau` = full maturity (e.g. `snowball_pde_solver.py:182-185`)
  and used as constant PDE coefficients;
- every QUAD engine: `get_rate(maturity)` / `get_div_yield(maturity)` /
  `get_vol(strike, maturity)` as constants of the transition density
  (e.g. `snowball_quad_engine.py:119-121`, `quad_adapters.py:216-218`).

Consequences:

- A `TermStructureDividendYield` (or term rate curve, or term vol) attached to
  a `PricingEnvironment` silently degrades to its value at one point. Bucketed
  carry/vol risk against these engines is degenerate: only the node(s)
  bracketing product maturity produce non-zero sensitivity.
- The implied-futures-carry feature (companion spec) cannot produce
  multi-tenor futures delta buckets for autocallables — its headline use case
  — because the entire carry curve collapses to `q(T_maturity)`.
- Path-dependent products (snowball KO observations, phoenix coupons, range
  accrual fixings) are priced with a drift that ignores the shape of the
  forward curve between valuation and maturity, so intermediate-date forwards
  implied by the simulation do not match market futures at those dates.

The model-based engines (Heston, Local Vol, SLV) already consume term
structures correctly through `forward_carry_on_grid`; this spec promotes that
architecture to all engines.

## Current state (verified inventory)

### Shared infrastructure that already exists

- `quantark/volmodels/curves.py:30` — `forward_rates_on_grid(rate_curve,
  t_grid)`: piecewise-constant forward rates per interval, DF-based and exact
  (`RateCurve.get_forward_rate`).
- `quantark/volmodels/curves.py:45` — `forward_carry_on_grid(zero_yield,
  t_grid)`: piecewise-constant forward carry from a zero-yield term structure,
  `(q(t1)·t1 − q(t0)·t0)/(t1 − t0)`.
- `PricingEnvironment.get_step_volatility(strike, t_start, t_end)`
  (`pricing_environment.py:96`): step vol from total-variance differencing.
- `TermStructureDividendYield` (`quantark/param/div/dividend_yield.py:68`):
  linear interpolation, flat extrapolation both sides.
- PDE solver family consolidated on `BasePDESolver` +
  `BackwardOperator` (`quantark/asset/equity/engine/pde/backward_operator.py`),
  which centralizes banded-system construction and caching.

### Engine inventory and disposition

| Family | Engines | Disposition |
|--------|---------|-------------|
| MC (constant-vol BSM) | euro, digital, asian, barrier, snowball, phoenix, single_sharkfin, double_sharkfin, accumulator, range_accrual, american (LSMC) | **Upgrade**: per-step drift/vol arrays |
| MC (model-based) | heston, heston_slv, local_vol | Already term-aware — no change |
| MC (SABR) | sabr_mc | **Upgrade drift only**: term r/q in the drift; SABR vol dynamics unchanged (constant SABR params are the model) |
| PDE (BSM) | european, american, barrier, double_barrier, one_touch, double_one_touch, snowball, phoenix, ko_reset_snowball | **Upgrade**: per-step coefficients via `BackwardOperator` |
| PDE (model-based) | local_vol (already term-aware), heston, heston_slv | **Upgrade heston/heston_slv carry**: currently scalar `carry=get_div_yield(T)` (`heston_pde_solver.py:59`); switch to the same forward-carry grid the Heston MC engine already uses |
| QUAD | european, snowball, phoenix, ko_reset_snowball, discrete_quad, quad_adapters | **Upgrade**: per-observation-interval forward `(r, q, σ)` |
| Analytical | black_scholes, digital, barrier, one_touch, double_barrier, asian, sharkfin, range_accrual, accumulator, deltaone | Europeans already exact given cumulative-to-T inputs — no change. Constant-parameter approximations (BS93/BAW American, closed-form barrier) keep sampling cumulative-to-T values as a **documented convention** |
| Tree | (none exist) | n/a |

## Component 1: shared term-sampling layer

### Placement

- Low-level grid samplers live in **`quantark/param/term_sampling.py`**:
  `forward_rates_on_grid` and `forward_carry_on_grid` move here from
  `volmodels/curves.py`; `step_vols_on_grid` and `discount_factors_on_grid`
  are new. The `param` layer defines `RateCurve` /
  `DividendYield` / `VolatilitySurface`, so it is the natural home and creates
  no import cycles. `quantark/volmodels/curves.py` re-exports the moved
  functions so existing model-based engines keep working unchanged.
- The environment-facing builder lives in
  **`quantark/priceenv/term_sampling.py`** (priceenv already depends on param).

### API

```python
@dataclass(frozen=True)
class TermCoefficients:
    """Per-interval forward market coefficients on a time grid.

    All arrays have length len(t_grid) - 1; entry i covers
    [t_grid[i], t_grid[i+1]].
    """

    t_grid: np.ndarray
    fwd_rates: np.ndarray        # drift component r
    fwd_carry: np.ndarray        # drift component q
    step_vols: np.ndarray        # per-interval vol (total-variance based)
    step_dfs: np.ndarray         # per-interval discount factors DF(t_i, t_i+1)

    @classmethod
    def from_env(
        cls,
        pricing_env: PricingEnvironment,
        t_grid: np.ndarray,
        ref_strike: float,
    ) -> "TermCoefficients":
        ...
```

Rules:

- `fwd_rates` from `forward_rates_on_grid` (DF-exact); `step_dfs` directly
  from the rate curve's discount factors, never re-derived as
  `exp(-r(T)·T)` ad hoc.
- `fwd_carry` from `forward_carry_on_grid(env.get_div_yield, t_grid)`.
- `step_vols` from total-variance differencing at `ref_strike`
  (the engine's existing vol reference strike — usually product strike).
- Flat inputs must produce arrays where every entry equals today's scalar —
  covered by a dedicated unit test.
- Engines consume `TermCoefficients`; **no engine re-derives forward
  quantities itself**.

## Component 2: signed carry (prerequisite behavior change)

Implied-from-futures carry is negative whenever futures trade in contango
(`q = r − ln(F/S)/T < 0` for F sufficiently above S). Current code forbids
this in four places. All four change:

1. `TermStructureDividendYield.__post_init__`
   (`dividend_yield.py:87-88`): replace `y < 0` rejection with a finiteness +
   magnitude check `|y| ≤ 1.0` (short-tenor implied carry legitimately
   spikes; ±100% is a fat sanity bound, not an economic statement).
2. `ContinuousDividendYield.__post_init__` (`dividend_yield.py:44-49`):
   allow `-0.20 ≤ q ≤ 0.20` (keep the existing 20% sanity magnitude,
   make it symmetric).
3. `ShiftedDividendYield` / `BucketedDividendYield`
   (`quantark/asset/equity/report/term_structure.py:99,110`): remove the
   `max(0.0, …)` clamps — a bumped-down carry below zero is a valid state,
   and the clamp silently corrupts bucketed rhoq near q = 0.
4. `GreeksCalculator._build_div_bumped_env` down-bump guard
   (`greeks_calculator.py:945`): remove the floor-at-zero adjustment; the
   down bump is always applied as requested.

Each change gets its own test, including a regression test that bucketed
rhoq at q = 0 is now symmetric in bump direction.

## Component 3: MC family

For each constant-vol engine, path generation changes from

```text
drift = (r - q - 0.5·σ²)·dt          # scalars
S_{k+1} = S_k · exp(drift + σ·√dt·Z)
```

to per-step arrays from `TermCoefficients` on the engine's existing
observation/step grid:

```text
drift_k = (fwd_r_k - fwd_q_k - 0.5·σ_k²)·dt_k
S_{k+1} = S_k · exp(drift_k + σ_k·√dt_k·Z_k)
```

- Discounting of cashflows at time `t` uses the rate curve's `DF(t)`
  (cumulative product of `step_dfs`), not `exp(-r·t)`.
- The loop shape, lifecycle-event handling, Sobol/Brownian-bridge/antithetic
  and other variance-reduction layers are indifferent to per-step scaling —
  no changes there.
- American LSMC: same path generator; the regression's discounting per
  exercise date switches to curve DFs.
- Common-random-number behavior (same seed ⇒ same normals) is unchanged,
  preserving finite-difference Greek quality.

## Component 4: PDE family

Changes concentrate in the shared infrastructure, not the nine solvers:

- `BackwardOperator` accepts per-segment `(r_k, q_k, σ_k)` instead of one
  constant triple. Its coefficient/factorization **cache keys on the segment
  values**: implied curves are piecewise with few nodes, so forward
  parameters are piecewise-constant and factorization reuse survives within
  segments. Coefficients rebuild only at segment boundaries, not every step.
- `BasePDESolver._solve` implementations replace the single
  `(r, q, sigma)` extraction with `TermCoefficients.from_env(env, t_vec,
  ref_strike)` and pass the per-step values through to the operator.
- Grid construction (`_build_grids`, spatial bounds sized by `σ√T`-type
  quantities) uses the **total** vol to maturity and total carry — the same
  numbers as today for flat inputs.
- Heston / Heston-SLV PDE: replace scalar `carry=get_div_yield(T)` with the
  forward-carry grid, matching what the Heston MC engine already does.
- Rannacher restarts, event application (KO/KI/coupon), BGK mode, and the
  event-distribution machinery are orthogonal to coefficient values and are
  untouched.
- **Performance gate:** flat-input pricing time within measurement noise of
  the current baseline on the standard snowball benchmark, protecting the
  event-distribution redesign gains. The tridiagonal solve is already O(n)
  per step; per-segment rebuilds add a bounded constant factor.

If threading per-step coefficients through `BackwardOperator` turns out to
require touching individual solvers' time loops beyond the coefficient
plumbing, that is a signal the solver-family consolidation is incomplete —
finish the consolidation as part of this work rather than special-casing.

## Component 5: QUAD family

QUAD propagates a lognormal transition density between consecutive
observation dates — structurally the cleanest fit:

- Each propagation leg `[t_i, t_{i+1}]` takes its own forward
  `(r_i, q_i, σ_i)` from `TermCoefficients` on the observation grid, entering
  the density mean/variance and the leg's discount factor.
- Changes concentrate in `quad_core.py` / `quad_math.py`; the product
  engines (`snowball_quad_engine`, `phoenix_quad_engine`,
  `ko_reset_snowball_quad_engine`, `quad_adapters`, `discrete_quad_engine`)
  swap scalar extraction for `TermCoefficients` and pass leg parameters down.
- The pre-existing KI-probability definitional gap (QUAD `ki_probability` =
  P(KI without prior KO) vs MC P(KI ever)) is explicitly **out of scope** and
  unaffected.

## Analytical engines: documented convention

- European closed forms are already exact under term structures **provided
  the inputs are cumulative-to-T quantities**: `get_vol(K, T)` is the Black
  vol to T, `get_div_yield(T)` the zero carry to T, `get_rate(T)` the zero
  rate to T. No code change; add a docstring note.
- Constant-parameter approximations (BS93/BAW American, closed-form barrier /
  one-touch / sharkfin formulas) mathematically assume constant coefficients.
  They continue to sample cumulative-to-T values. This is a documented
  convention, not a defect; users needing term-structure-consistent American /
  barrier prices use the upgraded PDE/MC/QUAD engines. Do **not** invent
  hybrid approximations.

## Testing

1. **Flat-input identity (regression).** The entire existing test suite
   passes unchanged. Additionally, a targeted test per family asserts
   bit-level (or ≤1e-14) equality between pre- and post-upgrade prices on
   flat inputs (pre-upgrade values captured as golden numbers).
2. **Exact European benchmark.** Under term `r(T)`, `q(T)`, `σ(T)`, the
   European call price is closed-form (Black-Scholes with integrated
   variance, integrated carry, and curve DF). Every family (MC within CI,
   PDE/QUAD within convergence tolerance) is validated against it on at
   least: upward-sloping, downward-sloping, and kinked term structures,
   including negative-carry segments.
3. **Forward reproduction.** With `q(T)` implied from synthetic futures
   marks, each engine's implied forward at every node `T_i` reprices
   `F(T_i)` exactly (MC: expectation of `S_{T_i}` within CI; PDE/QUAD: via a
   forward contract / linear payoff at `T_i`). This is the contract the
   implied-futures-carry feature depends on.
4. **Cross-family agreement.** MC vs PDE vs QUAD on the same term-structured
   snowball and phoenix within the existing cross-validation tolerances;
   plus a flat-in-strike Local Vol cross-check (constant-in-strike LV grid +
   term carry must agree with the upgraded BSM engines).
5. **Signed carry.** Curve construction, pricing, and rhoq bumping with
   negative q; bucketed rhoq symmetry at q = 0.
6. **Performance guardrail.** Flat-input snowball PDE benchmark within noise
   of the pre-upgrade baseline.

## Phasing (one spec, four gated phases)

| Phase | Content | Gate |
|-------|---------|------|
| 0 | `TermCoefficients` + samplers move/re-export; signed-carry changes; European term benchmark harness | sampler unit tests; flat-identity of samplers; signed-carry tests |
| 1 | MC family (11 BSM engines + sabr drift; LSMC discounting) | flat identity; tests 2–3; futures-carry spec unblocked |
| 2 | PDE family (`BackwardOperator` + 9 solvers + heston carry) | flat identity; tests 2–4; perf guardrail |
| 3 | QUAD family (`quad_core`/`quad_math` + 6 engines/adapters) | flat identity; tests 2–4 |

Each phase merges independently. The implied-futures-carry feature may start
after Phase 1.

## Non-goals

- No smile/skew dynamics in BSM engines — strike dependence of vol remains
  LV/SLV territory.
- No discrete cash dividends (proportional-yield world only).
- No FX engine changes (same pattern later; FX engines are out of scope).
- No new engine classes, engine enums, or parallel solver variants.
- No change to QUAD's KI-probability definition.
- No changes to the futures-carry feature itself (companion spec).

## Acceptance criteria

- All existing tests pass without modification.
- A `PricingEnvironment` carrying `TermStructureDividendYield`, a
  non-flat rate curve, and a term vol surface prices correctly (per the
  European benchmark) through every MC, PDE, and QUAD engine listed as
  "Upgrade" in the inventory.
- Engine forwards reprice the futures marks their carry curve was implied
  from, at every node, in every upgraded family.
- Bucketed rhoq / bucketed vega against upgraded engines produce non-zero
  sensitivity in the correct tenor buckets (no longer collapsing to the
  maturity node).
- Negative implied carry flows through construction, pricing, and Greek
  bumping without validation errors or clamping.
- Flat-input snowball PDE benchmark shows no measurable performance
  regression.
