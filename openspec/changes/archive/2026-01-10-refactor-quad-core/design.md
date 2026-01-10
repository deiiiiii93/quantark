# Design: Quadrature Core + Adapter Architecture

## Goals
- Encode the quadrature recursion (Eq. 3.5) in a reusable core that is independent of any product.
- Move product-specific logic into adapters that translate product parameters into boundary levels and linear payoff coefficients.
- Use actual observation dates for time stepping to match the paper’s piecewise-constant assumptions.

## Core API (proposed)
A single module (e.g., `asset/equity/engine/quad/quad_core.py`) that accepts:
- `observation_times`: strictly increasing array `t_1..t_M` (maturity is `t_M`).
- `K_minus[m]`, `K_plus[m]`: boundary levels at each observation `m` (0 or +inf allowed).
- `a_minus[m]`, `b_minus[m]`: payoff coefficients for `S <= K_minus[m]`.
- `a_plus[m]`, `b_plus[m]`: payoff coefficients for `S >= K_plus[m]`.
- `a_M`, `b_M`: terminal payoff coefficients for `K_minus[M] < S < K_plus[M]`.
- Market inputs per interval (`r_m`, `q_m`, `sigma_m`), or an accessor that provides these for each `m`.

The core computes `V_0(S0)` via the quadrature recursion using the same uniform log-grid/truncation strategy as today, but stepping over observation intervals.

## Adapter Responsibilities
Adapters map product definitions to the core inputs:

### Barrier Adapter
- Produces `K_minus/plus` per observation from barrier schedule.
- Encodes knock-out cash rebates as `a±=0`, `b±=rebate` at the observation date.
- Encodes terminal vanilla payoff inside the band via `a_M`, `b_M`.

### One-Touch Adapter
- Encodes one-touch as a pure boundary payoff (`a±=0`, `b±=rebate`) at each observation.
- Encodes pay-at-hit vs pay-at-expiry by discounting `b±` from settlement time to observation time.
- Encodes no-touch via parity against the one-touch valuation (adapter-level logic only).

## Time Stepping
- The core uses `Δt_m = t_m - t_{m-1}` from observation dates directly.
- This aligns with Section 2 and Proposition 3.4 assumptions (piecewise-constant parameters).

## Compatibility & Migration
- Existing `BarrierQuadEngine` and `OneTouchQuadEngine` are refactored to call adapters + core.
- The solver currently embedded in `barrier_quad_engine.py` becomes the new core module.
- Numerical behavior should be consistent; small differences are possible due to precise timing alignment.
