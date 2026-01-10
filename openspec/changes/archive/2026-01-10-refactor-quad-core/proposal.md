# Change: Refactor quadrature core into product-agnostic solver + adapters

## Why
The current quadrature implementation embeds product-specific logic inside a shared solver, which makes the codebase harder to reason about and obscures the true scope of the quadrature method described in `docs/quad/quad.md`. This refactor aligns the implementation with Proposition 3.4 (Eq. 3.5) and isolates product logic into clear adapter layers.

## What Changes
- Introduce a product-agnostic quadrature core that implements the recursion in Eq. 3.5 using boundary levels and linear payoff coefficients per observation date.
- Add adapter layers that translate each product (barrier, one-touch, etc.) into the core inputs (`K^-`, `K^+`, `a^-`, `b^-`, `a^+`, `b^+`, terminal `a_M`, `b_M`).
- Align time stepping to actual observation dates (piecewise-constant parameters) instead of implicit uniform grids in the core.
- Update quad engines to rely on adapters + core, removing any direct factor manipulation from engines.

## Impact
- Affected specs: `equity-quad-engine` (new)
- Affected code: `asset/equity/engine/quad/*`, related demos/tests
- Risk: minor pricing differences due to time-step alignment; will add targeted regression tests and update demos
