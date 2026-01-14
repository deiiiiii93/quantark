# Change: Refactor Snowball Quad Engine (Regime-Switching Quadrature)

## Why
The current snowball quadrature implementation decomposes the payoff into multiple primitives, which obscures the product logic and adds unnecessary overhead. We need a direct two-state quadrature recursion that models KO/KI transitions explicitly and supports continuous KI via Brownian-bridge transitions.

## What Changes
- Introduce a reusable quadrature math/grid utility extracted from `quad_core.py` for shared convolution/integration/interpolation primitives.
- Refactor `QuadratureCore` (and related quad engines) to use the shared quadrature math utility.
- Replace the snowball quadrature implementation with a direct two-state (V_in/V_out) backward recursion using a single quadrature pass.
- Add continuous KI handling via Brownian-bridge state-transition probabilities inside the recursion.
- Update documentation and add tests covering discrete KO + discrete/continuous KI.

## Impact
- Affected specs: `equity-quad-engine`, `snowball-quad-engine` (new)
- Affected code: `asset/equity/engine/quad/quad_core.py`, new `asset/equity/engine/quad/quad_math.py` (or similar), `asset/equity/engine/quad/snowball_quad_engine.py`, related exports/tests/docs
- No API breaking changes expected for existing quad engines; internal refactor only.
