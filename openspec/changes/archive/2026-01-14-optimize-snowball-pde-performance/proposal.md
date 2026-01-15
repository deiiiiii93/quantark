# Change: Optimize Snowball PDE Solver Performance

## Why
The current Snowball PDE solver is accurate but slower than Monte Carlo and quadrature for typical grid sizes. We want to reduce solve time while keeping pricing accuracy within existing tolerances.

## What Changes
- Profile the Snowball PDE solver to identify dominant costs (matrix assembly, factorization, RHS solves, boundary updates).
- Improve solver reuse and reduce per-step overhead (matrix caching, batched solves for V0/V1, minimize rebuilds).
- Add a banded/tridiagonal solve path when the spatial operator is tri-diagonal.
- Add a small benchmark script and documentation notes on performance tradeoffs.

## Impact
- Affected specs: `snowball-pde-engine`
- Affected code: `asset/equity/engine/pde/snowball_pde_solver.py`, `asset/equity/engine/pde/base_pde_solver.py`, benchmark scripts/docs
