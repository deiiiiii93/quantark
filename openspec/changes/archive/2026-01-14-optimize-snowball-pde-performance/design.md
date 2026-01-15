# Design: Snowball PDE Performance Optimization

## Overview
Target performance improvements for the two-surface Crank–Nicolson solver by reducing per-step overhead and leveraging the tridiagonal structure of the spatial operator.

## Approach
1. **Profiling**: instrument timing for matrix assembly, factorization, RHS builds, and solves.
2. **Reuse**: keep LU factorizations for repeated (dt, theta) pairs; avoid clearing cache unless grids change.
3. **Batch solves**: solve V0/V1 RHS in a single call when feasible (stacked RHS).
4. **Banded solver**: use a tridiagonal/banded solver for the 1D operator instead of generic sparse LU when applicable.
5. **Vectorization**: reduce Python loops for boundary/KI/KO updates and preallocate arrays.

## Success Criteria
- Demonstrate measurable speedup on standard snowball benchmarks (e.g., 200×200 and 400×400 grids) without material accuracy regression vs MC/Quad.

## Non-goals
- Model changes or payoff logic changes.
- API-breaking changes.
