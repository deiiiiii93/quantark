## 1. Profiling
- [x] Capture baseline timings (matrix build, LU, RHS, solve, boundary updates) for 200×200 and 400×400 grids.

## 2. Solver Optimization
- [x] Reduce matrix rebuilds and improve LU cache reuse across time steps.
- [x] Batch V0/V1 solves when possible.
- [x] Add banded/tridiagonal solver path for the spatial operator.

## 3. Memory/Vectorization
- [x] Preallocate arrays and vectorize boundary/KI/KO updates.

## 4. Benchmark & Validation
- [x] Add a benchmark script documenting before/after performance.
- [x] Compare pricing accuracy against MC/Quad for representative cases.
