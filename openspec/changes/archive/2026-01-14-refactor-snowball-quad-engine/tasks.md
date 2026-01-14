## 1. Implementation
- [ ] Extract shared quadrature grid/math utilities (FFT convolution, Simpson weights, interpolation) into a reusable module.
- [ ] Refactor `QuadratureCore` to use the shared quadrature math utility without changing behavior.
- [ ] Update discrete quadrature engines/adapters to use the refactored core (no behavioral changes).
- [ ] Re-implement `SnowballQuadEngine` with direct regime-switching recursion (V_in/V_out) on the shared grid.
- [ ] Implement Brownian-bridge continuous KI transitions inside the recursion.
- [ ] Preserve discrete KO handling with per-date coupons/principal and settlement timing.
- [ ] Add/adjust documentation for the new snowball quadrature method.
- [ ] Add tests for discrete KO + discrete KI, and discrete KO + continuous KI (bridge).

## 2. Validation
- [ ] Compare new snowball quad prices against SnowballMCEngine for baseline cases.
- [ ] Run targeted pytest selections for snowball and quad engines.
