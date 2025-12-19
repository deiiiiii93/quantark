# Design: Trinomial Volatility Schemes

## Overview
Add explicit volatility scheme selection to `ConvertibleBondTrinomialEngine` with three supported schemes:
1. Constant-volatility CRR-style trinomial (backward-compatible).
2. Fixed-`dx` log-price trinomial with time-dependent probabilities (recombining).
3. Variable-`dx` log-price trinomial with per-step `dx` and re-gridding/interpolation.

## Scheme Details

### Constant-volatility (CRR-style)
- Use existing `u/d` based on a fixed volatility (currently max vol).
- Document that this scheme does not support term-structure volatility.

### Fixed-`dx` log-price scheme
- Set `dx` once using a stability criterion (e.g., based on max forward vol and `sqrt(dt)` with a stretch factor).
- Build the log-price grid with fixed `dx` and use time-dependent probabilities per step that match the step-local mean and variance.
- Keep the tree recombining with indices aligned at each step.

### Variable-`dx` log-price scheme with re-gridding
- Compute per-step `dx_i` using step-local vol (e.g., `dx_i = sigma_i * sqrt(3 * dt)`).
- Use re-gridding to maintain recombination:
  - Maintain a fixed log-price grid indexing per time step.
  - When `dx` changes, map values between grids using linear interpolation in log-price.
- Ensure probability constraints and normalize when needed.

## Probability Matching
For log-price steps `{-dx, 0, +dx}` at time step `i`, match the first two moments under the risk-neutral measure:
- Mean of log-return `m1 = (r_i - q_i - 0.5 * sigma_i^2) * dt`
- Variance `s2 = sigma_i^2 * dt`
Solve for `p_u`, `p_m`, `p_d` and enforce `[0,1]` with re-normalization or smaller `dt` if invalid.

## Validation
- If the constant-vol scheme is chosen and a term-structure surface is detected, warn or raise (configurable) that term structure is not applied.
- Ensure probabilities are valid and raise `NumericalError` if not.
