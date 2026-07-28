# Snowball Quadrature Engine

This document describes the direct regime-switching quadrature engine implemented in
`asset/equity/engine/quad/snowball_quad_engine.py`. The method follows the discrete
quadrature framework in `docs/quad/quad.md` and maintains two value functions during
the backward recursion.

## Method Summary

The engine evolves two states on the same log-price grid:

1. **V_in(S, t)**: Value given knock-in has already occurred.
2. **V_out(S, t)**: Value given knock-in has not occurred.

### Terminal Conditions

- `V_in(S, T)` = `get_maturity_payoff_v1(S)`
- `V_out(S, T)` = `get_maturity_payoff_v0(S)`

### Backward Recursion

For each interval `(t_{m-1}, t_m)`:

1. **Diffusion step**: Apply FFT-based convolution to propagate `V_in` and `V_out`.
2. **Continuous KI (if contractually enabled)**: Use Brownian-bridge hit probabilities to mix
   `V_out` into `V_in` inside the diffusion step.
3. **Observation updates**:
   - **KO**: At KO observation dates, both states jump to the KO payoff when the
     KO barrier is breached. If `disable_ko_after_ki=True`, only `V_out` is
     overwritten.
   - **Discrete KI**: At KI observation dates, `V_out` switches to `V_in` when the
     KI barrier is breached.
   - **KO precedence**: If KO and KI trigger together, KO overrides KI unless
     `disable_ko_after_ki=True`.

The price is `V_out(S0, t0)`.

### Discrete-event projection

`QuadParams.event_projection` controls how discrete KO/coupon/KI thresholds
are represented on the uniform log-price grid:

- `cell_average` (default) applies a conservative dual-cell projection. The
  cell crossed by a threshold is split at the exact sub-cell location and the
  piecewise-linear continuation branches are integrated over each part.
- `nodal` preserves the legacy smoothing/hard-mask path for reproducibility.

Phoenix observations resolve coincident coupon, KO, and discrete-KI decisions
as one piecewise projection, so a threshold cell is averaged once and event
precedence is preserved. This removes the leading barrier-to-grid phase error
when a step-down schedule contains many different KO levels.

### Convolution integration rule

Autocallable engines default to composite trapezoid weights for FFT
convolution. Although Simpson's rule has higher formal order for globally
smooth integrands, its alternating weights amplify node phase after each
discontinuous KO/coupon/KI event. Trapezoid weights give monotone, second-order
refinement once events use cell-average projection.

Set `QuadParams.integration_rule="simpson"` together with
`event_projection="nodal"` to reproduce the legacy recursion.

### Automatic convergence

`QuadParams.auto_converge=True` prices nested odd grids
`N, 2(N-1)+1, ...` until two consecutive PVs satisfy

`abs(error) <= convergence_abs_tol + convergence_rel_tol * max(abs(PV))`.

The accepted result is the finest directly computed PV and its matching value
surface; scalar Richardson extrapolation is deliberately not returned because
it would be inconsistent with spot-grid Greeks and backward exposure
surfaces. Failure to converge by `max_convergence_grid_points` raises
`NumericalError`.

### Disabled-barrier sentinels

When `filter_unreachable_barriers=True` (default), each KO level is checked at
its own observation time against the cumulative log-return mean plus/minus
`barrier_reach_stddevs` Gaussian standard deviations (`num_std_devs` when
unset). A trigger outside the relevant directional tail is treated as
disabled for KO and grid alignment. Its observation remains in the recursion,
so coupon and other coincident event semantics are retained. Set the flag to
`False` for literal legacy treatment of every positive level.

## Supported Configuration

- Discrete KO observation schedules
- Discrete or continuous KI monitoring (continuous uses Brownian-bridge transitions)
- Discrete KI treatment is governed by `QuadParams.ki_monitoring_mode`
  (`KnockInMonitoringMode`):
  - `EXACT_DISCRETE` (default): every KI observation date is priced exactly.
    The engine adaptively refines its internal spatial grid when KI intervals
    would otherwise be under-resolved (`min_diffusion_stddev_cells`, default
    2.5 cells per interval diffusion stddev, capped by
    `max_adaptive_grid_points` with a loud `NumericalError` past the cap).
    Accuracy-oriented; the default removes the material deep-carry quote
    error observed at coarser settings.
  - `BGK_APPROXIMATION` (opt-in): replaces a dense discrete KI schedule with
    continuous Brownian-bridge monitoring at a Broadie-Glasserman-Kou shifted
    barrier — typically several times faster because no grid refinement or
    per-date steps are needed, at the cost of a first-order residual bias
    (a few bp of PV at daily spacing, growing with spacing and drift).
    Only eligible schedules are accepted: approximately regular spacing
    (median-band dispersion test), a constant resolved barrier, full-horizon
    coverage (edge gaps within 3x the median spacing), a minimum schedule
    density (`bgk_min_ki_observations`, default 100), and stable volatility;
    otherwise the engine raises `ValidationError`. Valuation-time KI state
    always follows the contractual discrete semantics.
- Fixed rebates or call-style rebates (via product V0 payoff)
- Airbag features (via product V1 payoff)
- Optional `disable_ko_after_ki` logic (KO only applies before KI)
- Conservative cell-average event projection with an explicit legacy nodal
  opt-out

## References

- Huang & Luo (2015): *A Simple and Efficient Numerical Method for Pricing Discretely
  Monitored Early-Exercise Options*
- `docs/quad/quad.md`
