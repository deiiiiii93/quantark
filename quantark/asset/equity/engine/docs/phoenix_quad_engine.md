# Phoenix Quadrature Engine

## 1. Model overview

`PhoenixQuadEngine` extends the two-regime Snowball FFT-quadrature recursion
with coupon state transitions, including optional memory coupons. It supports
discrete KO schedules and discrete or continuous KI monitoring.

The common diffusion formulation, assumptions, and KI monitoring modes are
documented in `snowball_quad_engine.md`.

## 2. Observation event

At a Phoenix observation, the post-event value is piecewise in log spot:

1. coupon missed and contract survives;
2. coupon paid and contract survives;
3. KO settlement, with the current coupon included only when its condition is
   met;
4. optional discrete-KI transition into the already-knocked-in continuation.

KO has precedence over a simultaneous KI unless
`disable_ko_after_ki=True`, in which case a simultaneous KI enters the
already-KI continuation where future KO is disabled.

With `QuadParams.event_projection="cell_average"` (default), all coincident
coupon, KO, and discrete-KI thresholds are applied as one conservative
piecewise projection. The complete branch function is integrated over the
dual cell crossed by each threshold. This avoids sequential double averaging
and reduces sensitivity to where a barrier falls between uniform FFT nodes.

`event_projection="nodal"` retains the legacy smoothing/hard-mask behavior for
historical reproduction.

## 3. Numerical considerations

- The diffusion grid remains uniform in log spot for FFT convolution.
- Only one level can be globally node-aligned; cell-average projection handles
  every other scheduled threshold at its exact sub-cell location.
- Barriers outside the numerical domain produce an all-survive or all-breach
  projection without expanding the grid.
- Coincident thresholds are deduplicated before projection.
- Projection acts on every memory-coupon state independently.
- FFT convolution defaults to composite trapezoid weights because they remain
  phase-stable after discontinuous observations. The legacy Simpson rule is
  available through `QuadParams.integration_rule="simpson"`.
- KO levels outside the observation-time model envelope are suppressed as
  disabled sentinels without removing the Phoenix coupon date.
- Optional nested-grid convergence control returns the accepted finest value
  surface and fails closed at its configured point cap.

## 4. References

- Huang & Luo (2015), *A Simple and Efficient Numerical Method for Pricing
  Discretely Monitored Early-Exercise Options*.
- `snowball_quad_engine.md`.
