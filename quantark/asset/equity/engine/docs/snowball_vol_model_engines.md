# Snowball Vol-Model Engines

## Scope

This note documents the Snowball Local Vol, Heston, and Heston-SLV MC/PDE engine mounts.
They extend the existing Snowball autocallable payoff/event logic and mirror the
Barrier vol-model engine family.

## Model Interfaces

- `LocalVolSnowballMCEngine`: Snowball MC with log-Euler paths under a `LocalVolSurface`.
- `HestonSnowballMCEngine`: Snowball MC with Heston full-truncation log-Euler,
  QE, or QE-M paths.
- `QESnowballMCEngine`: standalone Snowball MC facade fixed to Heston QE/QE-M.
- `HestonSLVSnowballMCEngine`: Snowball MC with Heston-SLV paths, using either a
  precomputed `LeverageSurface` or on-the-fly binned leverage.
- `HestonSLVQESnowballMCEngine`: standalone opt-in Snowball MC with frozen-step
  leverage and QE variance stepping for Heston-SLV paths.
- `LocalVolSnowballPDESolver`: existing 1D two-surface Snowball PDE with local-vol
  coefficients sampled on the spatial grid.
- `HestonSnowballPDESolver`: two-surface Snowball PDE on the shared Heston ADI core.
- `HestonSLVSnowballPDESolver`: two-surface Snowball PDE on the shared Heston-SLV ADI core.

## Numerical Semantics

The MC engines preserve `SnowballMCEngine` payoff classification, KO/KI observation
handling, settlement timing, and event-stat definitions by replacing only the path
generator. They support pseudo-random MC, Sobol QMC, and randomized Sobol QMC through
the existing `MonteCarloMethod` dispatch. Heston QE/QE-M uses the same variance
transition and Sobol dimension layout as the existing Heston MC kernel.
`QESnowballMCEngine` exposes that Heston QE route directly for callers that want a
standalone QE Snowball engine.

`HestonSLVSnowballMCEngine` keeps the established full-truncation log-Euler SLV
process. `HestonSLVQESnowballMCEngine` is a separate opt-in approximation: it freezes
the leverage over each step, samples the CIR variance with QE, and applies the
leverage-scaled QE spot update. With a unit leverage surface and `eta=1`, it reduces
to the Heston QE Snowball path.

The local-vol PDE keeps the original Snowball two-surface recursion and replaces the
Black-Scholes scalar operator with per-step local-vol tridiagonal coefficients.

The Heston and Heston-SLV PDE solvers reuse `HestonSLVADICore` with custom terminal
surfaces, Snowball boundary hooks, and KO/KI step hooks. `V1` is solved first and
snapshotted at ADI time nodes; `V0` then applies KI transitions from the matching `V1`
surface after KO jumps, matching the existing Snowball PDE event order.

Heston and Heston-SLV PDE event stats reuse the existing one-dimensional Snowball PDE
event-attribution sweep and reconcile the reported `pv` and maturity residual to the
2D Heston/SLV PDE price. This exposes the standard event-stats API for cash-leg and
reporting workflows while leaving native 2D first-passage indicator surfaces as a
future enhancement.

For 2D Heston/SLV Snowball PDEs, the default log-spot grid focus is Snowball-aware.
`grid_focus="auto"` centers the concentrated grid around the KI barrier when the product
has KI. This avoids using a single KO-focused grid for principal-excluded structures
whose value is dominated by the KI/downside transition. Callers can set
`grid_focus="ko"`, `"ki"`, `"strike"`, or `"spot"` explicitly for diagnostics or legacy
comparisons. `pin_critical_spots=True` additionally pins KO, KI, strike, spot, initial
price, and optional payoff/airbag levels as exact grid nodes; this is opt-in because
forcing far-away nodes into a single-center grid can be less stable on coarse meshes.

On the variance axis, concentrated Heston and Heston-SLV Snowball grids default to
`variance_grid_mode="auto"`. Ordinary and low-Feller configurations resolve to
`v_grid_power=2.5`, i.e. `v_i = V_max * (i / (n_v - 1))**2.5`, preserving the
near-zero resolution used by the Heston DCN PDE. A sigma-collapse/deterministic-
variance configuration instead resolves to `path_focused`: the mesh concentrates on
the CIR state path and represents both theta and v0 exactly. Explicit
`variance_grid_mode="legacy"|"power"|"path_focused"` remains available for
diagnostics. `v_grid_power=0.0` is the backwards-compatible legacy opt-out and an
explicit positive value is a power-grid request. Uniform grids resolve to legacy;
power/path-focused geometry on a uniform grid is rejected.

The V generator defaults to `v_drift_scheme="adaptive_upwind"`. It retains the
second-order centered drift on rows whose lower and upper generator coefficients are
both non-negative. A convection-dominated row switches only its drift term to the
directionally correct donor-cell stencil, making `I - theta*dt*A_v` an M-matrix in
the V direction. `_A2` and `_tri_V` consume the same cached coefficient arrays.
`variance_operator_diagnostics()` exposes centered violations, fallback rows,
off-diagonal minima, local Peclet numbers, grid mode, and theta/v0 pins. The legacy
`"centered"` scheme is retained as a diagnostic control, not the production default.
The degenerate v=0 PDE boundary and existing event Rannacher policy are unchanged.

`v_drift_scheme="semi_lagrangian"` (opt-in, spec WS-C) exists because that
donor-cell fallback is only **first-order** on the variance axis, and the
2026-08-10 attribution probe traced the entire `sigma_collapse` delta bias
(−0.112 contracts, 11σ) to exactly that error. Under this scheme the generator
keeps diffusion only — unconditionally an M-matrix, local Peclet identically
zero, no row can need a fallback — and the drift is transported along the exact
CIR characteristics `v_foot = theta + (v - theta)·exp(-kappa·dt)` by four-point
Lagrange interpolation clipped into its bracketing linear envelope, so the
transport cannot manufacture a new extremum at the KI/KO kinks. Mean reversion
contracts toward theta, so feet are always interior and no inflow boundary is
needed. Each step is Strang-split (advect `dt/2`, drift-free ADI step, advect
`dt/2`, re-impose boundaries), and the degenerate v=0 row becomes an implicit
identity because advection owns its drift. Measured convergence in `n_v` on a
one-year Heston put at the sigma-collapse parameters: successive-refinement
ratios of **2.03 / 1.98** for `adaptive_upwind` (first-order) against
**113 / 153** for `semi_lagrangian`, with both schemes agreeing at fine `n_v`
— the transport changes the discretization, not the PDE. `n_v=60` under this
scheme lands closer to the continuum than `n_v=120` under upwind.
Advection observability lives under the `"advection"` key of
`variance_operator_diagnostics()` (feet interior, cells traversed, max foot
displacement). Defaults are untouched: every non-`semi_lagrangian` path is
bitwise identical to before the scheme was added.

Greek estimators are model-specific but share one market-sensitivity contract.
Local Vol reads delta and gamma from its native one-solve PDE surface. The 2D Heston
and Heston-SLV solvers use the default central spot bump; `BaseEngine.calculate_greeks`
now resolves `create_bump_context` once and prices the base/up/down states through that
context, so every finite-difference price uses the base-market spatial layout. Model
parameters, Local Vol surfaces, and SLV leverage surfaces remain frozen by the engine
factory during these spot bumps.

## Greek certification

`example/mo_volmodels/16_adi_greek_certification.py` is the admission harness for
the central-bump Heston/Heston-SLV Snowball Greeks. It is independent of calibration
history and emits JSON, Markdown, and a small decision artifact. Its evidence stack
is:

1. deterministic reductions (Heston vanilla vs semi-analytical, constant-variance
   Snowball vs 1-D BSM PDE and QUAD, unit-leverage SLV vs Heston, and deterministic
   variance vs a time-dependent 1-D local-vol PDE);
2. separate `n_x`, `n_v`, and `n_t` ladders plus a 2%/1%/0.5%/0.25% spot-bump
   ladder—never a pooled refinement that can hide compensating errors;
3. ordinary, decayed, near-KO, near-KI, low-Feller, sigma-collapse, and near-expiry
   regimes;
4. QE-M RQMC repricings at `S-h`, `S`, and `S+h` on the same scrambled Sobol batch,
   with delta/gamma formed before estimating the outer-batch standard error; Heston
   integrates its independent terminal spot factor exactly, while SLV uses an
   unbiased path-frozen-leverage conditional control, antithetic terminal-factor
   strata, and midpoint Brownian-bridge strata. The QE variance and branch streams
   are also Brownian-bridge ordered without changing their marginal laws;
5. PASS/FAIL/INCONCLUSIVE equivalence intervals containing Student-t reference
   uncertainty, a paired coupled-QE substep-bias upper bound, and the sum of the
   separate deterministic `n_x`/`n_v`/`n_t` envelopes. The reference and substep
   components each use 97.5% coverage (at least 95% simultaneous by Bonferroni).

Delta is judged in IM hedge contracts (0.5 per-cell and 0.1 mean-signed-bias
bounds). Gamma is reported as the change in hedge contracts caused by a 1% spot
move. Barrier-adjacent values are explicitly finite-bump hedge exposures, not a
claim that a classical pointwise derivative exists. The smaller-bump ladder diagnoses
that semantic choice and is not counted as PDE discretization uncertainty. A quick
run is plumbing-only. Production cells are atomically checkpointed and `--resume`
rejects any source/configuration hash mismatch. A production route is emitted only
when all four anchors, all seven regimes, monotone V coefficients, contracting PDE
axes, both cell Greeks, and the mean signed delta-bias gate pass; otherwise the route
is `excluded_greek_unresolved`, never a noisy daily-MC Greek fallback.
The compact route artifact is self-hashed and carries its complete run
configuration and numerical-runtime identity. Stage 12 recomputes the live
implementation fingerprint before using it, so a post-certification source or
runtime change invalidates admission rather than inheriting an old verdict.
The Stage-11 route records the same automatic variance grid, adaptive-upwind drift,
degenerate boundary, concentrated spot grid, and eight-steps-per-dense-KI-tick
controls explicitly. Stage 12 rejects older gate artifacts whose fixed
`v_grid_power=2.5` would override the certified sigma-collapse grid policy.

## Limitations

- Heston/SLV Snowball PDE event stats use delegated one-dimensional event attribution
  with 2D PDE PV reconciliation; they are not native 2D ADI indicator-surface
  probabilities.
- Heston/SLV path-dependent PDE routes consume a `TermMarketContext` on the ADI
  grid for S-direction drift, reaction, boundary values, and maturity-paid
  rebate discounting.
- Heston/SLV PDE monitoring dates are snapped to the uniform ADI grid, following the
  current Barrier Heston/SLV PDE convention.
- The 2D ADI grid still has one smooth concentration center. Exact multi-level pinning
  is diagnostic-only by default; production cross-checks should still use spatial
  convergence gates for long-dated or principal-excluded autocallables.

## Term-Structured Market Inputs and Volatility Semantics

The vol-model Snowball engines consume deterministic rates and carry on the
engine time grid. Local Vol MC/PDE routes use per-step `r/q` plus a
`LocalVolSurface`. Heston and Heston-SLV MC routes use per-step `r/q` in their
path simulation. Heston and Heston-SLV path-dependent PDE routes pass a
`TermMarketContext` into `HestonSLVADICore`, so S-direction drift, reaction,
boundary values, and maturity-paid rebates are discounted from the curve rather
than from scalar maturity rates.

Market implied-volatility surfaces are not interpreted as direct scalar vol
paths for Heston or SLV pricing. Heston engines price from `HestonParams`.
Heston-SLV engines price from `HestonParams` plus a `LeverageSurface` and local
volatility artifact. Market IV surfaces calibrate or rebuild those artifacts
and belong to calibration and structured vol-risk APIs, not to legacy scalar
`calculate_greeks()` vega.

QUAD engines remain BSM/lognormal transition engines. Local Vol, Heston, and
Heston-SLV QUAD requests must fail explicitly through the capability registry
until a separate transition-kernel design is implemented.
