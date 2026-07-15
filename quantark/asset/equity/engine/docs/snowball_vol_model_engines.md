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
