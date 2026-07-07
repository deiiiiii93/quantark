# Phoenix Vol-Model Engines

## Scope

This note documents the Phoenix BSM, Local Vol, Heston, and Heston-SLV pricing
engine mounts. They mirror the Snowball vol-model engine family while preserving
Phoenix coupon, KO, KI, settlement, and event-order semantics.

## Model Interfaces

- `PhoenixMCEngine`: BSM/lognormal Phoenix MC engine.
- `PhoenixPDESolver`: BSM one-dimensional two-surface Phoenix PDE engine.
- `PhoenixQuadEngine`: BSM/lognormal quadrature Phoenix engine.
- `LocalVolPhoenixMCEngine`: Phoenix MC with log-Euler paths under a `LocalVolSurface`.
- `HestonPhoenixMCEngine`: Phoenix MC with Heston full-truncation log-Euler, QE, or QE-M paths.
- `QEPhoenixMCEngine`: standalone Phoenix MC facade fixed to Heston QE/QE-M.
- `HestonSLVPhoenixMCEngine`: Phoenix MC with Heston-SLV paths, using either a
  precomputed `LeverageSurface` or on-the-fly binned leverage.
- `HestonSLVQEPhoenixMCEngine`: standalone Phoenix MC with frozen-step leverage
  and QE variance stepping for Heston-SLV paths.
- `LocalVolPhoenixPDESolver`: one-dimensional Phoenix PDE with local-vol
  coefficients sampled on the spatial grid.
- `HestonPhoenixPDESolver`: non-memory Phoenix PDE on the shared Heston ADI core.
- `HestonSLVPhoenixPDESolver`: non-memory Phoenix PDE on the shared Heston-SLV ADI core.

## Numerical Semantics

The MC engines preserve `PhoenixMCEngine` payoff classification, coupon logic,
KO/KI observation handling, settlement timing, and result statistics by replacing
only the path generator. They support pseudo-random MC, Sobol QMC, and randomized
Sobol QMC through the existing `MonteCarloMethod` dispatch.

The local-vol PDE keeps the existing Phoenix vector-state recursion, including
memory coupons, and replaces the Black-Scholes scalar operator with per-step
local-vol tridiagonal coefficients.

The Heston and Heston-SLV PDE solvers reuse `HestonSLVADICore` with Phoenix
terminal surfaces and event hooks. Event order follows `PhoenixPDESolver`:
coupon jump first, KI transition second, and KO overwrite last. The 2D ADI
mount intentionally rejects memory coupons because memory requires a coupled
multi-state ADI recursion; use `PhoenixPDESolver` or a Phoenix MC engine for
memory coupon structures.

## Limitations

- Heston/SLV Phoenix PDE event stats are not exposed in this mount.
- Heston/SLV Phoenix PDE supports non-memory coupons only.
- Heston/SLV path-dependent PDE routes snap monitoring dates to the uniform ADI
  grid, following the current Snowball and Barrier Heston/SLV PDE convention.
- QUAD engines remain BSM/lognormal transition engines. Local Vol, Heston, and
  Heston-SLV QUAD requests must fail explicitly through the capability registry
  until a separate transition-kernel design is implemented.
