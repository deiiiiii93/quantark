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

## Limitations

- Heston/SLV Snowball PDE event stats are not exposed in this mount.
- Heston/SLV PDE uses the same cumulative-to-maturity rate and carry convention as
  the existing Heston/SLV Barrier PDE wrappers.
- Heston/SLV PDE monitoring dates are snapped to the uniform ADI grid, following the
  current Barrier Heston/SLV PDE convention.
