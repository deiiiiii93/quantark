# Change: Add Monte Carlo engine for barrier options

## Why
Barrier options are supported as products and via PDE solvers, but there is no Monte Carlo engine for barrier pricing. Adding MC support provides a consistent pricing path for path-dependent barriers and a benchmark against other methods.

## What Changes
- Add a `BarrierOptionMCEngine` under `asset/equity/engine/mc/` to price `BarrierOption` via Monte Carlo simulation.
- Support continuous and discrete monitoring (ObservationSchedule or legacy observation dates) with rebates and knock-in/knock-out logic.
- Provide optional Brownian-bridge barrier crossing handling for continuous monitoring.
- Integrate two-level enum selection for MC method (PSEUDO/QUASI/RANDOMIZED_QUASI) consistent with existing MC engines.
- Add documentation and tests covering barrier types and monitoring modes.

## Impact
- Affected specs: `equity-mc-pricing`
- Affected code: `asset/equity/engine/mc/`, `asset/equity/engine/__init__.py`, `asset/equity/engine/mc/__init__.py`, tests
