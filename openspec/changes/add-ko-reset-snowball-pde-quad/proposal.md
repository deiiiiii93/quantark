# Change: Add PDE and quadrature engines for KO-reset snowball options

## Why
KO-reset snowball options currently rely on Monte Carlo only. PDE and quadrature engines are needed for deterministic pricing and consistency with existing Snowball engine patterns.

## What Changes
- Add `KOResetSnowballPDESolver` (PDE) and `KOResetSnowballQuadEngine` (quadrature) for `KnockOutResetSnowballOption`.
- Follow the existing Snowball engine patterns (two-surface V0/V1 recursion).
- Integrate the PDE solver with `PDEEngine` dispatch.
- Document solver behavior and add tests.
- Limit PDE/quad support to ABSOLUTE post-KO schedules; REBASED mode falls back to validation error.

## Impact
- Affected specs: `ko-reset-snowball-pde-engine`, `ko-reset-snowball-quad-engine`
- Affected code: `asset/equity/engine/pde/`, `asset/equity/engine/quad/`, `asset/equity/engine/pde_engine.py`, `asset/equity/engine/__init__.py`, tests, docs
