# Change: Feature-aware default PDE grid generation

## Why
Up-and-out (and other barrier) PDE solves are a known difficult case when the barrier is near spot, especially with discrete monitoring. A uniform grid and generic time stepping can produce oscillations and unstable convergence near barrier enforcement times.

## What Changes
- Default PDE grids become feature-aware when the user relies on default mesh settings:
  - Event-aligned time grid for products with observation times (≈ `4 × days` per interval) and optional Rannacher smoothing after observation events.
  - Adaptive log-space spatial grid by default for barrier products, ensuring barriers (and other critical prices) are grid nodes and increasing grid resolution when needed near key segments.
- Adds a new `time_grid_type="event_aligned"` option and `PDEParams` tuning knobs to control auto-grid behavior.
- Preserves existing behavior for users providing custom mesh configurations or disabling `auto_grid`.

## Impact
- Affected specs: `equity-pde-engine`
- Affected code:
  - `asset/equity/engine/pde/base_pde_solver.py`
  - `asset/equity/engine/pde/time_grid.py`
  - `asset/equity/param/engine_params.py`
  - `test/test_pde_engine.py`

