## Context
Barrier PDE solvers can be numerically sensitive when the barrier is close to spot, particularly under discrete monitoring. Two common failure modes are:
- insufficient time resolution between observation events, and
- insufficient spatial resolution at (and around) barriers/strikes, leading to oscillations or poor convergence.

The existing PDE framework already supports:
- non-uniform spatial grids via Tavella–Randall (with critical points as segment boundaries), and
- time grids that include event times exactly.

This change refines the defaults so feature-rich products (barriers with observation schedules) get robust grids without requiring users to manually pick mesh sizes.

## Goals / Non-Goals
- Goals:
  - Use event-aligned time grids for discretely monitored barriers by default.
  - Ensure barriers (and other critical prices) are grid nodes in log-space and increase resolution near key segments when needed.
  - Apply Rannacher smoothing after payoff/discontinuity application times (maturity and observation events) when using default meshes.
  - Preserve existing behavior when the user supplies custom mesh settings or disables auto behavior.
- Non-Goals:
  - Full adaptive mesh refinement (AMR) during the solve.
  - Product-specific hardcoded mesh tables in parameter files.

## Decisions
- Decision: Add `time_grid_type="event_aligned"` in `TimeGrid` and allow `BasePDESolver` to select it automatically for default mesh usage.
  - Why: Discrete monitoring introduces known discontinuities at observation times; resolving each observation interval in days improves stability for barrier enforcement.
- Decision: Gate auto-grid behavior on “default mesh configuration”.
  - Why: If a user provides `grid_size/time_steps/time_grid_type/adaptive_grid/s_min/s_max` overrides, the solver should not surprise them by changing the mesh.
- Decision: Default to adaptive (Tavella–Randall) spatial grid for barrier products, and include spot/barrier/strike as critical points.
  - Why: Ensures barriers are nodes and concentrates resolution where the solution has steep gradients.

## Risks / Trade-offs
- More steps/points for some barrier products can increase runtime and memory usage.
  - Mitigation: Provide `max_time_steps` and `max_grid_size` caps and keep auto-grid gated to default mesh usage.
- Small behavioral shifts in barrier PDE results due to event-time Rannacher smoothing.
  - Mitigation: Apply only for default meshes; users can disable `auto_grid` to recover the previous behavior.

## Migration Plan
- Default behavior changes only when users rely on default mesh settings.
- Users who need the previous mesh behavior can set `PDEParams(auto_grid=False)` or explicitly set mesh parameters.

## Open Questions
- Day count for time-grid heuristics uses `EngineParams.bus_days_in_year` (default `252`) to match the project-wide convention and avoid introducing a separate calendar-day basis in PDE params.
