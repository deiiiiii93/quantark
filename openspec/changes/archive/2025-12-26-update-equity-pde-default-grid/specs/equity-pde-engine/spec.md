## ADDED Requirements
### Requirement: Feature-Aware Default PDE Grids
When `PDEEngine`/PDE solvers are used with default mesh settings, the system SHALL choose spatial and temporal grids based on product features to improve numerical stability for barrier products.

#### Scenario: Discrete barrier uses event-aligned time grid by default
- **WHEN** a discretely monitored `BarrierOption` with observation times is priced using `PDEEngine` with default `PDEParams`
- **THEN** the PDE solver time grid includes all observation times
- **AND** the solver uses an event-aligned grid with approximately `4 × days per interval` resolution between observation dates

#### Scenario: Barrier products use adaptive log-space grid by default
- **WHEN** a barrier product (single or double barrier / one-touch) is priced using `PDEEngine` with default `PDEParams`
- **THEN** the PDE solver uses an adaptive log-space grid concentrated at barriers (and strike when applicable)
- **AND** barrier levels are grid nodes

#### Scenario: Default meshes apply event-time Rannacher smoothing
- **WHEN** a discretely monitored barrier product is priced using `PDEEngine` with default `PDEParams`
- **THEN** the PDE solver applies Rannacher smoothing near maturity
- **AND** applies Rannacher smoothing after observation event times

#### Scenario: Custom mesh configuration bypasses auto grids
- **WHEN** a user supplies custom mesh configuration (e.g., explicit `grid_size` / `time_steps` / `time_grid_type`) or sets `auto_grid=False`
- **THEN** the PDE solver uses the user-provided mesh settings without overriding them

