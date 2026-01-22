## ADDED Requirements
### Requirement: Event-Time Alignment Enforcement
The system SHALL require discrete observation times to align exactly with the PDE time grid and SHALL raise a ValidationError when they do not.

#### Scenario: Misaligned discrete observation time
- **GIVEN** a discretely monitored product and a custom PDE time grid that does not include an observation time
- **WHEN** the PDE solver maps observation indices
- **THEN** a ValidationError is raised with guidance to use an event-aligned grid or increase time steps

#### Scenario: Aligned observation time
- **GIVEN** a discretely monitored product and an event-aligned PDE time grid
- **WHEN** the PDE solver maps observation indices
- **THEN** the observation time is matched exactly and no error is raised

### Requirement: Event-Adjacent Theta Control
The system SHALL allow a separate theta value to be applied immediately before event times.

#### Scenario: Event theta applied for a single step
- **GIVEN** a discretely monitored product with event-adjacent theta enabled
- **WHEN** the solver steps over a time interval immediately before an event time
- **THEN** the solver uses `event_theta` for that step and reverts to the base `theta` afterward

### Requirement: Barrier-Focused Grid Refinement
The system SHALL support optional log-space refinement around barrier levels to improve spatial resolution.

#### Scenario: Barrier refinement enabled
- **GIVEN** a PDE configuration with non-zero `barrier_refine_log_width`
- **WHEN** the spatial grid is constructed with auto-grid enabled
- **THEN** additional critical points are inserted around each barrier at the configured refinement levels

### Requirement: Barrier Domain Expansion
The system SHALL support optional expansion of spatial bounds around barriers when configured.

#### Scenario: Barrier domain expansion enabled
- **GIVEN** a PDE configuration with `barrier_domain_expand > 0`
- **WHEN** spatial bounds are resolved
- **THEN** bounds are expanded to include a buffer around barrier levels

### Requirement: Asymptotic Boundary Mode
The system SHALL support an asymptotic boundary mode for far-field pricing of linear payoff components.

#### Scenario: Call-rebate V0 upper boundary
- **GIVEN** `boundary_mode=asymptotic` and a call-rebate payoff at maturity
- **WHEN** the solver sets the upper boundary for the V0 surface
- **THEN** the boundary uses a linear asymptotic value derived from discounted call payoff components

#### Scenario: Standard V1 lower boundary without protection
- **GIVEN** `boundary_mode=asymptotic` and a standard snowball with no protection
- **WHEN** the solver sets the lower boundary for the V1 surface
- **THEN** the boundary uses a linear asymptotic value derived from discounted put payoff components
