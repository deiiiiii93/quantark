## ADDED Requirements

### Requirement: Product-Agnostic Quadrature Core
The system SHALL provide a quadrature core that implements the recursion in Eq. 3.5 using per-observation boundary levels (`K^-`, `K^+`) and linear payoff coefficients (`a^-`, `b^-`, `a^+`, `b^+`), plus terminal coefficients (`a_M`, `b_M`).

#### Scenario: Vanilla payoff via core inputs
- **GIVEN** a single observation date at maturity with `K^- = 0`, `K^+ = +inf`, and terminal coefficients matching a European call
- **WHEN** the quadrature core prices the contract
- **THEN** the result matches the European quadrature engine within numerical tolerance

### Requirement: Adapter-Based Product Mapping
The system SHALL map product-specific parameters to quadrature-core inputs via adapter functions or classes, keeping product logic out of the core implementation.

#### Scenario: Barrier rebate mapping
- **GIVEN** a discretely monitored barrier option with rebates
- **WHEN** the quad engine prices the option
- **THEN** rebates are represented as boundary coefficients at observation dates and the core is invoked without direct factor manipulation in the engine

### Requirement: Observation-Date Time Stepping
The quadrature core SHALL step over the actual observation dates (piecewise-constant parameters) rather than an implicit uniform time grid.

#### Scenario: Irregular observation schedule
- **GIVEN** an irregular set of observation times
- **WHEN** the quad engine prices the contract
- **THEN** the core uses the provided `Δt_m` intervals without resampling the schedule
