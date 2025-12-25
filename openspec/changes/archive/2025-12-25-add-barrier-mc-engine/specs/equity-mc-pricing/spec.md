## ADDED Requirements
### Requirement: Monte Carlo Pricing for Barrier Options
The system SHALL provide Monte Carlo pricing for single-barrier options using a `BarrierOptionMCEngine` class, supporting knock-in/knock-out barrier types, rebates, and discrete or continuous monitoring.

#### Scenario: Price knock-out barrier with continuous monitoring
- **WHEN** a knock-out barrier option is priced using `BarrierOptionMCEngine` with continuous monitoring
- **THEN** the engine SHALL simulate paths under BSM dynamics and apply barrier conditions along the path
- **AND** the price SHALL be discounted using the risk-free rate

#### Scenario: Price knock-in barrier with discrete monitoring schedule
- **WHEN** a knock-in barrier option is priced using a discrete `ObservationSchedule`
- **THEN** the engine SHALL evaluate barrier hits at the scheduled observation times
- **AND** the payoff SHALL follow the schedule aggregation mode

#### Scenario: Rebate handling at hit versus expiry
- **WHEN** a rebate is specified with `pay_at_hit=True`
- **THEN** the engine SHALL pay the rebate at the hit time (discounted accordingly)
- **AND** if no hit occurs, the rebate SHALL be paid at expiry when applicable

### Requirement: Optional Brownian-Bridge Barrier Handling
The system SHALL provide an optional Brownian-bridge barrier crossing approximation for continuous monitoring in the barrier MC engine.

#### Scenario: Brownian-bridge enabled for continuous barriers
- **WHEN** the barrier MC engine is configured to use Brownian-bridge handling
- **THEN** the engine SHALL estimate barrier crossing between time steps using Brownian-bridge probabilities
- **AND** the pricing result SHALL use those estimates to determine barrier hits

### Requirement: MC Method Selection for Barrier Engine
The system SHALL support the two-level enum pattern for MC method selection in the barrier MC engine.

#### Scenario: Initialize with two-level enum
- **WHEN** the engine is initialized with `EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)`
- **THEN** the engine SHALL configure QMC path generation for barrier pricing

#### Scenario: Initialize with method string
- **WHEN** the engine is initialized with method string "pseudo" or "quasi"
- **THEN** the engine SHALL map to the corresponding `MonteCarloMethod`

### Requirement: Standard Error Estimation for Barrier MC
The system SHALL provide a standard error estimate for barrier MC pricing.

#### Scenario: Standard error output
- **WHEN** the barrier MC engine completes pricing
- **THEN** the engine SHALL compute standard error as `std(discounted_payoffs) / sqrt(N)`
