# equity-mc-pricing Specification

## Purpose
TBD - created by archiving change implement-euro-mc-engine. Update Purpose after archive.
## Requirements
### Requirement: Monte Carlo Pricing for European Vanilla Options
The system SHALL provide Monte Carlo pricing for European vanilla call and put options using the `EuropeanMCEngine` class.

#### Scenario: Price European call with normal MC
- **WHEN** a European call option is priced using `MonteCarloMethod.PSEUDO`
- **THEN** the engine returns a price and standard error estimate
- **AND** the price converges to the Black-Scholes analytical price as num_paths increases

#### Scenario: Price European put with QMC
- **WHEN** a European put option is priced using `MonteCarloMethod.QUASI`
- **THEN** the engine uses Sobol sequences for path generation
- **AND** the convergence rate is faster than normal MC

#### Scenario: Price with RQMC adaptive batching
- **WHEN** `MonteCarloMethod.RANDOMIZED_QUASI` is used with target standard error
- **THEN** the engine runs adaptive batches until target precision is achieved
- **AND** the final result includes batch statistics (batches used, total paths)

### Requirement: Two-Level Enum Method Selection
The system SHALL support the two-level enum pattern for method selection consistent with other QuantArk engines.

#### Scenario: Initialize with two-level enum
- **WHEN** engine is initialized with `EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)`
- **THEN** the engine extracts the method and configures QMC path generation

#### Scenario: Initialize with method enum directly
- **WHEN** engine is initialized with `MonteCarloMethod.QUASI` directly
- **THEN** the engine accepts it and configures QMC path generation

#### Scenario: Initialize with string (backward compatibility)
- **WHEN** engine is initialized with method string "quasi"
- **THEN** the engine converts to `MonteCarloMethod.QUASI` and proceeds

### Requirement: Path Generator Integration
The system SHALL integrate with `GBMPathGenerator` for all path simulation needs.

#### Scenario: Normal MC path generation
- **WHEN** `MonteCarloMethod.PSEUDO` is selected
- **THEN** `GBMPathGenerator` is configured with `PseudoRandomNormalGenerator`

#### Scenario: QMC path generation
- **WHEN** `MonteCarloMethod.QUASI` is selected
- **THEN** `GBMPathGenerator` is configured with `SobolNormalGenerator`

#### Scenario: RQMC path generation with scrambling
- **WHEN** `MonteCarloMethod.RANDOMIZED_QUASI` is selected
- **THEN** `GBMPathGenerator` uses scrambled Sobol with different scrambles per batch

### Requirement: Variance Reduction Support
The system SHALL support variance reduction techniques configured through `MCParams`.

#### Scenario: Antithetic variates in MC mode
- **WHEN** `MCParams.use_antithetic=True` with pseudo MC
- **THEN** the path generator produces antithetic pairs
- **AND** the variance is reduced compared to independent sampling

#### Scenario: No antithetic variates in QMC mode
- **WHEN** `MCParams.use_antithetic=True` with QMC
- **THEN** antithetic variates are NOT applied (incompatible with QMC)

### Requirement: Standard Error Estimation
The system SHALL provide standard error estimates for MC convergence diagnostics.

#### Scenario: Standard error for normal MC
- **WHEN** normal MC is run with N paths
- **THEN** standard error is estimated as `std(payoffs) / sqrt(N)`

#### Scenario: Standard error for RQMC batching
- **WHEN** RQMC is run with B batches
- **THEN** standard error is computed from batch-to-batch variance
- **AND** reflects RQMC convergence properties

### Requirement: Parameter Validation
The system SHALL validate all pricing inputs and raise descriptive errors for invalid configurations.

#### Scenario: Invalid MC parameters
- **WHEN** `MCParams` has non-positive num_paths or time_steps
- **THEN** a `ValidationError` is raised with a descriptive message

#### Scenario: Unsupported product type
- **WHEN** the engine is asked to price a non-European product
- **THEN** a `PricingError` is raised indicating European options only

#### Scenario: Invalid method string
- **WHEN** an unknown method string is provided
- **THEN** a `ValidationError` is raised listing valid methods

### Requirement: Discounting and Payoff Calculation
The system SHALL correctly discount expected payoffs using risk-free rates from the pricing environment.

#### Scenario: Risk-neutral pricing for call
- **WHEN** pricing a European call
- **THEN** terminal payoffs are `max(S_T - K, 0)`
- **AND** the price is `exp(-r*T) * E[payoff]`

#### Scenario: Risk-neutral pricing for put
- **WHEN** pricing a European put
- **THEN** terminal payoffs are `max(K - S_T, 0)`
- **AND** the price is `exp(-r*T) * E[payoff]`

#### Scenario: Dividend yield handling
- **WHEN** the underlying has dividend yield q
- **THEN** the drift is `(r - q)` in the path simulation
- **AND** the discounting uses the risk-free rate r

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

