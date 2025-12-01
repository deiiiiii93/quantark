# Equity Monte Carlo Pricing Specification

## ADDED Requirements

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
