## ADDED Requirements
### Requirement: Barrier Option Analytical Pricing
The system SHALL provide a closed-form analytical engine for single-barrier options supporting knock-in/knock-out, up/down barriers, continuous monitoring, discrete monitoring with barrier shift, and expiry-only monitoring via decomposition.

#### Scenario: Continuous monitoring pricing
- **WHEN** pricing a `BarrierOption` with `observation_type=CONTINUOUS`
- **THEN** the engine SHALL use closed-form barrier formulas for knock-out with optional rebate
- **AND** SHALL price knock-in via parity against vanilla minus knock-out

#### Scenario: Discrete monitoring with barrier shift
- **WHEN** pricing a discretely observed barrier with a regular observation interval and fixed per-record payoff
- **THEN** the engine SHALL apply the beta-based barrier shift (β=0.5825971579) using `dt=frequency` to adjust the barrier before using continuous formulas
- **AND** SHALL raise `ValidationError` if the schedule is irregular or payoffs are inconsistent for analytical shift

#### Scenario: Expiry-only monitoring decomposition
- **WHEN** `observation_type=EXPIRY`
- **THEN** the engine SHALL decompose the barrier payoff into combinations of European vanillas and cash digitals without tenor/365 scaling
- **AND** SHALL honor rebates by adding a digital-style rebate term when the barrier condition is not met

#### Scenario: Method selection and integration
- **WHEN** the barrier analytical engine is instantiated
- **THEN** it SHALL inherit from `BaseEngine`, source market data from `PricingEnvironment`, and raise `PricingError` for unsupported product types

#### Scenario: Input validation and stability
- **WHEN** inputs are invalid (non-positive spot/strike/barrier, negative volatility, non-regular discrete schedule)
- **THEN** the engine SHALL raise `ValidationError`
- **AND** SHALL return intrinsic value when maturity is effectively zero

