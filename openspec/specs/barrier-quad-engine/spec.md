# barrier-quad-engine Specification

## Purpose
TBD - created by archiving change add-barrier-quad-engine. Update Purpose after archive.
## Requirements
### Requirement: Barrier Option Quadrature Pricing

The system SHALL provide a `BarrierQuadEngine` that prices single barrier options using FFT-based numerical quadrature with Simpson's rule integration.

#### Scenario: Price UP_OUT barrier call option

- **WHEN** a `BarrierOption` with `barrier_type=UP_OUT`, `option_type=CALL`, and discrete observation dates is priced
- **THEN** the engine SHALL return a price that converges to the analytical solution as grid resolution increases
- **AND** the price SHALL be less than or equal to the corresponding vanilla call price

#### Scenario: Price DOWN_OUT barrier put option

- **WHEN** a `BarrierOption` with `barrier_type=DOWN_OUT`, `option_type=PUT`, and discrete observation dates is priced
- **THEN** the engine SHALL return a price that converges to the analytical solution as grid resolution increases
- **AND** the price SHALL be less than or equal to the corresponding vanilla put price

#### Scenario: Price knock-in via identity

- **WHEN** a `BarrierOption` with `barrier_type=UP_IN` or `DOWN_IN` is priced
- **THEN** the engine SHALL compute the price using the identity KI = Vanilla - KO
- **AND** the sum of knock-in and knock-out prices SHALL equal the vanilla option price

#### Scenario: Reject unsupported product types

- **WHEN** a product other than `BarrierOption` is passed to the engine
- **THEN** the engine SHALL raise a `PricingError` with a descriptive message

### Requirement: Discrete Observation Schedule Support

The system SHALL support discrete barrier monitoring via `ObservationSchedule` or legacy `observation_dates`.

#### Scenario: Use ObservationSchedule for barrier monitoring

- **WHEN** a `BarrierOption` has an `ObservationSchedule` with specific observation times
- **THEN** the engine SHALL check barrier conditions only at those observation times
- **AND** SHALL use per-observation barrier levels if specified in the schedule

#### Scenario: Use legacy observation_dates

- **WHEN** a `BarrierOption` has `observation_dates` but no `ObservationSchedule`
- **THEN** the engine SHALL check barrier conditions at the specified year fractions
- **AND** SHALL use the uniform barrier level from the product

#### Scenario: Default to maturity-only observation

- **WHEN** a `BarrierOption` has no observation schedule and `observation_type=EXPIRY`
- **THEN** the engine SHALL check the barrier condition only at maturity

### Requirement: Rebate Handling

The system SHALL support rebate payments for knocked-out options with configurable payment timing.

#### Scenario: Rebate paid at barrier hit

- **WHEN** a `BarrierOption` has `rebate > 0` and `pay_at_hit=True`
- **THEN** the engine SHALL include the discounted rebate value in the option price
- **AND** the rebate SHALL be discounted from the expected hit time

#### Scenario: Rebate paid at expiry

- **WHEN** a `BarrierOption` has `rebate > 0` and `pay_at_hit=False`
- **THEN** the engine SHALL include the rebate discounted from maturity
- **AND** the rebate SHALL be paid only if the option is knocked out

### Requirement: Numerical Convergence

The system SHALL achieve spectral convergence with increasing grid resolution.

#### Scenario: Convergence with grid size

- **WHEN** the same barrier option is priced with increasing `grid_points` (e.g., 501, 1001, 2001)
- **THEN** the price error relative to analytical solution SHALL decrease
- **AND** the convergence rate SHALL be approximately O(1/N⁴) for Simpson's rule

#### Scenario: Reasonable default accuracy

- **WHEN** a barrier option is priced with default `QuadParams` (1001 grid points)
- **THEN** the price SHALL be within 0.5% of the analytical solution for standard market parameters

### Requirement: Integration with Existing Architecture

The system SHALL integrate with the existing engine architecture and parameter classes.

#### Scenario: Use QuadParams for configuration

- **WHEN** a `BarrierQuadEngine` is created with custom `QuadParams`
- **THEN** the engine SHALL use the specified `grid_points` and `num_std_devs`
- **AND** SHALL validate parameters on construction

#### Scenario: Export from engine package

- **WHEN** a user imports from `asset.equity.engine.quad`
- **THEN** `BarrierQuadEngine` SHALL be available in the module's public API

#### Scenario: Engine type identification

- **WHEN** the engine's `engine_type` attribute is accessed
- **THEN** it SHALL return `EngineType.QUADRATURE`

