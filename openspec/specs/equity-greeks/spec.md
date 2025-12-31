# equity-greeks Specification

## Purpose
TBD - created by archiving change update-equity-greeks-observation-theta. Update Purpose after archive.
## Requirements
### Requirement: Numerical theta bump with observation schedules
The system SHALL compute numerical theta for barrier-like equity products that use an `ObservationSchedule` by advancing valuation time, excluding observation entries that are in the past at the bumped valuation, and leaving future observation data unchanged.

#### Scenario: Date-based observation records
- **WHEN** numerical theta advances the valuation date by a bump (e.g., +1 day) for a product with an `ObservationSchedule` containing `observation_date` entries
- **THEN** records with `observation_date` on or before the bumped valuation date SHALL be treated as already observed and excluded from the bumped pricing run
- **AND** remaining records SHALL keep their original `observation_date` values (no forward shift), with time-to-observation recalculated from the bumped valuation date

#### Scenario: Time-based observation records
- **WHEN** numerical theta applies a time bump Δt (e.g., 1/365) for a product with an `ObservationSchedule` containing `observation_time` entries measured from the valuation date
- **THEN** records with `observation_time` ≤ Δt SHALL be treated as already observed and excluded from the bumped pricing run
- **AND** remaining records SHALL keep their original observation data while their effective time-to-observation is reduced by Δt (i.e., resolved relative to the bumped valuation date, not shifted forward)

#### Scenario: Past observations locked
- **WHEN** past observation records are excluded during the theta bump
- **THEN** their barrier checks SHALL NOT be re-evaluated, and aggregation modes SHALL apply only to the remaining future records

#### Scenario: Legacy and continuous monitoring unaffected
- **WHEN** a product has no `ObservationSchedule` or uses continuous/expiry-only monitoring
- **THEN** numerical theta bump behavior SHALL remain unchanged from current implementation

### Requirement: Factor-specific bump sizes for numerical Greeks

The system SHALL support configurable bump sizes for each risk factor in numerical Greeks calculation via a `BumpConfig` class with per-factor bump sizes.

#### Scenario: Default bump configuration

- **GIVEN** a `BumpConfig` with default values
- **THEN** the following bump sizes SHALL be used:
  - `spot_bump = 0.01` (1% relative bump for delta/gamma)
  - `vol_bump = 0.01` (1 vol point absolute bump for vega)
  - `time_bump_days = 1` (1 day absolute bump for theta)
  - `rate_bump = 0.0001` (1bp absolute bump for rho)
  - `div_bump = 0.0001` (1bp absolute bump for dividend_rho)

#### Scenario: Custom bump configuration for precision

- **GIVEN** a `BumpConfig` with `spot_bump=0.001` (0.1%)
- **WHEN** calculating numerical delta/gamma
- **THEN** the spot SHALL be bumped by ±0.1% for central difference

#### Scenario: Override bump size per calculation

- **GIVEN** a `GreeksCalculator` with default `BumpConfig`
- **WHEN** calling `calculate_numerical_vega(vol_bump=0.02)`
- **THEN** vega SHALL be calculated using 2 vol point bump instead of default 1 vol point

#### Scenario: Backward compatibility with bump_size

- **GIVEN** an `EngineParams` with `bump_size=0.001` and no `bump_config`
- **WHEN** creating a `GreeksCalculator` with these params
- **THEN** a `BumpConfig` SHALL be auto-created with `spot_bump=0.001`

### Requirement: Dividend yield sensitivity (dividend_rho)

The system SHALL calculate dividend_rho (∂price/∂dividend_yield) using numerical finite difference with configurable bump size.

#### Scenario: Dividend rho for call option

- **GIVEN** a European call option with positive dividend yield
- **WHEN** calculating dividend_rho
- **THEN** the result SHALL be negative (higher dividend yield → lower call price)

#### Scenario: Dividend rho for put option

- **GIVEN** a European put option with positive dividend yield
- **WHEN** calculating dividend_rho
- **THEN** the result SHALL be positive (higher dividend yield → higher put price)

#### Scenario: Dividend rho with zero dividend yield

- **GIVEN** an option with zero dividend yield
- **WHEN** calculating dividend_rho
- **THEN** the calculation SHALL complete successfully using the configured div_bump

### Requirement: Bump size validation

The system SHALL validate bump sizes in `BumpConfig.__post_init__` to ensure positive values within reasonable ranges.

#### Scenario: Spot bump validation

- **WHEN** creating a `BumpConfig` with `spot_bump <= 0` or `spot_bump > 0.1`
- **THEN** a `ValidationError` SHALL be raised

#### Scenario: Vol bump validation

- **WHEN** creating a `BumpConfig` with `vol_bump <= 0` or `vol_bump > 0.1`
- **THEN** a `ValidationError` SHALL be raised

#### Scenario: Time bump validation

- **WHEN** creating a `BumpConfig` with `time_bump_days <= 0` or `time_bump_days > 30`
- **THEN** a `ValidationError` SHALL be raised

#### Scenario: Rate bump validation

- **WHEN** creating a `BumpConfig` with `rate_bump <= 0` or `rate_bump > 0.01`
- **THEN** a `ValidationError` SHALL be raised

#### Scenario: Div bump validation

- **WHEN** creating a `BumpConfig` with `div_bump <= 0` or `div_bump > 0.01`
- **THEN** a `ValidationError` SHALL be raised

