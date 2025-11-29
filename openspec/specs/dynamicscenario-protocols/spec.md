# dynamicscenario-protocols Specification

## Purpose
TBD - created by archiving change add-fi-dynamicscenario. Update Purpose after archive.
## Requirements
### Requirement: Base Dynamic Scenario Engine Protocol

The system SHALL provide a `BaseDynamicScenarioEngine` protocol that defines the contract for all asset-class-specific dynamic scenario engines.

The protocol MUST define:
- `run(portfolio, day_path, hedge_strategy, transaction_cost_model)` method returning scenario results
- Support for any portfolio type implementing a base portfolio protocol
- Configurable risk metrics calculation
- Optional hedging strategy integration

#### Scenario: Engine Protocol Conformance

- **GIVEN** an `EquityDynamicScenarioEngine` implementing the protocol
- **WHEN** `run()` is called with an equity portfolio and day path
- **THEN** it executes the simulation and returns `DynamicScenarioResults`
- **AND** the results conform to `BaseScenarioResults` protocol

#### Scenario: FI Engine Protocol Conformance

- **GIVEN** an `FIDynamicScenarioEngine` implementing the protocol
- **WHEN** `run()` is called with an FI portfolio and rate path
- **THEN** it executes the simulation and returns `FIDynamicScenarioResults`
- **AND** the results include FI-specific risk measures (DV01, convexity)

### Requirement: Risk Metrics Adapter Protocol

The system SHALL provide a `RiskMetricsAdapter` protocol that abstracts risk measure calculation across asset classes.

The adapter MUST support:
- Computing risk measures for the specific asset class (Greeks for equity, DV01/duration for FI)
- Returning a standardized dictionary of risk measure values
- Integration with day step state capture

#### Scenario: Equity Risk Metrics

- **GIVEN** an equity portfolio and pricing environment
- **WHEN** `calculate_risk_measures()` is called
- **THEN** it returns a dictionary with delta, gamma, vega, theta values

#### Scenario: FI Risk Metrics

- **GIVEN** an FI portfolio and pricing environment
- **WHEN** `calculate_risk_measures()` is called
- **THEN** it returns a dictionary with dv01, convexity, modified_duration values

### Requirement: Extended Path Components for Rate Curves

The system SHALL extend `ParameterChange` and `PathBuilder` to support rate curve changes.

The extension MUST support:
- Rate parallel shifts (all tenors move by same amount)
- Key-rate bumps (specific tenor changes)
- Curve twists (short and long ends move differently)

#### Scenario: Parallel Rate Shift

- **GIVEN** a `ParameterChange` with parameter "rate" and stress type ABSOLUTE
- **WHEN** applied to a pricing environment
- **THEN** the rate curve shifts uniformly by the specified amount

#### Scenario: Key-Rate Bump via PathBuilder

- **GIVEN** a `PathBuilder` with `rate_key_bump(tenor="5Y", bps=25)` called
- **WHEN** the path is built
- **THEN** the resulting `DayStep` contains a change targeting the 5Y tenor with +25bps

