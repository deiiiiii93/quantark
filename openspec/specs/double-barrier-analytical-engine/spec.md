# double-barrier-analytical-engine Specification

## Purpose
TBD - created by archiving change add-double-barrier-analytical-engine. Update Purpose after archive.
## Requirements
### Requirement: Engine prices double knock-out options continuously
The system SHALL provide an analytical engine that prices double knock-out call and put options under continuous barrier monitoring using the Ikeda & Kuintomo (1992) formula.

#### Scenario: Continuous double knock-out call
- **WHEN** a `DoubleBarrierOption` (call, knock-out, lower=80, upper=120) is priced with `DoubleBarrierOptionAnalyticalEngine` using continuous observation
- **THEN** the price SHALL match Table 4-15 benchmark values within 1e-4 absolute error for the corresponding parameters

#### Scenario: Continuous double knock-out put
- **WHEN** a `DoubleBarrierOption` (put, knock-out, lower=80, upper=120) is priced with `DoubleBarrierOptionAnalyticalEngine` using continuous observation
- **THEN** the price SHALL be computed via the Ikeda & Kuintomo infinite-series formula and be non-negative

### Requirement: Engine supports knock-in via knock-out parity
The system SHALL price double knock-in options by parity: knock-in = vanilla - knock-out.

#### Scenario: Double knock-in call
- **WHEN** a `DoubleBarrierOption` (call, knock-in) is priced
- **THEN** the engine SHALL compute the price as `BlackScholesEngine.call_price() - DoubleBarrierOptionAnalyticalEngine(knock-out price)`

#### Scenario: Double knock-in put
- **WHEN** a `DoubleBarrierOption` (put, knock-in) is priced
- **THEN** the engine SHALL compute the price as `BlackScholesEngine.put_price() - DoubleBarrierOptionAnalyticalEngine(knock-out price)`

### Requirement: Engine supports daily observation via barrier shift
The system SHALL support daily (discrete) barrier monitoring by applying a barrier-shift adjustment and then using the continuous formula.

#### Scenario: Daily observed double barrier
- **WHEN** a `DoubleBarrierOption` is priced with observation type `DAILY`
- **THEN** the engine SHALL shift the lower barrier down and the upper barrier up by `exp(±0.5826 * σ * sqrt(1/252))` before pricing

### Requirement: Engine supports expiry-only observation
The system SHALL support barrier monitoring only at maturity by pricing the truncated-domain vanilla payoff.

#### Scenario: Expiry-observed double barrier call
- **WHEN** a `DoubleBarrierOption` (call) is priced with observation type `EXPIRY`
- **THEN** the price SHALL equal the expected payoff of `max(S_T - X, 0) * 1_{L < S_T < U}`

#### Scenario: Expiry-observed double barrier put
- **WHEN** a `DoubleBarrierOption` (put) is priced with observation type `EXPIRY`
- **THEN** the price SHALL equal the expected payoff of `max(X - S_T, 0) * 1_{L < S_T < U}`

### Requirement: Input validation and edge cases
The system SHALL validate all inputs and handle edge cases safely.

#### Scenario: Invalid strike outside barriers
- **WHEN** the strike is less than or equal to the lower barrier, or greater than or equal to the upper barrier
- **THEN** the engine SHALL raise `ValidationError`

#### Scenario: Spot already outside barriers for knock-out
- **WHEN** the spot price is already at or outside a knock-out barrier
- **THEN** the engine SHALL return 0 immediately

#### Scenario: Zero time to maturity
- **WHEN** time to maturity is zero
- **THEN** the engine SHALL return the intrinsic value subject to the barrier condition

