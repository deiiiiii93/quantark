# snowball-quad-engine Specification

## Purpose
TBD - created by archiving change refactor-snowball-quad-engine. Update Purpose after archive.
## Requirements
### Requirement: Regime-Switching Quadrature for Snowball
The system SHALL implement a snowball quadrature engine that propagates two value functions (knocked-in and not-knocked-in) within a single backward quadrature recursion.

#### Scenario: Price standard snowball with regime switching
- **GIVEN** a SnowballOption with discrete KO observations and discrete KI monitoring
- **WHEN** `SnowballQuadEngine.price(snowball, pricing_env)` is called
- **THEN** the engine returns a price derived from a single recursion over `V_in` and `V_out` (no payoff decomposition into primitives)

### Requirement: Terminal Conditions for V_in and V_out
The system SHALL set terminal conditions using the snowball product payoff helpers.

#### Scenario: V_in and V_out initialization at maturity
- **GIVEN** a SnowballOption and maturity time
- **WHEN** the recursion initializes terminal values
- **THEN** `V_in(S, T)` equals `get_maturity_payoff_v1(S)` and `V_out(S, T)` equals `get_maturity_payoff_v0(S)`

### Requirement: Discrete KO Observations
The system SHALL apply KO barrier checks only at discrete KO observation times, overwriting both states with the KO payoff when breached.

#### Scenario: KO jump at observation
- **GIVEN** a SnowballOption with KO barriers at monthly observation dates
- **WHEN** the recursion steps over a KO observation date
- **THEN** both `V_in` and `V_out` are set to the KO payoff for `S` beyond the KO barrier

### Requirement: Discrete KI Observations
The system SHALL apply discrete KI monitoring only at KI observation times, switching the not-knocked-in state to the knocked-in state when the KI barrier is breached.

#### Scenario: Discrete KI transition
- **GIVEN** a SnowballOption with KI observation dates
- **WHEN** the recursion steps over a KI observation date
- **THEN** `V_out(S, t)` is set to `V_in(S, t)` for `S` beyond the KI barrier

### Requirement: Continuous KI via Brownian Bridge
The system SHALL support continuous KI monitoring by applying Brownian-bridge transition probabilities between observation dates.

#### Scenario: Continuous KI transition
- **GIVEN** a SnowballOption with `ki_continuous=True`
- **WHEN** the recursion steps over a time interval
- **THEN** `V_out` mixes into `V_in` using Brownian-bridge hit probabilities within that interval

### Requirement: KO Precedence at Coincident Observations
The system SHALL prioritize KO over KI when both are triggered at the same observation time.

#### Scenario: KO and KI triggered at same observation
- **GIVEN** a SnowballOption with KO and KI barriers breached at the same observation
- **WHEN** the recursion applies observation logic
- **THEN** both `V_in` and `V_out` take the KO payoff and KI transition is not applied

### Requirement: Validation for Supported Configurations
The system SHALL validate snowball configurations for quadrature compatibility and raise clear errors for unsupported features.

#### Scenario: Supported airbag and call-rebate
- **GIVEN** a SnowballOption with airbag features and call-rebate V0 enabled
- **WHEN** `SnowballQuadEngine.price(...)` is called
- **THEN** the engine accepts the configuration without raising a configuration error

#### Scenario: disable_ko_after_ki supported
- **GIVEN** a SnowballOption with `disable_ko_after_ki=True`
- **WHEN** `SnowballQuadEngine.price(...)` is called
- **THEN** the engine accepts the configuration and applies KO suppression after KI

### Requirement: Airbag V1 payoff support
The system SHALL support airbag snowball configurations by using the product V1 payoff when initializing the knocked-in terminal condition.

#### Scenario: Airbag payoff at maturity
- **GIVEN** a SnowballOption with an airbag barrier and airbag participation configuration
- **WHEN** `SnowballQuadEngine.price(...)` initializes terminal values
- **THEN** `V_in(S, T)` reflects the airbag-adjusted V1 payoff

### Requirement: Call-rebate V0 payoff support
The system SHALL support call-rebate V0 configurations by using the product V0 payoff when initializing the not-knocked-in terminal condition.

#### Scenario: Call-rebate V0 payoff at maturity
- **GIVEN** a SnowballOption with `call_rebate_enabled=True`
- **WHEN** `SnowballQuadEngine.price(...)` initializes terminal values
- **THEN** `V_out(S, T)` reflects the call-rebate V0 payoff

### Requirement: disable_ko_after_ki interaction
The system SHALL suppress KO after KI when `disable_ko_after_ki=True`.

#### Scenario: KO ignored after KI
- **GIVEN** a SnowballOption with `disable_ko_after_ki=True`
- **WHEN** KO is observed after KI has occurred
- **THEN** KO does not overwrite the knocked-in value surface

