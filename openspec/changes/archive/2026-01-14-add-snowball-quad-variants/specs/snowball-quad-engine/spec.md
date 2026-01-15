## ADDED Requirements
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

## MODIFIED Requirements
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
