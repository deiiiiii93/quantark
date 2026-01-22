# equity-mc-pricing Specification (Delta)

## ADDED Requirements

### Requirement: RQMC Target Std Scaling
The system SHALL allow RQMC target standard error to be specified in absolute
terms or scaled relative to notional/price.

#### Scenario: Absolute target std (default)
- **GIVEN** `rqmc_target_std_mode="absolute"`
- **WHEN** RQMC pricing is executed
- **THEN** the target standard error is used as-is

#### Scenario: Relative target std by notional
- **GIVEN** `rqmc_target_std_mode="relative_notional"` and a product with notional
- **WHEN** RQMC pricing is executed
- **THEN** the target standard error is scaled by notional

#### Scenario: Relative target std by price scale
- **GIVEN** `rqmc_target_std_mode="relative_price"`
- **WHEN** RQMC pricing is executed
- **THEN** the target standard error is scaled by the configured price scale

