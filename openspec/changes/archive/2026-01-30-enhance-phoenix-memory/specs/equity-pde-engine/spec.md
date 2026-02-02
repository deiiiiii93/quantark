# equity-pde-engine Specification (Delta)

## ADDED Requirements

### Requirement: Phoenix Memory Coupon Support (PDE)
The system SHALL support `memory_coupon=True` Phoenix options in the PDE engine
by accounting for accumulated missed coupons across observation dates.

#### Scenario: PDE prices memory Phoenix
- **GIVEN** a Phoenix option with `memory_coupon=True`
- **WHEN** the option is priced using `PhoenixPDESolver`
- **THEN** the price reflects accumulated coupons when the coupon barrier is hit

