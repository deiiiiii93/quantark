# equity-quad-engine Specification (Delta)

## ADDED Requirements

### Requirement: Phoenix Memory Coupon Support (QUAD)
The system SHALL support `memory_coupon=True` Phoenix options in the quadrature
engine by accounting for accumulated missed coupons across observation dates.

#### Scenario: QUAD prices memory Phoenix
- **GIVEN** a Phoenix option with `memory_coupon=True`
- **WHEN** the option is priced using `PhoenixQuadEngine`
- **THEN** the price reflects accumulated coupons when the coupon barrier is hit

