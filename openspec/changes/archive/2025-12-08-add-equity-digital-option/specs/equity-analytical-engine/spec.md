## ADDED Requirements

### Requirement: Cash digital option analytical pricing
The system SHALL provide closed-form Black-Scholes pricing for European cash-or-nothing digital call and put options.

#### Scenario: Price digital call with BSM
- **WHEN** pricing a European cash digital call with payout P, maturity T > 0, spot S, strike K, rate r, dividend yield q, and volatility σ
- **THEN** the price SHALL be P * exp(-r*T) * N(d₂) where d₂ = [ln(S/K) + (r - q - 0.5σ²)T] / (σ√T)
- **AND** the engine SHALL source S, r, q, and σ from the `PricingEnvironment`

#### Scenario: Price digital put with BSM
- **WHEN** pricing a European cash digital put with payout P, maturity T > 0, spot S, strike K, rate r, dividend yield q, and volatility σ
- **THEN** the price SHALL be P * exp(-r*T) * N(-d₂) using the same d₂ definition
- **AND** the engine SHALL source market inputs from the `PricingEnvironment`

#### Scenario: Handle near-expiry payoff
- **WHEN** time to maturity is less than 1e-10
- **THEN** the engine SHALL return the product payoff evaluated at the current spot without further computation

#### Scenario: Input validation and product type checks
- **WHEN** spot ≤ 0, strike ≤ 0, volatility ≤ 0, or payout ≤ 0
- **THEN** the engine SHALL raise `ValidationError`
- **WHEN** a non-digital product is passed to the digital pricing engine
- **THEN** the engine SHALL raise `PricingError`

