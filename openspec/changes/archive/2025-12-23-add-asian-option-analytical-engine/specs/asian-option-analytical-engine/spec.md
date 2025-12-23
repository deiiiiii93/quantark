# asian-option-analytical-engine Specification

## Purpose

Provide analytical pricing methods for Asian options, offering fast deterministic pricing as an alternative to Monte Carlo simulation. Supports both geometric (exact) and arithmetic (approximation) averaging with multiple method implementations.

## ADDED Requirements

### Requirement: Geometric Average Option Pricing (Kemna-Vorst)

The system SHALL provide exact closed-form pricing for geometric average-rate Asian options using the Kemna-Vorst method.

#### Scenario: Price geometric average call option

- **GIVEN** an AsianOption with averaging_type=GEOMETRIC, asian_strike_type=FIXED
- **AND** S=80, K=85, T=0.25, r=0.05, b=0.08, σ=0.20
- **WHEN** pricing with method=KEMNA_VORST
- **THEN** the engine SHALL compute σ_A = σ/√3 and b_A = (b - σ²/6)/2
- **AND** SHALL apply the generalized BSM formula with adjusted parameters
- **AND** the price SHALL match theoretical value within 1e-6

#### Scenario: Price geometric average put option

- **GIVEN** an AsianOption with averaging_type=GEOMETRIC, option_type=PUT
- **AND** S=80, K=85, T=0.25, r=0.05, b=0.08, σ=0.20
- **WHEN** pricing with KEMNA_VORST method
- **THEN** the put price SHALL be approximately 4.6922

### Requirement: Turnbull-Wakeman Arithmetic Approximation

The system SHALL provide the Turnbull-Wakeman moment-matching approximation for arithmetic average-rate options.

#### Scenario: Price arithmetic average call with Turnbull-Wakeman

- **GIVEN** an AsianOption with averaging_type=ARITHMETIC, asian_strike_type=FIXED
- **WHEN** pricing with method=TURNBULL_WAKEMAN
- **THEN** the engine SHALL compute exact first moment M₁ and second moment M₂
- **AND** SHALL derive adjusted b_A = ln(M₁)/T and σ_A = √[ln(M₂)/T - 2b_A]
- **AND** SHALL apply the generalized BSM formula

#### Scenario: Handle b=0 case in Turnbull-Wakeman

- **GIVEN** an AsianOption with cost-of-carry b = 0 (e.g., futures option)
- **WHEN** pricing with TURNBULL_WAKEMAN
- **THEN** the engine SHALL use M₁ = 1 and the special M₂ formula for b=0
- **AND** SHALL NOT raise a division by zero error

#### Scenario: In-period pricing with Turnbull-Wakeman

- **GIVEN** an AsianOption where m observations have already occurred
- **AND** the realized average S_A is known
- **WHEN** pricing with TURNBULL_WAKEMAN
- **THEN** the engine SHALL adjust strike X_adj = (T₂/T)×X - (τ/T)×S_A
- **AND** SHALL multiply the option value by T/T₂

### Requirement: Levy Arithmetic Approximation

The system SHALL provide the Levy approximation for arithmetic average-rate options.

#### Scenario: Price arithmetic average call with Levy

- **GIVEN** an AsianOption with averaging_type=ARITHMETIC
- **AND** S=6.80, K=6.90, T=T₂=0.5, r=0.07, b=-0.02, σ=0.14
- **WHEN** pricing with method=LEVY
- **THEN** the call price SHALL be approximately 0.0944

#### Scenario: Compute Asian put via put-call parity

- **WHEN** pricing an arithmetic average put with LEVY method
- **THEN** the engine SHALL use: p = c - S_E + X*×exp(-r×T₂)
- **AND** the put price SHALL be approximately 0.2237 for the example parameters

#### Scenario: Reject b=0 with Levy method

- **GIVEN** an AsianOption with b = 0
- **WHEN** attempting to price with LEVY method
- **THEN** the engine SHALL raise ValidationError with message indicating b=0 not supported
- **AND** SHALL suggest using TURNBULL_WAKEMAN instead

### Requirement: Curran Geometric Conditioning Approximation

The system SHALL provide the Curran approximation using geometric conditioning for arithmetic average options.

#### Scenario: Price with Curran approximation

- **GIVEN** an AsianOption with averaging_type=ARITHMETIC
- **AND** discrete observation schedule with n fixings
- **WHEN** pricing with method=CURRAN
- **THEN** the engine SHALL compute μ_i, σ_i, σ_xi for each fixing i
- **AND** SHALL compute modified strike K_m
- **AND** SHALL apply the geometric conditioning formula

#### Scenario: In-period pricing with Curran

- **GIVEN** an AsianOption with m observations already realized
- **WHEN** pricing with CURRAN
- **THEN** the engine SHALL adjust strike: X = (n×X - m×S_A)/(n-m)
- **AND** SHALL scale result by (n-m)/n

### Requirement: Discrete Arithmetic Approximation (Haug-Haug-Margrabe)

The system SHALL provide the discrete arithmetic average approximation from Haug, Haug, and Margrabe.

#### Scenario: Price discrete arithmetic average call

- **GIVEN** an AsianOption with discrete observation schedule
- **AND** S=100, S_A=110, K=105, t1=0, T=0.5, n=360, m=180, r=0.07, b=0.02, σ=0.25
- **WHEN** pricing with method=DISCRETE_HHM
- **THEN** the call price SHALL be approximately 2.0971

#### Scenario: Handle single fixing remaining

- **GIVEN** an AsianOption where m = n - 1 (only one fixing left)
- **WHEN** pricing with DISCRETE_HHM
- **THEN** the engine SHALL use adjusted BSM formula with X̂ = n×X - (n-1)×S_A
- **AND** SHALL weight by 1/n

#### Scenario: Handle certain exercise for call

- **GIVEN** an AsianOption where S_A > (n/m)×X
- **WHEN** pricing a call with any discrete method
- **THEN** the engine SHALL return exp(-r×T)×(Ŝ_A - X) where Ŝ_A is expected final average
- **AND** SHALL return 0 for a put

### Requirement: Floating-Strike Asian Option Pricing

The system SHALL price floating-strike (average strike) Asian options using Henderson-Wojakowski symmetry.

#### Scenario: Price floating-strike call via symmetry

- **GIVEN** an AsianOption with asian_strike_type=FLOATING, option_type=CALL
- **WHEN** pricing with any analytical method
- **THEN** the engine SHALL transform to fixed-strike put with r → r-b, b → -b
- **AND** SHALL use the appropriate fixed-strike pricing method

#### Scenario: Price floating-strike put via symmetry

- **GIVEN** an AsianOption with asian_strike_type=FLOATING, option_type=PUT
- **WHEN** pricing
- **THEN** the engine SHALL transform to fixed-strike call with parameter adjustment

### Requirement: Method Selection Pattern

The system SHALL support method selection using the two-level enum pattern.

#### Scenario: Select method via EngineType pattern

- **WHEN** creating engine with `EngineType.ANALYTICAL(AsianAnalyticalMethod.TURNBULL_WAKEMAN)`
- **THEN** the engine SHALL use TURNBULL_WAKEMAN method for all pricing

#### Scenario: Select method via direct enum

- **WHEN** creating engine with `AsianAnalyticalMethod.CURRAN`
- **THEN** the engine SHALL use CURRAN method

#### Scenario: Select method via string

- **WHEN** creating engine with `method="levy"`
- **THEN** the engine SHALL use LEVY method (case-insensitive)

#### Scenario: Auto-select based on averaging type

- **WHEN** no method is specified
- **AND** the option uses geometric averaging
- **THEN** the engine SHALL default to KEMNA_VORST
- **WHEN** the option uses arithmetic averaging
- **THEN** the engine SHALL default to TURNBULL_WAKEMAN

### Requirement: Greeks Calculation

The system SHALL calculate option Greeks (delta, gamma, vega, theta, rho).

#### Scenario: Analytical Greeks for geometric options

- **WHEN** calculating Greeks for a geometric average option
- **THEN** the engine SHALL use closed-form derivatives where available
- **AND** delta, gamma, vega SHALL match finite-difference estimates within 1e-4

#### Scenario: Numerical Greeks for arithmetic approximations

- **WHEN** calculating Greeks for arithmetic average options
- **THEN** the engine SHALL use finite-difference method with configurable bump size
- **AND** SHALL apply central differences for accuracy

### Requirement: Input Validation and Edge Cases

The system SHALL validate inputs and handle edge cases robustly.

#### Scenario: Validate product type

- **WHEN** a non-AsianOption product is passed
- **THEN** the engine SHALL raise PricingError with descriptive message

#### Scenario: Handle near-expiry option

- **WHEN** time to maturity T < 1e-10
- **THEN** the engine SHALL return the intrinsic payoff directly

#### Scenario: Validate positive parameters

- **WHEN** spot S ≤ 0, strike K ≤ 0, or volatility σ ≤ 0
- **THEN** the engine SHALL raise ValidationError

#### Scenario: Handle numerical instability

- **WHEN** parameters would cause overflow (e.g., extreme moneyness)
- **THEN** the engine SHALL use safe math utilities from util.numerical
- **AND** SHALL clamp extreme values or fallback to simpler approximation

### Requirement: Integration with PricingEnvironment

The system SHALL integrate with the standard PricingEnvironment for market data.

#### Scenario: Extract market data from PricingEnvironment

- **WHEN** pricing an Asian option
- **THEN** the engine SHALL obtain spot from pricing_env.spot
- **AND** SHALL obtain volatility from pricing_env.get_vol(K, T)
- **AND** SHALL obtain rate from pricing_env.get_rate(T)
- **AND** SHALL obtain dividend yield from pricing_env.get_div_yield(T)

#### Scenario: Use observation schedule from product

- **WHEN** the AsianOption has observation_records defined
- **THEN** the engine SHALL call product.resolve_observations(pricing_env)
- **AND** SHALL correctly handle past and future observations
