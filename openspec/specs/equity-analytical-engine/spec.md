# equity-analytical-engine Specification

## Purpose
TBD - created by archiving change add-american-option-analytical-engine. Update Purpose after archive.
## Requirements
### Requirement: American Option Analytical Pricing with Multiple Methods

The system SHALL provide an analytical pricing engine for American vanilla options supporting three approximation methods: BS93, BS02, and BAW.

#### Scenario: Price American call with BS93 method

- **WHEN** user creates an `AmericanOption` with `option_type=CALL` and selects BS93 method
- **THEN** the engine SHALL compute price using Bjerksund-Stensland (1993) single-barrier approximation
- **AND** the price SHALL be at least the European call price
- **AND** the price SHALL be at least the intrinsic value

#### Scenario: Price American call with BS02 method

- **WHEN** user selects BS02 method for American call pricing
- **THEN** the engine SHALL use Bjerksund-Stensland (2002) two-barrier approximation with bivariate normal CDF
- **AND** SHALL calculate dual trigger prices I1 and I2
- **AND** the price SHALL be more accurate than BS93 for most cases

#### Scenario: Price American call with BAW method

- **WHEN** user selects BAW method for American call pricing
- **THEN** the engine SHALL use Barone-Adesi-Whaley (1987) quadratic approximation
- **AND** SHALL iteratively find critical stock price S* using Newton-Raphson
- **AND** the price SHALL be at least the European call price

#### Scenario: Price American put via put-call transformation (BS93/BS02)

- **WHEN** pricing American put with BS93 or BS02 method
- **THEN** the engine SHALL apply put-call transformation (swap S and K, transform r and b)
- **AND** SHALL price using American call formula on transformed parameters
- **AND** SHALL restore original parameters after computation

#### Scenario: Price American put directly (BAW)

- **WHEN** pricing American put with BAW method
- **THEN** the engine SHALL use direct put pricing formula (not transformation)
- **AND** SHALL find critical price S** below which early exercise is optimal
- **AND** the price SHALL be at least the European put price

#### Scenario: Method selection via EngineParams

- **WHEN** user creates engine with EngineParams specifying method="BS02"
- **THEN** the engine SHALL use BS02 for all American option pricing
- **WHEN** no method is specified
- **THEN** the engine SHALL default to BS93 method

#### Scenario: Handle zero dividend call options

- **WHEN** pricing American call with dividend yield = 0 and risk-free rate >= 0
- **THEN** the engine SHALL use European option pricing for all methods (early exercise never optimal)
- **AND** the price SHALL equal the European call price

### Requirement: BS93 Method Implementation

The engine SHALL implement Bjerksund-Stensland (1993) approximation with single-barrier approach.

#### Scenario: Calculate phi auxiliary function (BS93)

- **WHEN** computing American call price with BS93
- **THEN** the engine SHALL implement φ(γ, H, I) function
- **AND** SHALL use formula: exp(λT) * S^γ * [N(d) - (I/S)^κ * N(d₂)]
- **AND** SHALL calculate λ = -r + γb + 0.5γ(γ-1)σ², κ = 2b/σ² + 2γ - 1

#### Scenario: Compute exercise boundary (BS93)

- **WHEN** applying BS93 method
- **THEN** SHALL calculate beta = (0.5 - b/σ²) + √[(0.5 - b/σ²)² + 2r/σ²]
- **AND** SHALL calculate B_0 = max(K, rK/(r-b)), B_∞ = βK/(β-1)
- **AND** SHALL calculate trigger price I = B_0 + (B_∞ - B_0)(1 - exp(h_τ))
- **AND** SHALL return S - K if S >= I (immediate exercise)

### Requirement: BS02 Method Implementation

The engine SHALL implement Bjerksund-Stensland (2002) approximation with improved two-barrier approach.

#### Scenario: Calculate phi auxiliary function (BS02)

- **WHEN** computing American call price with BS02
- **THEN** the engine SHALL implement enhanced φ(S, T, γ, H, I) function
- **AND** SHALL use formula: e^(λT) * S^γ * [N(-d) - (I/S)^κ * N(-d₂)]

#### Scenario: Calculate psi auxiliary function (BS02)

- **WHEN** using BS02 method
- **THEN** SHALL implement Ψ(S, T, γ, H, I₂, I₁, t₁) for second barrier
- **AND** SHALL use bivariate normal CDF M(x, y, ρ) for correlation handling
- **AND** SHALL calculate four bivariate terms with correlation ρ = ±√(t₁/T)

#### Scenario: Compute dual trigger prices (BS02)

- **WHEN** pricing with BS02
- **THEN** SHALL calculate t₁ = 0.5(√5 - 1)T (golden ratio approximation)
- **AND** SHALL calculate h₁ and h₂ for dual barriers
- **AND** SHALL calculate I₁ = B_0 + (B_∞ - B_0)(1 - exp(h₁))
- **AND** SHALL calculate I₂ = B_0 + (B_∞ - B_0)(1 - exp(h₂))
- **AND** SHALL return S - K if S >= I₂ (immediate exercise)

#### Scenario: Robust alpha calculation (BS02)

- **WHEN** calculating α₁ and α₂ coefficients in BS02
- **THEN** SHALL use log-space calculation: log(α) = ln(I - K) - β*ln(I)
- **AND** SHALL check for underflow (log(α) < -700) and set α = 0 if true
- **AND** SHALL handle I <= K case by setting α = 0

### Requirement: BAW Method Implementation

The engine SHALL implement Barone-Adesi-Whaley (1987) quadratic approximation method.

#### Scenario: Calculate BAW parameters

- **WHEN** using BAW method
- **THEN** SHALL calculate M = 2r/σ², N = 2b/σ², K = 1 - exp(-rT)
- **AND** SHALL calculate discriminant = (N-1)² + 4M/K
- **AND** SHALL fallback to European pricing if discriminant < 0
- **AND** SHALL calculate q₂ = (-(N-1) + √discriminant)/2 for calls
- **AND** SHALL calculate q₁ = (-(N-1) - √discriminant)/2 for puts

#### Scenario: Find critical price iteratively (BAW)

- **WHEN** pricing American call with BAW
- **THEN** SHALL find critical price S* where S* - K - c(S*) + (S*/q₂)(1 - e^((b-r)T)N(d₁)) = 0
- **AND** SHALL use Newton-Raphson iteration with initial guess S* = K
- **AND** SHALL converge to tolerance of 1e-6 or within 100 iterations

#### Scenario: Calculate American premium (BAW)

- **WHEN** spot S < S* (no early exercise)
- **THEN** SHALL calculate European call c_BSM
- **AND** SHALL calculate A₂ = (S*/q₂)(1 - e^((b-r)T)N(d₁(S*)))
- **AND** SHALL return C = c_BSM + A₂(S/S*)^q₂

#### Scenario: Handle early exercise condition (BAW)

- **WHEN** spot S >= S* for calls or S <= S** for puts
- **THEN** SHALL return intrinsic value directly (S - K for calls, K - S for puts)

### Requirement: Integration with Existing Architecture

The `AmericanOptionAnalyticalEngine` SHALL integrate seamlessly with QuantArk's existing architecture.

#### Scenario: Inherit from BaseEngine

- **WHEN** creating the engine class
- **THEN** it SHALL inherit from `BaseEngine`
- **AND** SHALL implement the `price(product, pricing_env)` method
- **AND** SHALL accept optional `EngineParams` in constructor with method selection

#### Scenario: Use PricingEnvironment for market data

- **WHEN** pricing an American option
- **THEN** the engine SHALL extract spot, volatility, rates, and dividends from `PricingEnvironment`
- **AND** SHALL use `pricing_env.spot`, `pricing_env.get_vol()`, `pricing_env.get_rate()`, `pricing_env.get_div_yield()`
- **AND** SHALL calculate cost-of-carry b = r - q

#### Scenario: Raise appropriate exceptions

- **WHEN** invalid inputs are provided (negative spot, strike, volatility)
- **THEN** the engine SHALL raise `ValidationError`
- **WHEN** numerical computation fails
- **THEN** the engine SHALL raise `NumericalError`
- **WHEN** unsupported product type is provided
- **THEN** the engine SHALL raise `PricingError`
- **WHEN** invalid method is specified
- **THEN** the engine SHALL raise `ValidationError` with available methods

### Requirement: Numerical Stability and Edge Cases

The engine SHALL handle edge cases and maintain numerical stability across all parameter ranges for all three methods.

#### Scenario: Safe mathematical operations

- **WHEN** performing mathematical operations (log, sqrt, exp)
- **THEN** the engine SHALL use safe wrappers preventing log(0), sqrt(negative), exp(overflow)
- **AND** SHALL clamp volatility to range [0.001, 5.0]
- **AND** SHALL clamp maturity to range [1e-6, 30.0]

#### Scenario: Fallback to European pricing

- **WHEN** numerical instability is detected (NaN, Inf results, negative discriminant)
- **THEN** the engine SHALL fallback to European option pricing
- **AND** SHALL use `BlackScholesEngine` or internal European formula
- **AND** SHALL emit a warning about using fallback method

#### Scenario: Handle near-expiry options

- **WHEN** time to maturity < 1e-10
- **THEN** the engine SHALL return intrinsic value directly for all methods
- **AND** SHALL NOT attempt numerical computation

#### Scenario: Handle negative interest rates

- **WHEN** risk-free rate is negative
- **THEN** BS93/BS02 SHALL apply special case logic for American puts
- **AND** SHALL use European pricing when r <= 0 and r <= dividend yield
- **WHEN** using BAW with negative rates
- **THEN** SHALL check early exercise conditions appropriately

#### Scenario: Bivariate normal CDF computation (BS02)

- **WHEN** computing bivariate normal CDF M(x, y, ρ)
- **THEN** SHALL use scipy.stats.multivariate_normal for accurate computation
- **AND** SHALL handle edge cases: |ρ| < 1e-10 → N(x)*N(y), ρ ≈ 1 → min(N(x), N(y))
- **AND** SHALL handle ρ ≈ -1 → max(N(x) + N(y) - 1, 0)

