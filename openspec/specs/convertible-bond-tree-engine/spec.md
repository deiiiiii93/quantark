# convertible-bond-tree-engine Specification

## Purpose
TBD - created by archiving change add-convertible-bond. Update Purpose after archive.
## Requirements
### Requirement: Binomial Tree Engine (Goldman Sachs Model)
The system SHALL provide a `ConvertibleBondBinomialEngine` implementing the Goldman Sachs credit-adjusted binomial tree model for pricing convertible bonds.

#### Scenario: Basic pricing with binomial tree
- **WHEN** `price()` is called on a `ConvertibleBond` with `ConvertibleBondBinomialEngine`
- **THEN** the engine returns the convertible bond price using credit-adjusted discounting

#### Scenario: Credit-adjusted discount rate
- **WHEN** pricing with the binomial engine and credit_spread is specified
- **THEN** the discount rate at each node is `y = p*r + (1-p)*d` where p is conversion probability, r is risk-free rate, d is risky rate

#### Scenario: Conversion probability tracking
- **WHEN** the binomial tree is constructed
- **THEN** at each node, the probability of eventual conversion is computed and used for credit adjustment

#### Scenario: American-style conversion
- **WHEN** conversion is allowed and stock price exceeds conversion price
- **THEN** the holder's option to convert is evaluated at each tree node

#### Scenario: Call forcing conversion
- **WHEN** the issuer calls the bond and parity exceeds call price
- **THEN** the rational holder converts to stock rather than accepting the call

### Requirement: Trinomial Tree Engine (Hull-White Model)
The system SHALL provide a `ConvertibleBondTrinomialEngine` implementing the Hull-White trinomial tree model with explicit default probability at each node and an explicit volatility scheme selection.

#### Scenario: Scheme selection for trinomial engine
- **WHEN** `ConvertibleBondTrinomialEngine` is initialized with a specific trinomial volatility scheme
- **THEN** the engine prices using the selected scheme and exposes it in its configuration

### Requirement: Tree Configuration
The system SHALL accept tree configuration parameters for grid resolution and time stepping.

#### Scenario: Number of time steps
- **WHEN** `ConvertibleBondBinomialEngine` or `ConvertibleBondTrinomialEngine` is initialized with `num_steps=200`
- **THEN** the tree has 200 time steps from valuation to maturity

#### Scenario: Default number of steps
- **WHEN** a tree engine is initialized without specifying num_steps
- **THEN** the default of 100 time steps is used

#### Scenario: Volatility specification
- **WHEN** pricing a convertible bond with stock_volatility=0.30
- **THEN** the tree is calibrated to 30% annual volatility of the underlying stock

### Requirement: Early Exercise Handling
The system SHALL correctly handle American-style conversion, callable, and putable features at each tree node.

#### Scenario: Conversion decision
- **WHEN** at a tree node where conversion is allowed
- **THEN** bond value is `max(holding_value, parity)`

#### Scenario: Call decision
- **WHEN** at a tree node where the bond is callable and holding_value > call_price
- **THEN** the issuer calls, and holder chooses `max(parity, call_price)`

#### Scenario: Put decision
- **WHEN** at a tree node where the bond is putable and holding_value < put_price
- **THEN** bond value is `max(holding_value, put_price)`

#### Scenario: Combined provisions
- **WHEN** at a tree node with all provisions active
- **THEN** the value is `max(parity, put_price, min(holding_value, max(call_price, parity)))`

### Requirement: Coupon Handling
The system SHALL incorporate coupon payments at the appropriate tree nodes.

#### Scenario: Coupon at tree node
- **WHEN** a coupon payment date falls within a time step
- **THEN** the coupon amount is added to the holding value at nodes within that step

#### Scenario: Accrued interest on conversion
- **WHEN** the holder converts between coupon dates
- **THEN** accrued interest is forfeited (conversion value is parity only)

#### Scenario: Accrued interest on call
- **WHEN** the issuer calls and holder does not convert
- **THEN** call price includes accrued interest

### Requirement: Discrete Dividend Handling
The system SHALL handle discrete dividend payments on the underlying stock.

#### Scenario: Discrete dividend in tree
- **WHEN** a discrete dividend payment occurs during the tree lifetime
- **THEN** the stock price is reduced by the dividend amount at that time step

#### Scenario: Adjustment of tree around dividend
- **WHEN** constructing tree with discrete dividends
- **THEN** the tree recombines correctly after dividend adjustment

### Requirement: Output Results
The system SHALL return pricing results including price, delta, and optionally the full tree for analysis.

#### Scenario: Price output
- **WHEN** `price()` is called
- **THEN** the convertible bond dirty price is returned

#### Scenario: Delta output
- **WHEN** `calculate_delta()` is called
- **THEN** the sensitivity of bond price to stock price is returned

#### Scenario: Detailed results (optional)
- **WHEN** `price_with_details()` is called
- **THEN** method-specific details (e.g., conversion probabilities) MAY be returned alongside the price

#### Scenario: Full tree output (optional)
- **WHEN** `price_with_details(return_tree=True)` is called
- **THEN** the full tree structure with values at each node is returned for analysis

### Requirement: PricingEnvironment Integration
The system SHALL accept `PricingEnvironment` for market data input.

#### Scenario: Market data from PricingEnvironment
- **WHEN** `price(product, pricing_env)` is called
- **THEN** spot price, risk-free rate, and volatility are extracted from pricing_env

#### Scenario: Credit data override
- **WHEN** pricing_env does not contain credit data
- **THEN** credit parameters from the product (credit_spread, hazard_rate) are used

### Requirement: Validation and Error Handling
The system SHALL validate inputs and provide clear error messages.

#### Scenario: Missing volatility
- **WHEN** pricing without volatility in pricing_env or product
- **THEN** a `MarketDataError` is raised indicating volatility is required

#### Scenario: Invalid tree parameters
- **WHEN** num_steps <= 0
- **THEN** a `ValidationError` is raised indicating num_steps must be positive

#### Scenario: Product type validation
- **WHEN** a non-ConvertibleBond product is passed to the engine
- **THEN** a `ValidationError` is raised indicating unsupported product type

### Requirement: Trinomial Volatility Schemes
The system SHALL support multiple volatility schemes for the trinomial convertible bond tree.

#### Scenario: Constant-volatility trinomial scheme
- **WHEN** the constant-volatility scheme is selected
- **THEN** the tree uses a CRR-style fixed volatility grid and does not apply term-structure volatility

#### Scenario: Fixed-dx log-price scheme
- **WHEN** the fixed-dx log-price scheme is selected
- **THEN** the tree uses a constant log step and per-step probabilities that match the step-local variance

#### Scenario: Variable-dx log-price scheme with re-gridding
- **WHEN** the variable-dx log-price scheme is selected
- **THEN** the tree recomputes the log step per time interval and re-grids values to maintain a recombining lattice

### Requirement: Time-Dependent Parameters in Trinomial Tree
The system SHALL query interest rates and volatility at each time step during trinomial tree backward induction, using maximum volatility for grid stability.

#### Scenario: Maximum volatility for grid spacing
- **GIVEN** a `ConvertibleBondTrinomialEngine` pricing a convertible bond
- **WHEN** the trinomial tree grid is constructed
- **THEN** the engine calculates the maximum volatility over the bond's life and uses it for the vertical grid spacing ($dx$)

#### Scenario: Local forward rate per time step
- **GIVEN** a `ConvertibleBondTrinomialEngine` performing backward induction
- **WHEN** processing time step $i$ corresponding to time $t = i \cdot \Delta t$
- **THEN** the engine uses `rate_curve.get_forward_rate(t, t + dt)` for the local interest rate

#### Scenario: Local volatility per time step
- **GIVEN** a `ConvertibleBondTrinomialEngine` performing backward induction
- **WHEN** processing time step $i$ corresponding to time $t$
- **THEN** the engine derives a per-step effective volatility `sigma_step(t, t+dt)` from implied vols via `pricing_env.get_vol(strike, time_to_maturity)` using total variance differences

#### Scenario: Transition probabilities recalculated per step
- **GIVEN** a trinomial tree with time-varying parameters
- **WHEN** backward induction processes each time step
- **THEN** the transition probabilities $(p_u, p_m, p_d)$ are recalculated using local rate and volatility

#### Scenario: Non-flat rate curve produces different price
- **GIVEN** a convertible bond with 4-year maturity
- **AND** a stepped rate curve: 1% for years 0-2, 9% for years 2-4
- **WHEN** pricing with `ConvertibleBondTrinomialEngine`
- **THEN** the price differs from a flat 5% curve by more than 0.1% of face value

### Requirement: Binomial Engine Non-Flat Curve Warning
The system SHALL warn users when the binomial engine is used with non-flat rate curves or volatility surfaces.

#### Scenario: Warning logged for non-flat rate curve
- **GIVEN** a `ConvertibleBondBinomialEngine`
- **AND** a `PricingEnvironment` with `InterpolatedRateCurve` (non-flat)
- **WHEN** `price()` or `price_with_details()` is called
- **THEN** a warning is logged: "Binomial GS engine approximates piecewise curves using a flat rate/vol to maturity. Use PDE or Trinomial engines for better accuracy."

#### Scenario: Warning logged for non-flat volatility surface
- **GIVEN** a `ConvertibleBondBinomialEngine`
- **AND** a `PricingEnvironment` with a non-`FlatVolSurface` volatility surface
- **WHEN** `price()` or `price_with_details()` is called
- **THEN** a warning is logged recommending PDE or Trinomial engines

#### Scenario: No warning for flat curves
- **GIVEN** a `ConvertibleBondBinomialEngine`
- **AND** a `PricingEnvironment` with `FlatRateCurve` and `FlatVolSurface`
- **WHEN** `price()` is called
- **THEN** no warning about curve approximation is logged

#### Scenario: Binomial engine unchanged mathematically
- **GIVEN** a `ConvertibleBondBinomialEngine` with flat curves
- **WHEN** `price()` is called before and after this change
- **THEN** the prices are identical (no mathematical changes)

