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
The system SHALL provide a `ConvertibleBondTrinomialEngine` implementing the Hull-White trinomial tree model with explicit default probability at each node.

#### Scenario: Basic pricing with trinomial tree
- **WHEN** `price()` is called on a `ConvertibleBond` with `ConvertibleBondTrinomialEngine`
- **THEN** the engine returns the convertible bond price using trinomial tree with default branch

#### Scenario: Three-branch model with default
- **WHEN** the trinomial tree is constructed at each node
- **THEN** there are three branches: up move, down move, and default (stock to zero, bond to recovery)

#### Scenario: Default probability at each step
- **WHEN** hazard_rate λ is specified
- **THEN** probability of default in each time step Δt is `1 - exp(-λ*Δt)`

#### Scenario: Recovery on default
- **WHEN** default occurs at a node
- **THEN** the bond value jumps to recovery_rate * face_value

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

