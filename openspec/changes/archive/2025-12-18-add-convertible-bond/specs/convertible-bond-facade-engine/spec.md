# convertible-bond-facade-engine Specification

## Purpose
Provides a unified `ConvertibleBondEngine` facade that dispatches pricing requests to the appropriate specialized engine (tree or PDE) based on method selection. This enables a consistent API for convertible bond pricing while allowing users to select the most appropriate numerical method for their use case.

## ADDED Requirements

### Requirement: Unified Engine Interface
The system SHALL provide a `ConvertibleBondEngine` class that implements a unified pricing interface and dispatches to specialized engines based on method selection.

#### Scenario: Pricing via facade engine
- **WHEN** `ConvertibleBondEngine.price(product, pricing_env)` is called
- **THEN** the engine delegates to the appropriate specialized engine and returns a `float` price

#### Scenario: Method-based dispatch to tree engine
- **WHEN** `ConvertibleBondEngine` is initialized with a tree method (BINOMIAL_GS, TRINOMIAL_HW)
- **THEN** pricing is delegated to `ConvertibleBondBinomialEngine` or `ConvertibleBondTrinomialEngine`

#### Scenario: Method-based dispatch to PDE engine
- **WHEN** `ConvertibleBondEngine` is initialized with a PDE method (JUMP_DIFFUSION, TF)
- **THEN** pricing is delegated to `ConvertibleBondJumpDiffusionEngine` or `ConvertibleBondTFEngine`

### Requirement: Method Selection Pattern
The system SHALL support method selection via the two-level enum pattern consistent with project conventions.

#### Scenario: Two-level enum method selection
- **WHEN** `ConvertibleBondEngine(method=EngineType.TREE(ConvertibleBondMethod.BINOMIAL_GS))` is created
- **THEN** the engine uses Goldman Sachs binomial tree method

#### Scenario: Direct enum method selection
- **WHEN** `ConvertibleBondEngine(method=ConvertibleBondMethod.TF)` is created
- **THEN** the engine uses Tsiveriotis-Fernandes PDE method

#### Scenario: String-based method selection (backward compatibility)
- **WHEN** `ConvertibleBondEngine(method="binomial_gs")` is created
- **THEN** the engine accepts the string and uses binomial GS method

#### Scenario: Default method
- **WHEN** `ConvertibleBondEngine()` is created without specifying method
- **THEN** the engine uses BINOMIAL_GS as the default method

### Requirement: ConvertibleBondMethod Enum
The system SHALL provide a `ConvertibleBondMethod` enum defining available pricing methods.

#### Scenario: Available methods
- **WHEN** `ConvertibleBondMethod` enum is accessed
- **THEN** it contains: BINOMIAL_GS, TRINOMIAL_HW, JUMP_DIFFUSION, TF

#### Scenario: Method descriptions
- **WHEN** `ConvertibleBondMethod.BINOMIAL_GS.value` is accessed
- **THEN** the value is "binomial_gs" for string-based lookup

### Requirement: Engine Parameter Passthrough
The system SHALL pass engine-specific parameters to the underlying specialized engines.

#### Scenario: Tree parameter passthrough
- **WHEN** `ConvertibleBondEngine(method=ConvertibleBondMethod.BINOMIAL_GS, num_steps=200)` is created
- **THEN** the underlying tree engine is initialized with 200 steps

#### Scenario: PDE parameter passthrough
- **WHEN** `ConvertibleBondEngine(method=ConvertibleBondMethod.TF, num_space_steps=150, scheme="implicit_euler")` is created
- **THEN** the underlying PDE engine is initialized with the specified parameters

#### Scenario: Invalid parameter for method
- **WHEN** a PDE-specific parameter is passed with a tree method
- **THEN** the parameter is ignored (or warning logged) and pricing proceeds

### Requirement: Greeks Calculation
The system SHALL provide Greeks calculation that delegates to the underlying engine.

#### Scenario: Delta calculation via facade
- **WHEN** `ConvertibleBondEngine.calculate_delta(product, pricing_env)` is called
- **THEN** the delta is computed by the underlying engine and returned

#### Scenario: Numerical Greeks via bump-and-reprice
- **WHEN** `calculate_numerical_greeks()` is called
- **THEN** Greeks are computed using finite difference bumps on the facade engine

#### Scenario: Greeks method consistency
- **WHEN** Greeks are calculated with different underlying methods
- **THEN** results are numerically consistent (within expected method tolerance)

### Requirement: Results Container
The system SHALL support a structured results container with price and optional additional outputs via `price_with_details()`.

#### Scenario: Basic result
- **WHEN** `price_with_details()` returns
- **THEN** a `ConvertibleBondResult` containing at least `price` attribute is returned

#### Scenario: Extended results
- **WHEN** `price_with_details()` is called
- **THEN** the result includes price, delta, conversion probability, and method-specific details

#### Scenario: Decomposition results (TF method)
- **WHEN** using TF method with `return_decomposition=True`
- **THEN** the result includes both total value and cash-only component value

### Requirement: Error Handling
The system SHALL provide clear error messages for invalid inputs and method selection.

#### Scenario: Invalid method string
- **WHEN** `ConvertibleBondEngine(method="invalid_method")` is created
- **THEN** a `ValidationError` is raised listing valid methods

#### Scenario: Unsupported product type
- **WHEN** `price()` is called with a non-ConvertibleBond product
- **THEN** a `ValidationError` is raised indicating ConvertibleBond is required

#### Scenario: Missing required market data
- **WHEN** `price()` is called without required market data in pricing_env
- **THEN** a `MarketDataError` is raised with specific missing field

### Requirement: PricingEnvironment Integration
The system SHALL accept `PricingEnvironment` as the standard market data container.

#### Scenario: Standard PricingEnvironment usage
- **WHEN** `price(product, pricing_env)` is called with a standard PricingEnvironment
- **THEN** spot, rate curve, and vol surface are extracted and used for pricing

#### Scenario: Credit data from product
- **WHEN** pricing_env lacks credit data but product has credit_spread or hazard_rate
- **THEN** the credit parameters from the product are used

### Requirement: Engine Representation
The system SHALL provide informative string representation for debugging.

#### Scenario: String representation
- **WHEN** `repr(engine)` is called
- **THEN** a string showing engine type and method is returned, e.g., "ConvertibleBondEngine(method=BINOMIAL_GS)"

### Requirement: Consistency with Existing Patterns
The system SHALL follow existing engine patterns for compatibility with portfolio, backtest, and stresstest modules.

#### Scenario: BaseEngine-like interface
- **WHEN** `ConvertibleBondEngine` is used
- **THEN** it provides `price(product, pricing_env)` method signature consistent with equity engines

#### Scenario: Integration with GreeksCalculator
- **WHEN** `GreeksCalculator.calculate_numerical_greeks(engine, product, pricing_env)` is called
- **THEN** the calculator works correctly with ConvertibleBondEngine

#### Scenario: Portfolio pricing compatibility
- **WHEN** a portfolio contains ConvertibleBond products
- **THEN** `ConvertibleBondEngine` can be used to price them in batch
