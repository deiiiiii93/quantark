# convertible-bond-facade-engine Specification

## Purpose
TBD - created by archiving change add-convertible-bond. Update Purpose after archive.
## Requirements
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

#### Scenario: Extended results with risk metrics
- **WHEN** `price_with_details()` is called
- **THEN** the result includes `floor_bond_price`, `floor_bond_dv01`, `floor_bond_cs01`, `floor_bond_duration`, `floor_bond_convexity`, `dv01`, `cs01`, `modified_duration`, and `convexity` fields

#### Scenario: Risk metrics populated on demand
- **WHEN** `price_with_details(include_risk_metrics=True)` is called (default)
- **THEN** all risk metrics are computed and populated in the result

#### Scenario: Skip risk metrics for performance
- **WHEN** `price_with_details(include_risk_metrics=False)` is called
- **THEN** risk metrics fields are set to 0.0 to avoid computational overhead

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

### Requirement: Floor Bond Price Calculation
The system SHALL provide a method to calculate the floor bond (straight bond) price for a convertible bond.

#### Scenario: Floor bond price calculation
- **GIVEN** a `ConvertibleBond` with coupon schedule, maturity, and credit spread
- **WHEN** `ConvertibleBondEngine.floor_bond_price(bond)` is called
- **THEN** the engine returns the present value of all bond cashflows discounted at the risky rate (risk-free + credit spread), ignoring conversion and call/put options

#### Scenario: Floor bond price with zero credit spread
- **GIVEN** a `ConvertibleBond` with `credit_spread=0.0`
- **WHEN** `floor_bond_price(bond)` is called
- **THEN** the floor bond is discounted using only the risk-free rate

#### Scenario: Floor bond price as lower bound
- **GIVEN** a `ConvertibleBond` with embedded options
- **WHEN** both `price(bond)` and `floor_bond_price(bond)` are called
- **THEN** `price >= floor_bond_price` (convertible is worth at least its floor value)

### Requirement: Floor Bond Risk Metrics
The system SHALL provide DV01, CS01, modified duration, and convexity calculations for the floor bond.

#### Scenario: Floor bond DV01 calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.floor_bond_dv01(bond)` is called
- **THEN** the engine returns the floor bond's price change per basis point rate move

#### Scenario: Floor bond CS01 calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.floor_bond_cs01(bond)` is called
- **THEN** the engine returns the floor bond's price change per basis point credit spread move

#### Scenario: Floor bond CS01 equals DV01
- **GIVEN** a `ConvertibleBond` with credit spread
- **WHEN** both `floor_bond_dv01(bond)` and `floor_bond_cs01(bond)` are called
- **THEN** the values are equal (since floor bond discounts at r+s, both have identical effect)

#### Scenario: Floor bond duration calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.floor_bond_duration(bond)` is called
- **THEN** the engine returns the floor bond's modified duration (weighted average time of cashflows)

#### Scenario: Floor bond convexity calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.floor_bond_convexity(bond)` is called
- **THEN** the engine returns the floor bond's convexity (weighted average time squared of cashflows)

#### Scenario: Floor bond metrics consistency
- **GIVEN** floor bond price, DV01, and duration
- **WHEN** metrics are computed
- **THEN** `floor_bond_dv01 ≈ floor_bond_duration × floor_bond_price × 0.0001` (within numerical tolerance)

### Requirement: Convertible Bond Interest Rate and Credit Risk Metrics
The system SHALL provide numerical DV01, CS01, duration, and convexity calculations for the full convertible bond.

#### Scenario: Convertible DV01 calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.dv01(bond)` is called
- **THEN** the engine returns the convertible's price change per basis point rate move, computed via numerical rate bumping

#### Scenario: Convertible CS01 calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.cs01(bond)` is called
- **THEN** the engine returns the convertible's price change per basis point credit spread move, computed via numerical spread bumping

#### Scenario: Convertible duration calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.modified_duration(bond)` is called
- **THEN** the engine returns the convertible's modified duration derived from DV01

#### Scenario: Convertible convexity calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.convexity(bond)` is called
- **THEN** the engine returns the convertible's convexity via central difference rate bumping

#### Scenario: Interest rate bump isolation
- **WHEN** computing convertible DV01
- **THEN** only the risk-free rate is bumped, not the credit spread, isolating interest rate risk

#### Scenario: Credit spread bump isolation
- **WHEN** computing convertible CS01
- **THEN** only the credit spread is bumped, not the risk-free rate, isolating credit risk

### Requirement: Extended results
`ConvertibleBondEngine.price_with_details()` SHALL include a `conversion_probability` that is mathematically consistent with the underlying method:
- Tree methods: computed directly from the lattice optimal policy
- PDE methods: computed from an auxiliary PDE for the conversion event indicator under the same optimal policy constraints

#### Scenario: PDE method probability is not a heuristic
- **WHEN** `ConvertibleBondEngine(method=ConvertibleBondMethod.TF)` or `ConvertibleBondMethod.JUMP_DIFFUSION` is used
- **THEN** `conversion_probability` is produced by the PDE engine and propagated through the facade result

