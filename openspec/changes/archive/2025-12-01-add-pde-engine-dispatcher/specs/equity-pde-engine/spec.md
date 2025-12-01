# Equity PDE Engine Specification

## ADDED Requirements

### Requirement: Unified PDE Engine Interface
The system SHALL provide a unified `PDEEngine` class that implements the `BaseEngine` interface and automatically dispatches pricing requests to product-specific PDE solvers based on the product type.

#### Scenario: European option pricing via PDEEngine
- **WHEN** a `EuropeanVanillaOption` is priced using `PDEEngine`
- **THEN** the engine delegates to `EuropeanPDESolver` and returns the option price

#### Scenario: American option pricing via PDEEngine
- **WHEN** an `AmericanOption` is priced using `PDEEngine`
- **THEN** the engine delegates to `AmericanPDESolver` and returns the option price

#### Scenario: Barrier option pricing via PDEEngine
- **WHEN** a `BarrierOption` is priced using `PDEEngine`
- **THEN** the engine delegates to `BarrierPDESolver` and returns the option price

#### Scenario: Double barrier option pricing via PDEEngine
- **WHEN** a `DoubleBarrierOption` is priced using `PDEEngine`
- **THEN** the engine delegates to `DoubleBarrierPDESolver` and returns the option price

#### Scenario: One-touch option pricing via PDEEngine
- **WHEN** an `OneTouchOption` is priced using `PDEEngine`
- **THEN** the engine delegates to `OneTouchPDESolver` and returns the option price

#### Scenario: Double one-touch option pricing via PDEEngine
- **WHEN** a `DoubleOneTouchOption` is priced using `PDEEngine`
- **THEN** the engine delegates to `DoubleOneTouchPDESolver` and returns the option price

#### Scenario: Unsupported product type
- **WHEN** an unsupported product type is priced using `PDEEngine`
- **THEN** a `ValidationError` is raised with a message listing supported product types

### Requirement: PDE Method Selection
The system SHALL support PDE method selection via a two-level enum pattern `EngineType.PDE(PDEMethod.METHOD_NAME)` consistent with the project's analytical engine pattern.

#### Scenario: Crank-Nicolson method selection
- **WHEN** `PDEEngine` is initialized with `method=EngineType.PDE(PDEMethod.CRANK_NICOLSON)`
- **THEN** the underlying PDE solver uses the Crank-Nicolson scheme

#### Scenario: Explicit Euler method selection
- **WHEN** `PDEEngine` is initialized with `method=EngineType.PDE(PDEMethod.EXPLICIT_EULER)`
- **THEN** the underlying PDE solver uses the explicit Euler scheme

#### Scenario: Implicit Euler method selection
- **WHEN** `PDEEngine` is initialized with `method=EngineType.PDE(PDEMethod.IMPLICIT_EULER)`
- **THEN** the underlying PDE solver uses the implicit Euler scheme

#### Scenario: Default method
- **WHEN** `PDEEngine` is initialized without specifying a method
- **THEN** the engine uses Crank-Nicolson as the default method

#### Scenario: Direct method enum
- **WHEN** `PDEEngine` is initialized with `method=PDEMethod.CRANK_NICOLSON`
- **THEN** the engine accepts the method and uses Crank-Nicolson scheme

#### Scenario: String-based method (backward compatibility)
- **WHEN** `PDEEngine` is initialized with `method="crank_nicolson"`
- **THEN** the engine accepts the string and uses Crank-Nicolson scheme

### Requirement: Greeks Calculation Integration
The system SHALL enable `GreeksCalculator.calculate_numerical_greeks()` to work seamlessly with `PDEEngine` for all supported product types.

#### Scenario: Delta calculation via numerical Greeks
- **WHEN** `GreeksCalculator.calculate_numerical_greeks()` is called with a `PDEEngine` and a `EuropeanVanillaOption`
- **THEN** the calculator returns accurate delta by bumping spot price and repricing via PDE

#### Scenario: Gamma calculation via numerical Greeks
- **WHEN** `GreeksCalculator.calculate_numerical_greeks()` is called with a `PDEEngine` and an `AmericanOption`
- **THEN** the calculator returns accurate gamma using second-order finite differences via PDE pricing

#### Scenario: Vega calculation via numerical Greeks
- **WHEN** `GreeksCalculator.calculate_numerical_greeks()` is called with a `PDEEngine` and a `BarrierOption`
- **THEN** the calculator returns accurate vega by bumping volatility and repricing via PDE

### Requirement: Parameter Configuration
The system SHALL accept `PDEParams` for configuring grid resolution, time stepping scheme, and numerical stability features (Rannacher smoothing).

#### Scenario: Custom grid configuration
- **WHEN** `PDEEngine` is initialized with `PDEParams(num_space_steps=500, num_time_steps=200)`
- **THEN** the underlying PDE solvers use the specified grid resolution

#### Scenario: Scheme selection via params
- **WHEN** `PDEEngine` is initialized with `PDEParams(scheme="implicit_euler")`
- **THEN** the underlying PDE solvers use implicit Euler time stepping

#### Scenario: Rannacher smoothing control
- **WHEN** `PDEEngine` is initialized with `PDEParams(rannacher_steps=4)`
- **THEN** the underlying PDE solvers apply 4 steps of implicit Euler for smoothing near maturity

### Requirement: Error Handling and Validation
The system SHALL validate product compatibility and provide clear error messages when products are not supported by PDE methods.

#### Scenario: Invalid product type error
- **WHEN** a product not in the supported list (e.g., a swap or bond) is priced with `PDEEngine`
- **THEN** a `ValidationError` is raised with message: "PDEEngine does not support product type {type}. Supported types: EuropeanVanillaOption, AmericanOption, BarrierOption, DoubleBarrierOption, OneTouchOption, DoubleOneTouchOption"

#### Scenario: Null product
- **WHEN** `PDEEngine.price()` is called with `product=None`
- **THEN** a `ValidationError` is raised indicating product cannot be None

### Requirement: Numerical Consistency
The system SHALL produce prices via `PDEEngine.price()` that are numerically consistent (within tolerance) with direct calls to the corresponding PDE solver.

#### Scenario: European option price consistency
- **WHEN** an `EuropeanVanillaOption` is priced using `PDEEngine` and directly via `EuropeanPDESolver` with identical parameters
- **THEN** the prices agree within numerical tolerance (1e-6 relative error)

#### Scenario: American option price consistency
- **WHEN** an `AmericanOption` is priced using `PDEEngine` and directly via `AmericanPDESolver` with identical parameters
- **THEN** the prices agree within numerical tolerance (1e-6 relative error)

#### Scenario: Barrier option price consistency
- **WHEN** a `BarrierOption` is priced using `PDEEngine` and directly via `BarrierPDESolver` with identical parameters
- **THEN** the prices agree within numerical tolerance (1e-6 relative error)
