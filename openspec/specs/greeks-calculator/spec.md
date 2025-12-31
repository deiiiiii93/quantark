# greeks-calculator Specification

## Purpose
TBD - created by archiving change add-pde-greeks-mode. Update Purpose after archive.
## Requirements
### Requirement: Greeks Calculation Mode Enum

The system SHALL provide a `GreeksCalculationMode` enum with three values:

- `BUMP`: Always use finite difference bump method
- `ENGINE`: Use engine's `calculate_greeks()` if available
- `AUTO`: Use engine method for PDE engines, bump otherwise

#### Scenario: Default mode is BUMP
- **GIVEN** a new `GreeksCalculator()`
- **WHEN** accessing `greeks_mode` attribute
- **THEN** value is `GreeksCalculationMode.BUMP`

### Requirement: GreeksCalculator greeks_mode Parameter

The system SHALL accept a `greeks_mode` parameter in `GreeksCalculator.__init__()`.

#### Scenario: Create calculator with AUTO mode
- **GIVEN** `GreeksCalculationMode.AUTO` is defined
- **WHEN** creating `GreeksCalculator(greeks_mode=GreeksCalculationMode.AUTO)`
- **THEN** calculator is created successfully
- **AND** `calc.greeks_mode == GreeksCalculationMode.AUTO`

#### Scenario: Backward compatibility - no mode specified
- **GIVEN** existing code that creates `GreeksCalculator()`
- **WHEN** calculator is created without `greeks_mode` parameter
- **THEN** calculator defaults to `BUMP` mode
- **AND** existing behavior is preserved

### Requirement: Engine Greeks Detection

The system SHALL provide a `_should_use_engine_greeks()` method that determines whether to use the engine's `calculate_greeks()` method.

#### Scenario: BUMP mode never uses engine method
- **GIVEN** calculator with `greeks_mode = GreeksCalculationMode.BUMP`
- **AND** a PDE engine with `calculate_greeks()` method
- **WHEN** calling `_should_use_engine_greeks(pde_engine)`
- **THEN** returns `False`

#### Scenario: ENGINE mode always uses engine method
- **GIVEN** calculator with `greeks_mode = GreeksCalculationMode.ENGINE`
- **AND** any engine with `calculate_greeks()` method
- **WHEN** calling `_should_use_engine_greeks(engine)`
- **THEN** returns `True`

#### Scenario: AUTO mode uses engine for PDE
- **GIVEN** calculator with `greeks_mode = GreeksCalculationMode.AUTO`
- **AND** a PDE engine with `engine_type = EngineType.PDE`
- **WHEN** calling `_should_use_engine_greeks(pde_engine)`
- **THEN** returns `True`

#### Scenario: AUTO mode uses bump for analytical
- **GIVEN** calculator with `greeks_mode = GreeksCalculationMode.AUTO`
- **AND** an analytical engine with `engine_type = EngineType.ANALYTICAL`
- **WHEN** calling `_should_use_engine_greeks(analytical_engine)`
- **THEN** returns `False`

### Requirement: PDE Grid-Based Greeks

The system SHALL use PDE engine's grid-based `calculate_greeks()` when mode is `AUTO` or `ENGINE`.

#### Scenario: AUTO mode with PDE engine
- **GIVEN** a European call option
- **AND** a PDE engine with grid_size=300
- **AND** calculator with `greeks_mode = GreeksCalculationMode.AUTO`
- **WHEN** calling `calculate_numerical_greeks(product, env, pde_engine)`
- **THEN** delta is extracted from PDE grid
- **AND** gamma is positive (expected for call option)
- **AND** gamma < 0.1 (reasonable range for ATM call)

#### Scenario: BUMP mode with PDE engine
- **GIVEN** a European call option
- **AND** a PDE engine
- **AND** calculator with `greeks_mode = GreeksCalculationMode.BUMP`
- **WHEN** calling `calculate_numerical_greeks(product, env, pde_engine)`
- **THEN** delta is calculated via bump method
- **AND** gamma is calculated via bump method

### Requirement: Analytical Engine Behavior

The system SHALL use bump method for analytical engines (which don't have grid-based Greeks).

#### Scenario: AUTO mode with BlackScholes engine
- **GIVEN** a European call option
- **AND** BlackScholes engine
- **AND** calculator with `greeks_mode = GreeksCalculationMode.AUTO`
- **WHEN** calling `calculate_numerical_greeks(product, env, bs_engine)`
- **THEN** bump method is used for delta/gamma
- **AND** all Greeks are returned correctly

### Requirement: American Option PDE Greeks

The system SHALL calculate correct Greeks for American options using PDE grid method.

#### Scenario: American put with PDE engine
- **GIVEN** an American put option (strike=100, maturity=1yr)
- **AND** a PDE engine with grid_size=300
- **AND** calculator with `greeks_mode = GreeksCalculationMode.AUTO`
- **WHEN** calling `calculate_numerical_greeks(product, env, pde_engine)`
- **THEN** delta < 0 (put has negative delta)
- **AND** gamma > 0 (convexity)
- **AND** theta < 0 (time decay)

