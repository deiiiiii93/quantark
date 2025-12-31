# base-engine Specification

## Purpose
TBD - created by archiving change add-pde-greeks-mode. Update Purpose after archive.
## Requirements
### Requirement: Engine Type Attribute

The system SHALL provide an `engine_type` class attribute on `BaseEngine` that indicates the engine type category.

#### Scenario: BaseEngine has default type
- **GIVEN** `BaseEngine` class
- **WHEN** accessing `BaseEngine.engine_type`
- **THEN** returns `EngineType.ANALYTICAL` (default)

### Requirement: Engine Type Values

All concrete engine classes SHALL set their `engine_type` to the appropriate `EngineType` value.

#### Scenario: Analytical engines type
- **GIVEN** `BlackScholesEngine` class
- **WHEN** accessing `BlackScholesEngine.engine_type`
- **THEN** returns `EngineType.ANALYTICAL`
- **AND** same for `AmericanOptionAnalyticalEngine`, `DeltaOneEngine`, etc.

#### Scenario: Monte Carlo engines type
- **GIVEN** `EuropeanMCEngine` class
- **WHEN** accessing `EuropeanMCEngine.engine_type`
- **THEN** returns `EngineType.MONTE_CARLO`
- **AND** same for all MC engines

#### Scenario: PDE engines type
- **GIVEN** `PDEEngine` class
- **WHEN** accessing `PDEEngine.engine_type`
- **THEN** returns `EngineType.PDE`
- **AND** same for `BasePDESolver`

### Requirement: Instance Access

Engine instances SHALL inherit the `engine_type` from their class.

#### Scenario: Instance type access
- **GIVEN** a `PDEEngine()` instance
- **WHEN** accessing `instance.engine_type`
- **THEN** returns `EngineType.PDE`
- **AND** no separate instance storage is required

### Requirement: Type Detection

The system SHALL enable easy engine type detection using `getattr()`.

#### Scenario: Detect PDE engine
- **GIVEN** any engine instance
- **WHEN** calling `getattr(engine, 'engine_type', None)`
- **THEN** returns appropriate `EngineType` value
- **OR** returns `None` if attribute not present (defensive)

