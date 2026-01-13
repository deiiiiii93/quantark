## ADDED Requirements
### Requirement: Adapter Registry for Discrete Quadrature
The system SHALL resolve product-specific quad input adapters and delegate `QuadCoreInputs` construction to them, keeping `DiscreteQuadEngine` product-agnostic.

#### Scenario: Barrier and one-touch adapters
- **GIVEN** a barrier option or one-touch option
- **WHEN** the discrete quadrature engine prices the product
- **THEN** the resolved adapter constructs `QuadCoreInputs` and the engine does not access product-specific fields directly

### Requirement: Unified Discrete Pricing Flow
The system SHALL use a single internal pricing pipeline for discrete quadrature that accepts a resolved adapter and shared grid parameters.

#### Scenario: Shared pipeline for discrete products
- **GIVEN** adapters for barrier and one-touch products
- **WHEN** both products are priced
- **THEN** the engine uses the same internal pricing pipeline for grid setup and core invocation
