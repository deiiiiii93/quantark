## ADDED Requirements
### Requirement: Numerical Greeks coverage
The system SHALL provide numerical point Greeks for vanna (∂²V/∂S∂σ), volga (∂²V/∂σ²), and cross spot–dividend sensitivity dDelta/dq using explicit bump conventions.

#### Scenario: Compute point Vanna/Volga/dDelta-dq
- **GIVEN** a product, pricing environment, and engine
- **WHEN** numerical Greeks are requested
- **THEN** the result includes vanna, volga, and dDelta/dq with documented bump sizes
