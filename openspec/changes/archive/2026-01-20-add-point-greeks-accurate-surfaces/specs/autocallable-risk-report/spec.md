## MODIFIED Requirements
### Requirement: Advanced volatility risk surfaces
The system SHALL support an optional high-accuracy mode that computes report surfaces by invoking the Greeks calculator at each grid node, and SHALL use point Greeks for executive dashboard values when enabled.

#### Scenario: High-accuracy surface mode
- **GIVEN** a SnowballOption and a PricingEnvironment
- **WHEN** high-accuracy surface mode is enabled
- **THEN** surfaces are computed via per-node Greeks calculation and reported alongside their bump conventions
