## ADDED Requirements

### Requirement: Shared Quadrature Math Utilities
The system SHALL provide reusable quadrature math utilities for grid construction, Simpson integration weights, FFT convolution, and spot interpolation so that multiple quadrature engines can share numerical primitives without duplicating logic.

#### Scenario: Shared math used by core and snowball engine
- **GIVEN** the quadrature core and snowball quadrature engine
- **WHEN** each engine performs diffusion steps
- **THEN** both use the shared quadrature math utility for grid setup and convolution
