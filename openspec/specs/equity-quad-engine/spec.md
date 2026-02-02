# equity-quad-engine Specification

## Purpose
TBD - created by archiving change refactor-quad-core. Update Purpose after archive.
## Requirements
### Requirement: Product-Agnostic Quadrature Core
The system SHALL provide a quadrature core that implements the recursion in Eq. 3.5 using per-observation boundary levels (`K^-`, `K^+`) and linear payoff coefficients (`a^-`, `b^-`, `a^+`, `b^+`), plus terminal coefficients (`a_M`, `b_M`).

#### Scenario: Vanilla payoff via core inputs
- **GIVEN** a single observation date at maturity with `K^- = 0`, `K^+ = +inf`, and terminal coefficients matching a European call
- **WHEN** the quadrature core prices the contract
- **THEN** the result matches the European quadrature engine within numerical tolerance

### Requirement: Adapter-Based Product Mapping
The system SHALL map product-specific parameters to quadrature-core inputs via adapter functions or classes, keeping product logic out of the core implementation.

#### Scenario: Barrier rebate mapping
- **GIVEN** a discretely monitored barrier option with rebates
- **WHEN** the quad engine prices the option
- **THEN** rebates are represented as boundary coefficients at observation dates and the core is invoked without direct factor manipulation in the engine

### Requirement: Observation-Date Time Stepping
The quadrature core SHALL step over the actual observation dates (piecewise-constant parameters) rather than an implicit uniform time grid.

#### Scenario: Irregular observation schedule
- **GIVEN** an irregular set of observation times
- **WHEN** the quad engine prices the contract
- **THEN** the core uses the provided `Δt_m` intervals without resampling the schedule

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

### Requirement: Shared Quadrature Math Utilities
The system SHALL provide reusable quadrature math utilities for grid construction, Simpson integration weights, FFT convolution, and spot interpolation so that multiple quadrature engines can share numerical primitives without duplicating logic.

#### Scenario: Shared math used by core and snowball engine
- **GIVEN** the quadrature core and snowball quadrature engine
- **WHEN** each engine performs diffusion steps
- **THEN** both use the shared quadrature math utility for grid setup and convolution

### Requirement: Phoenix Memory Coupon Support (QUAD)
The system SHALL support `memory_coupon=True` Phoenix options in the quadrature
engine by accounting for accumulated missed coupons across observation dates.

#### Scenario: QUAD prices memory Phoenix
- **GIVEN** a Phoenix option with `memory_coupon=True`
- **WHEN** the option is priced using `PhoenixQuadEngine`
- **THEN** the price reflects accumulated coupons when the coupon barrier is hit

### Requirement: Configurable Quadrature Stability Controls
The system SHALL allow quadrature engines to configure FFT padding, spectral filtering, and log-domain width to reduce aliasing and truncation artifacts.

#### Scenario: FFT padding and filtering applied
- **GIVEN** quadrature parameters that specify FFT padding and filter settings
- **WHEN** the quad engine performs an FFT convolution step
- **THEN** the convolution uses the configured padding and spectral filter

#### Scenario: Log-domain width uses configured standard deviations
- **GIVEN** quadrature parameters with a specified log-domain standard deviation width
- **WHEN** the quad grid is constructed
- **THEN** the grid spans the configured number of standard deviations in log space

### Requirement: Barrier-Aligned Grid Support
The system SHALL support aligning the log-price grid to a specified barrier or level to reduce interpolation noise at event boundaries.

#### Scenario: Barrier alignment places a barrier on a grid node
- **GIVEN** a quad engine configured with a barrier alignment target
- **WHEN** the grid is constructed
- **THEN** the target barrier level coincides with a grid node within numerical tolerance

### Requirement: Event-Step Smoothing for Discontinuities
The system SHALL optionally smooth discontinuous event operators (e.g., KO/KI/coupon jumps) over a configurable number of grid cells.

#### Scenario: Smoothing enabled
- **GIVEN** event smoothing is configured with a positive number of grid cells
- **WHEN** the quad engine applies an event operator at an observation time
- **THEN** the transition blends values across the boundary rather than using a hard mask

#### Scenario: Smoothing disabled
- **GIVEN** event smoothing is configured with zero cells
- **WHEN** the quad engine applies an event operator at an observation time
- **THEN** the transition uses a hard mask without blending

