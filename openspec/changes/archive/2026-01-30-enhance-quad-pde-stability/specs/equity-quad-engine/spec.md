## ADDED Requirements
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
