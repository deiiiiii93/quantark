# Change: Add Barrier Option Quadrature Engine

## Why

The codebase currently supports barrier option pricing via analytical, Monte Carlo, and PDE methods. However, for discrete barrier monitoring with complex observation schedules, Monte Carlo can be slow and analytical methods have limitations. A quadrature-based engine provides a deterministic, fast alternative with spectral accuracy for discrete barrier monitoring.

This extends the recently added `EuropeanQuadEngine` to path-dependent barrier options, leveraging the existing reference implementation in `docs/quad/ref_scripts/option_quad.py`.

## What Changes

- Add `BarrierQuadEngine` class in `asset/equity/engine/quad/barrier_quad_engine.py`
- Support all 4 single barrier types: UP_IN, UP_OUT, DOWN_IN, DOWN_OUT
- Implement FFT-based convolution with Simpson's rule integration
- Handle discrete observation schedules via `ObservationSchedule`
- Support rebate handling (at hit vs at expiry)
- Update engine exports in `asset/equity/engine/quad/__init__.py`
- Add unit tests in `test/test_barrier_quad_engine.py`
- Add demo in `example/barrier_quad_demo.py`

## Impact

- **Affected specs**: None (new capability)
- **Related specs**: `equity-barrier-products`, `base-engine`
- **Affected code**:
  - `asset/equity/engine/quad/` - New engine file
  - `asset/equity/engine/__init__.py` - Export update
  - `test/` - New test file
  - `example/` - New demo file
- **Breaking changes**: None
