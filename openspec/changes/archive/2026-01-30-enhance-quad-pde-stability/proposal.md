# Change: Enhance quad/PDE stability controls

## Why
Phoenix and Snowball pricing showed grid-size oscillations and sensitivity to event-time alignment. We need explicit, configurable stability controls and strict event alignment to improve convergence and comparability across engines.

## What Changes
- Add configurable quadrature stability controls (FFT padding, spectral filter, domain width, barrier alignment, event-step smoothing).
- Enforce exact event-time alignment in PDE solvers for discretely monitored products.
- Update Phoenix engine comparison demo defaults to use the tuned grids and RQMC method.

## Impact
- Affected specs: `equity-quad-engine`, `equity-pde-engine`.
- Affected code: quad math utilities, quad params, Phoenix/Snowball quad engines, Snowball/Phoenix PDE solvers, demo script.
