# Change: Add point Vanna/Volga/dDelta-dq and high-accuracy surfaces

## Why
Surface-level finite differences are useful for shape diagnostics but can be noisy for path-dependent autocallables. Traders need stable point sensitivities and an optional high-accuracy surface mode based on per-node Greeks calculation.

## What Changes
- Extend `GreeksCalculator` with point Vanna, Volga, and dDelta/dq (cross spot–dividend) using explicit bump conventions.
- Add a report option to compute “high-accuracy” surfaces by calling the Greeks calculator per grid node (slower, more stable).
- Use point Greeks for executive dashboard values when enabled.

## Impact
- Affected specs: `greeks-calculator`, `autocallable-risk-report`
- Affected code: `asset/equity/riskmeasures/greeks_calculator.py`, `asset/equity/report/autocallable_risk_report.py`, tests.
