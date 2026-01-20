# Change: Enhance autocallable risk report into a dashboard

## Why
The Snowball report should move beyond PV/Greeks output into a risk management dashboard that highlights skew, barrier, basis, and lifecycle risks. This provides actionable trader-level insights for China index autocallables.

## What Changes
- Add skew/smile risk shocks and higher-order vol sensitivities (Vanna/Volga).
- Add barrier proximity metrics and barrier-zoom risk grids around KI/KO.
- Add higher-order time Greeks (Charm/Color) and lifecycle context (pre/post KI).
- Add executive dashboard summary and stress-scenario tables.
- Add conditional cashflow projections (expected vs conditional-on-KO date).

## Impact
- Affected specs: `autocallable-risk-report`
- Affected code: report generator, surface/grid utilities, plotting, and tests.

## Out of Scope / Already Implemented
- Bucketed Vega/Rho and the Spot×Vol ladder table already exist in the report pipeline.
