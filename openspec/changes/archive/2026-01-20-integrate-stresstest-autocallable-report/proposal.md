# Change: Integrate stresstest scenarios into autocallable risk report

## Why
The report's stress scenario table should leverage the existing stresstest framework to ensure consistency with portfolio-level stress tooling and scenario libraries.

## What Changes
- Build the report's stress scenario table using the stresstest scenario engine (EquityStressEngine).
- Allow report inputs to supply stresstest scenarios or simple shock configs that are converted to ScenarioBuilder scenarios.
- Extend stresstest adapters to support term-structure vol/dividend inputs for autocallable reporting.

## Impact
- Affected specs: `autocallable-risk-report`
- Affected code: `asset/equity/report/autocallable_risk_report.py`, `stresstest/stress/stress_applicator.py`, tests.
