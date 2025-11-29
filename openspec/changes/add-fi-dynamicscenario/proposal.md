# Change: Add Fixed Income Support to Dynamic Scenario Module

## Why

The current `dynamicscenario` module is designed specifically for equity portfolios, using equity-style risk measures (delta, gamma, vega, theta) and hedging. Fixed Income portfolios built with `FIPortfolio` cannot be simulated through dynamic multi-day scenarios because:
- `DynamicScenarioEngine` hardcodes `Portfolio` and only reports equity Greeks.
- `PathLibrary` provides equity-focused patterns (spot/vol changes) but lacks FI-specific rate shock paths (parallel shifts, curve twists, steepeners/flatteners).
- Results and visualizations only emit equity Greeks and cannot track DV01, convexity, or duration evolution.

We need parity with the `add-fi-backtest` and `add-fi-stresstest` changes so rate products can be simulated through dynamic scenarios alongside equities.

## What Changes

### Phase 1: Base Dynamic Scenario Protocols
- Introduce `BaseDynamicScenarioEngine`, `BaseScenarioResults`, and `RiskMetricsAdapter` protocols under `dynamicscenario/base.py`.
- Refactor existing equity implementation into `dynamicscenario/equity/` and keep backward-compatible re-exports.
- Extend path components to support rate curve changes (parallel shifts, key-rate bumps, curve twists).

### Phase 2: Fixed Income Dynamic Scenario Implementation
- Implement `FIDynamicScenarioConfig`, `FIDynamicScenarioEngine`, and `FIDayResult` that ingest `FIPortfolio` objects.
- Create FI-specific path patterns in `FIPathLibrary` (rate hike cycle, parallel shift, curve twist, steepener/flattener).
- Compute and track DV01, convexity, key-rate DV01, and duration at each day step.
- Support FI hedging strategies (`DV01NeutralStrategy`) with bond futures execution.

### Phase 3: Reporting, Examples, and Documentation
- Extend `DynamicScenarioVisualizer` with FI-specific plots (DV01 evolution, duration tracking, curve shift attribution).
- Extend `DynamicReportGenerator` for FI-specific reports.
- Provide an end-to-end example (`example/dynamic_scenario_fi_demo.py`).
- Update `dynamicscenario/README.md` with FI workflow documentation.

## Impact

- Affected specs: `dynamicscenario-protocols` (new), `fi-dynamicscenario` (new)
- Affected code:
  - `dynamicscenario/base.py` (new protocols)
  - `dynamicscenario/equity/*` (refactored existing engine/config/results)
  - `dynamicscenario/fi/*` (new FI implementations)
  - `dynamicscenario/path/*` (extend path components for rate shocks)
  - `dynamicscenario/report/*` (FI visualizations and reports)
  - `dynamicscenario/results/*` (FI-specific result classes)
- Dependencies: builds on `add-fi-backtest` (for `FIPortfolio`, DV01 calculators, and hedge metadata)

