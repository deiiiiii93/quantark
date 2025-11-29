# Change: Add Fixed Income Stress Test Support

## Why

The current `stresstest` module only understands equity-style portfolios that expose spot/volatility risk and delta-based Greeks. Portfolios built with the upcoming `fi-portfolio` capability cannot be evaluated under stress because:
- `StressTestEngine` hardcodes `portfolio.Portfolio` and only reports equity Greeks.
- `StressApplicator` cannot express yield-curve bucket shocks or credit spread bumps needed for DV01/convexity analysis.
- Reporting/visualization layers lack FI risk metrics, so portfolio managers have no way to measure duration gaps under scenario shocks.

We need parity with the `add-fi-backtest` change so rate products can be stressed alongside equities.

## What Changes

### Phase 1: Base Stress Protocols
- Introduce `BaseStressEngine`, `ScenarioEvaluator`, and `StressMetricsAdapter` protocols under `stresstest/base.py`.
- Refactor existing equity implementation into `stresstest/equity/` (engine, config, results, reporter) and keep backward-compatible re-exports.
- Extend `StressApplicator` to delegate parameter mutations through asset adapters so FI engines can override rate/credit stress logic.

### Phase 2: Fixed Income Stress Implementation
- Implement `FIStressConfig`, `FIStressEngine`, and `FIMetricsCalculator` that ingest `FIPortfolio` objects and compute DV01, convexity, carry, and liquidity metrics per scenario.
- Create `FIStressResults`/exporters that persist curve shifts, DV01 deltas, and hedge impact, mirroring the FI backtest schema.
- Add FI-focused scenario helpers (parallel shift, steepener/flattener, spread shock) plus templated YAML definitions.

### Phase 3: Reporting, Examples, and Tests
- Extend `StressTestVisualizer`/report generator with FI pages (DV01 waterfalls, curve shift attribution).
- Provide an end-to-end example (`examples/stresstest/fi_rate_shocks.py`) and pytest coverage for DV01 aggregation + scenario storage.
- Update docs/README to explain how to run FI stress workflows and note dependencies on `fi-portfolio`.

## Impact

- Affected specs: `stresstest-protocols` (new), `fi-stresstest` (new)
- Affected code:
  - `stresstest/base.py` (new protocols)
  - `stresstest/equity/*` (refactored existing engine/config/results/reporting)
  - `stresstest/fi/*` (new FI implementations, metrics, exporters, examples)
  - `stresstest/stress/stress_applicator.py` (rate/credit adapters)
  - `stresstest/scenario/*` (FI scenario templates & validation enhancements)
  - `stresstest/results/*`, `stresstest/report/*`, README/docs
- Dependencies: builds on `add-fi-backtest` (for `FIPortfolio`, DV01 calculators, and hedge metadata)

