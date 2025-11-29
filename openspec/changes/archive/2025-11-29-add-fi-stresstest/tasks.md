## 1. Protocol Scaffolding
- [x] 1.1 Add `stresstest/base.py` with `BaseStressEngine`, `ScenarioRunner`, `StressMetricsAdapter` protocols.
- [x] 1.2 Move current equity implementation into `stresstest/equity/` and keep `StressTestEngine` alias for backward compatibility.
- [x] 1.3 Update existing unit tests/imports to consume the new module layout.

## 2. Adapter-Ready Stress Application
- [x] 2.1 Extend `StressApplicator` to register asset-specific adapters (spot/vol for equity, rate/spread for FI).
- [x] 2.2 Add validation helpers so scenarios can declare key-rate buckets and spread shocks.
- [x] 2.3 Document the adapter contract in `stresstest/README.md`.

## 3. FI Stress Engine & Metrics
- [x] 3.1 Implement `FIStressConfig`, `FIStressEngine`, and `FIMetricsCalculator` that operate on `FIPortfolio`.
- [x] 3.2 Capture DV01, convexity, key-rate DV01, and carry per scenario; expose via `FIStressResults`.
- [x] 3.3 Persist FI metrics through exporters/reports and ensure `StressTestResults` can surface asset-specific data.

## 4. Scenario Library & Reporting
- [x] 4.1 Add FI scenario templates (parallel shift, steepener/flattener, spread shock) with YAML storage support.
- [x] 4.2 Extend `StressTestVisualizer`/`ReportGenerator` with DV01 waterfalls, curve-shift plots, and summary tables.
- [x] 4.3 Provide an executable example (`examples/stresstest/fi_rate_shocks.py`) demonstrating the full workflow.

## 5. Validation & Testing
- [x] 5.1 Add pytest coverage for DV01 aggregation, adapter dispatch, and FI scenario serialization.
- [x] 5.2 Run existing stress-test regression suite plus new FI-focused tests.
- [x] 5.3 Capture docs/README updates and ensure `openspec validate add-fi-stresstest --strict` passes.

