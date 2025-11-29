## 1. Protocol Scaffolding

- [x] 1.1 Add `dynamicscenario/base.py` with `BaseDynamicScenarioEngine`, `BaseScenarioResults`, `RiskMetricsAdapter` protocols.
- [x] 1.2 Create `dynamicscenario/equity/` directory and move existing engine implementation there.
- [x] 1.3 Update `dynamicscenario/__init__.py` with backward-compatible aliases (keep `DynamicScenarioEngine` pointing to equity).
- [x] 1.4 Verify existing equity dynamic scenario tests still pass.

## 2. Extended Path Components

- [x] 2.1 Extend `ParameterChange` in `path/day_path.py` to support rate curve parameters (parallel shift, key-rate bumps).
- [x] 2.2 Extend `PathBuilder` with rate curve methods: `rate_parallel_shift()`, `rate_curve_twist()`, `rate_key_bump()`.
- [x] 2.3 Create `path/fi_path_library.py` with FI-specific patterns: `parallel_shift()`, `steepener()`, `flattener()`, `rate_hike_cycle()`.
- [x] 2.4 Add FI path pattern tests.

## 3. FI Dynamic Scenario Engine

- [x] 3.1 Create `dynamicscenario/fi/__init__.py` with exports.
- [x] 3.2 Implement `fi/config.py` with `FIDynamicScenarioConfig` (DV01 thresholds, curve adapters, hedge settings).
- [x] 3.3 Implement `fi/engine.py` with `FIDynamicScenarioEngine` that operates on `FIPortfolio`.
- [x] 3.4 Implement `fi/results.py` with `FIDayResult` and `FIDynamicScenarioResults` tracking DV01, convexity, duration.
- [x] 3.5 Support FI hedging integration with `DV01NeutralStrategy` and bond futures execution.
- [x] 3.6 Update `dynamicscenario/__init__.py` to export FI classes.

## 4. FI Risk Measures & State Tracking

- [x] 4.1 Implement DV01 evolution tracking at each day step.
- [x] 4.2 Implement convexity and modified duration evolution tracking.
- [x] 4.3 Implement optional key-rate DV01 tracking (per tenor).
- [x] 4.4 Implement rate curve state capture in `FIMarketState`.
- [x] 4.5 Compute and track hedge impact (futures positions, DV01 neutralized).

## 5. Visualization & Reporting

- [x] 5.1 Extend `DynamicScenarioVisualizer` with FI-specific plots: DV01 evolution, duration tracking, rate curve evolution.
- [x] 5.2 Extend `DynamicReportGenerator` with FI-specific report sections.
- [x] 5.3 Extend `DynamicResultExporter` to export FI metrics (DV01 series, curve states).

## 6. Example & Documentation

- [x] 6.1 Create `example/dynamic_scenario_fi_demo.py` demonstrating FI dynamic scenario workflow.
- [x] 6.2 Update `dynamicscenario/README.md` (or create if not exists) with FI workflow documentation.
- [x] 6.3 Ensure `openspec validate add-fi-dynamicscenario --strict` passes.
