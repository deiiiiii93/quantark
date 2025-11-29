## ADDED Requirements

### Requirement: FI Stress Configuration

The system SHALL provide an `FIStressConfig` that captures all inputs required to stress an `FIPortfolio`.

The configuration MUST support:
- Selecting DV01 and convexity thresholds that trigger additional reporting detail.
- Enabling/disabling key-rate DV01 tracking and curve metadata persistence.
- Declaring hedging artifacts (bond futures specs, DV01 per contract) so hedge impact can be reported.
- Referencing FI-focused scenario templates (parallel shift, steepener/flattener, spread shock).

#### Scenario: Config Validation
- **GIVEN** an `FIStressConfig` with DV01 threshold, curve adapters, and futures metadata
- **WHEN** `validate()` is called
- **THEN** it verifies that the attached portfolio exposes FI risk measures
- **AND** it raises a `ValidationError` if key-rate tracking is enabled without tenor buckets.

### Requirement: FI Stress Engine

The system SHALL provide an `FIStressEngine` that implements `BaseStressEngine` and evaluates FI portfolios under stress scenarios.

The engine MUST:
- Accept `FIPortfolio` instances plus FI-aware scenarios and pricing environments.
- Apply rate and spread stresses using adapter hooks (parallel shift, key-rate bumps, twists).
- Recalculate DV01, convexity, key-rate DV01, and carry for each scenario.
- Record hedge adjustments (if bond futures or swaps are present) and include them in the results.

#### Scenario: DV01 Shock Evaluation
- **GIVEN** an FI portfolio with $250k DV01 and a +100 bps parallel shock
- **WHEN** `run_static_scenarios()` executes
- **THEN** the engine recomputes DV01 under the stressed curve
- **AND** reports the expected P&L (DV01 × shock) within tolerance.

#### Scenario: Key-Rate Twist
- **GIVEN** a scenario that bumps 2y by +50 bps and 10y by -25 bps
- **WHEN** the engine runs
- **THEN** it applies tenor-specific shocks via adapters
- **AND** reports key-rate DV01 contributions for each tenor.

### Requirement: FI Stress Results & Metrics

The system SHALL provide `FIStressResults` objects that extend the stress result envelope with FI metrics.

The results MUST include:
- Scenario-level DV01, convexity, key-rate DV01 vectors, and carry.
- Curve shock metadata (parallel shift amount, tenor bumps, spread deltas).
- Hedge impact statistics (DV01 neutralized, futures contracts used).
- Accessor methods such as `get_dv01_series()` and `get_curve_shift_summary()`.

#### Scenario: DV01 Series Access
- **GIVEN** completed FI stress results over five scenarios
- **WHEN** `get_dv01_series()` is called
- **THEN** it returns a DataFrame with pre- and post-hedge DV01 per scenario
- **AND** includes the applied shock magnitude for context.

#### Scenario: Hedge Impact Summary
- **GIVEN** a portfolio with bond futures hedges
- **WHEN** `get_hedge_summary()` is invoked
- **THEN** it reports DV01 neutralized, contracts traded, and resulting net DV01
- **AND** ties each metric back to the originating scenario.

### Requirement: FI Scenario Library & Reporting

The system SHALL extend the scenario library, exporters, and reports with FI-specific capabilities.

The module MUST:
- Provide builder helpers for parallel shifts, curve twists, and spread shocks (with YAML/JSON serialization).
- Export FI stress results to Parquet/CSV with DV01 vectors and curve metadata.
- Render DV01 waterfalls, curve shift plots, and exposure tables inside the HTML report and visualizer.
- Document the workflow in README plus an executable FI stress example.

#### Scenario: Parallel Shift Template
- **GIVEN** `FIScenarioLibrary.parallel_shift(100)` is called
- **WHEN** the resulting scenario is saved to YAML and reloaded
- **THEN** it preserves the +100 bps metadata
- **AND** the engine interprets it as a curve-parallel adapter stress.

#### Scenario: Reporting Outputs
- **GIVEN** FI stress results with DV01 vectors
- **WHEN** `ReportGenerator.generate_report()` runs with FI data
- **THEN** the report includes DV01 waterfall and curve twist plots
- **AND** exports the underlying data to the configured formats.

