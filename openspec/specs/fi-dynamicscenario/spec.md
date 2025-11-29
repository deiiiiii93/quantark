# fi-dynamicscenario Specification

## Purpose
TBD - created by archiving change add-fi-dynamicscenario. Update Purpose after archive.
## Requirements
### Requirement: FI Dynamic Scenario Configuration

The system SHALL provide an `FIDynamicScenarioConfig` that captures all inputs required for FI dynamic scenario analysis.

The configuration MUST support:
- Enabling DV01, convexity, and duration calculation at each step
- Optional key-rate DV01 tracking with configurable tenor buckets
- Hedge settings (bond futures specification, DV01 per contract)
- Export and reporting options consistent with equity config

#### Scenario: Config Validation

- **GIVEN** an `FIDynamicScenarioConfig` with DV01 thresholds and curve settings
- **WHEN** `validate()` is called
- **THEN** it verifies that the attached portfolio exposes FI risk measures
- **AND** it raises `ValidationError` if key-rate tracking is enabled without tenor buckets

#### Scenario: Config with Hedging

- **GIVEN** an `FIDynamicScenarioConfig` with hedge_enabled=True and futures_spec
- **WHEN** the config is used with `FIDynamicScenarioEngine`
- **THEN** hedging is executed using the specified bond futures

### Requirement: FI Dynamic Scenario Engine

The system SHALL provide an `FIDynamicScenarioEngine` that implements `BaseDynamicScenarioEngine` and simulates FI portfolios through multi-day rate scenarios.

The engine MUST:
- Accept `FIPortfolio` instances plus FI-aware day paths
- Apply rate curve changes from day path steps to pricing environments
- Recalculate DV01, convexity, and duration at each day step
- Optionally execute hedges via `DV01NeutralStrategy` and bond futures
- Record hedge positions and their DV01 impact

#### Scenario: Multi-Day Rate Shock Simulation

- **GIVEN** an FI portfolio with $500k DV01 and a 5-day parallel shift path (+20bps/day)
- **WHEN** `run()` executes the simulation
- **THEN** the engine applies cumulative +100bps over 5 days
- **AND** tracks DV01 evolution at each step
- **AND** reports expected P&L consistent with DV01 sensitivity

#### Scenario: FI Portfolio with Hedging

- **GIVEN** an FI portfolio and DV01 neutral strategy with threshold $50,000
- **WHEN** `run()` executes with a rate path causing DV01 to exceed threshold
- **THEN** the engine executes futures hedge trades
- **AND** records trade details in day results
- **AND** post-hedge DV01 is within threshold tolerance

#### Scenario: Rate Curve Update at Each Step

- **GIVEN** an FI dynamic scenario in progress
- **WHEN** stepping to a new day with rate changes
- **THEN** the pricing environment rate curve is updated
- **AND** all bond prices and risk measures are recalculated

### Requirement: FI Day Result

The system SHALL provide an `FIDayResult` that extends day result functionality with FI-specific metrics.

The result MUST include:
- Portfolio DV01 (pre-hedge and post-hedge)
- Portfolio convexity
- Portfolio modified duration
- Optional key-rate DV01 vector
- Rate curve state (current rate at key tenors)
- Hedge position state (futures contracts held)

#### Scenario: FI Day Result Access

- **GIVEN** a completed FI day result
- **WHEN** `dv01` property is accessed
- **THEN** it returns the portfolio DV01 for that day
- **AND** includes both pre-hedge and post-hedge values if hedging occurred

#### Scenario: FI Day Result with Key-Rate DV01

- **GIVEN** a day result with key-rate tracking enabled
- **WHEN** `get_key_rate_dv01()` is called
- **THEN** it returns a dictionary mapping tenors to DV01 contributions

### Requirement: FI Dynamic Scenario Results

The system SHALL provide `FIDynamicScenarioResults` with FI-specific aggregation and analysis methods.

The results MUST include:
- Time series of portfolio DV01, convexity, and duration
- Time series of rate curve states
- Hedge trade history with DV01 impact per trade
- Summary statistics for FI hedging effectiveness

#### Scenario: DV01 Evolution Series

- **GIVEN** completed FI dynamic scenario results over 10 days
- **WHEN** `get_dv01_evolution()` is called
- **THEN** it returns a DataFrame with DV01 values at each day
- **AND** includes pre-hedge and post-hedge columns

#### Scenario: Duration Evolution

- **GIVEN** completed FI dynamic scenario results
- **WHEN** `get_duration_evolution()` is called
- **THEN** it returns a DataFrame with modified duration at each day

#### Scenario: Rate Curve Evolution

- **GIVEN** completed FI dynamic scenario results with rate changes
- **WHEN** `get_rate_evolution()` is called
- **THEN** it returns a DataFrame with rate curve states at each day

### Requirement: FI Path Library

The system SHALL provide `FIPathLibrary` with Fixed Income-specific day path patterns.

The library MUST provide:
- `parallel_shift(days, total_bps)` - uniform rate change across tenors
- `steepener(days, short_bps, long_bps)` - short rates down, long rates up
- `flattener(days, short_bps, long_bps)` - short rates up, long rates down
- `rate_hike_cycle(days, total_bps)` - gradual rate increases
- `rate_cut_cycle(days, total_bps)` - gradual rate decreases
- `historical_fed_tightening_2022()` - modeled on 2022 Fed rate hikes

#### Scenario: Parallel Shift Path

- **GIVEN** `FIPathLibrary.parallel_shift(days=5, total_bps=50)`
- **WHEN** the path is built
- **THEN** it contains 5 day steps with +10bps rate change each
- **AND** cumulative rate change equals +50bps

#### Scenario: Steepener Path

- **GIVEN** `FIPathLibrary.steepener(days=5, short_bps=-25, long_bps=25)`
- **WHEN** the path is built
- **THEN** short-end rates decrease by -25bps total
- **AND** long-end rates increase by +25bps total
- **AND** the yield curve steepens

#### Scenario: Rate Hike Cycle

- **GIVEN** `FIPathLibrary.rate_hike_cycle(days=10, total_bps=100)`
- **WHEN** the path is built
- **THEN** rates increase gradually over 10 days
- **AND** total cumulative increase equals +100bps

### Requirement: FI Visualization Extensions

The system SHALL extend `DynamicScenarioVisualizer` with FI-specific plots.

The visualizer MUST provide:
- DV01 evolution chart showing pre-hedge and post-hedge DV01 over days
- Duration evolution chart
- Rate curve evolution chart (rate at key tenors over days)
- Hedge trade chart showing futures positions and trade activity

#### Scenario: DV01 Evolution Plot

- **GIVEN** FI dynamic scenario results with hedging
- **WHEN** `plot_dv01_evolution()` is called
- **THEN** it renders a chart showing DV01 over days
- **AND** distinguishes pre-hedge vs post-hedge DV01

#### Scenario: Rate Curve Evolution Plot

- **GIVEN** FI dynamic scenario results with rate changes
- **WHEN** `plot_rate_evolution()` is called
- **THEN** it renders a chart showing rate levels over days

### Requirement: FI Report Extensions

The system SHALL extend `DynamicReportGenerator` with FI-specific report sections.

The report MUST include:
- FI portfolio summary (DV01, duration, convexity at start and end)
- DV01 evolution section with chart
- Rate scenario summary (total rate change, curve shape change)
- Hedge effectiveness metrics (DV01 tracking error, hedge frequency)

#### Scenario: FI Report Generation

- **GIVEN** FI dynamic scenario results with hedging
- **WHEN** `generate_report()` runs with FI data
- **THEN** the report includes DV01 evolution chart
- **AND** includes hedge effectiveness summary
- **AND** exports underlying data to configured formats

