## ADDED Requirements

### Requirement: DV01 Neutral Strategy

The system SHALL provide a `DV01NeutralStrategy` for hedging Fixed Income portfolios based on DV01 exposure.

The strategy MUST:
- Monitor portfolio DV01 and trigger hedges when threshold is exceeded
- Calculate hedge size in bond futures contracts to neutralize DV01
- Support configurable rebalance frequency (daily, on-threshold, continuous)
- Track hedge statistics (number of hedges, total DV01 hedged)

#### Scenario: DV01 Threshold Trigger

- **GIVEN** a strategy with DV01 threshold of $50,000
- **WHEN** portfolio DV01 exceeds $50,000
- **THEN** the strategy signals that hedging should occur
- **AND** calculates the number of futures contracts to hedge

#### Scenario: DV01 Hedge Calculation

- **GIVEN** portfolio DV01 of $100,000 and futures contract DV01 of $1,000
- **WHEN** `calculate_hedge_size()` is called
- **THEN** it returns 100 futures contracts (short) to neutralize exposure
- **AND** the target portfolio DV01 becomes zero

### Requirement: Convexity Neutral Strategy

The system SHALL provide a `ConvexityNeutralStrategy` for hedging both DV01 and convexity exposure.

The strategy MUST:
- Monitor both DV01 and convexity thresholds
- Calculate hedge sizes for multiple instruments if needed
- Support target convexity level configuration
- Build on `DV01NeutralStrategy` with additional convexity logic

#### Scenario: Dual Hedge Trigger

- **GIVEN** DV01 threshold of $50,000 and convexity threshold of $1,000,000
- **WHEN** either threshold is exceeded
- **THEN** the strategy signals hedging should occur
- **AND** calculates appropriate hedge for the exceeded measure

### Requirement: Fixed Income Hedge Executor

The system SHALL provide an `FIHedgeExecutor` for executing hedges using bond futures.

The executor MUST:
- Implement the `BaseHedgeExecutor` protocol
- Create and manage bond futures positions
- Calculate transaction costs for futures trades
- Track hedge position history

#### Scenario: Bond Futures Hedge Execution

- **GIVEN** an FI portfolio and hedge size of 50 contracts (short)
- **WHEN** `execute_hedge()` is called with the futures specification
- **THEN** a short futures position of 50 contracts is created
- **AND** a trade record is returned with execution details

#### Scenario: Hedge Position Update

- **GIVEN** an existing futures hedge position of 50 contracts (short)
- **WHEN** an additional hedge of 20 contracts (short) is executed
- **THEN** the position is updated to 70 contracts (short)
- **AND** the trade record shows the incremental trade

### Requirement: Fixed Income Backtest Engine

The system SHALL provide an `FIBacktestEngine` for simulating Fixed Income hedging strategies.

The engine MUST:
- Implement the `BaseBacktestEngine` protocol
- Accept `FIPortfolio` and `FIBacktestConfig` as inputs
- Update rate curves from market data time series
- Calculate portfolio DV01/convexity at each step
- Execute hedges via `FIHedgeExecutor`
- Record state history with FI-specific metrics

#### Scenario: Complete FI Backtest Simulation

- **GIVEN** an FI portfolio with 3 bonds and a DV01 neutral strategy
- **WHEN** `run()` is called with 6 months of rate data
- **THEN** the engine simulates daily rebalancing
- **AND** returns results with DV01 tracking, P&L, and hedge statistics

#### Scenario: Rate Curve Update

- **GIVEN** an FI backtest in progress
- **WHEN** stepping to a new date with updated rate data
- **THEN** the pricing environment rate curve is updated
- **AND** all bond prices and risk measures are recalculated

### Requirement: Fixed Income Backtest Configuration

The system SHALL provide an `FIBacktestConfig` for configuring FI backtests.

The configuration MUST support:
- Initial FI portfolio with positions
- FI hedging strategy selection
- Bond futures specification for hedging
- Transaction cost model
- Market data adapter for rates and bond prices
- Date range and frequency settings

#### Scenario: FI Config Validation

- **GIVEN** an `FIBacktestConfig` with all required fields
- **WHEN** the config is validated
- **THEN** it verifies the portfolio contains valid FI positions
- **AND** it verifies the hedge instrument is compatible

### Requirement: Fixed Income Backtest Results

The system SHALL provide `FIBacktestResults` with Fixed Income-specific metrics and data.

The results MUST include:
- Time series of portfolio DV01, convexity, and market value
- Time series of hedge positions (futures contracts)
- Trade history with DV01 impact per trade
- Summary statistics for FI hedging effectiveness

#### Scenario: FI Results Access

- **GIVEN** completed FI backtest results
- **WHEN** `get_dv01_series()` is called
- **THEN** it returns the time series of portfolio DV01
- **AND** includes both pre-hedge and post-hedge values

### Requirement: Fixed Income Metrics

The system SHALL provide `FIMetricsCalculator` for Fixed Income-specific performance metrics.

The metrics MUST include:
- DV01 tracking error (deviation from target)
- Average absolute DV01 exposure
- Hedge frequency and average hedge size
- Rate-adjusted returns
- Duration contribution to P&L

#### Scenario: DV01 Tracking Error Calculation

- **GIVEN** time series of DV01 values and target of zero
- **WHEN** `dv01_tracking_error()` is called
- **THEN** it returns the RMSE of DV01 from target
- **AND** provides both absolute and percentage metrics

