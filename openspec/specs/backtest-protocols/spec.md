# backtest-protocols Specification

## Purpose
TBD - created by archiving change add-fi-backtest. Update Purpose after archive.
## Requirements
### Requirement: Base Position Protocol

The system SHALL provide a `BasePosition` protocol that defines the interface for all position types across asset classes.

The protocol MUST define methods for:
- Getting current market value given a pricing context
- Getting unrealized P&L
- Getting risk measures (asset-class specific)
- Serialization to dictionary format

#### Scenario: Position Protocol Implementation

- **GIVEN** any asset class (equity, fixed income, etc.)
- **WHEN** a position class implements `BasePosition`
- **THEN** it provides market value, P&L, and risk measure calculations
- **AND** it can be used with generic portfolio operations

### Requirement: Base Portfolio Protocol

The system SHALL provide a `BasePortfolio` protocol that defines the interface for all portfolio types across asset classes.

The protocol MUST define methods for:
- Adding and removing positions
- Getting portfolio-level market value and P&L
- Aggregating risk measures across positions
- Converting to DataFrame representation

#### Scenario: Portfolio Protocol Implementation

- **GIVEN** a collection of positions of any asset class
- **WHEN** a portfolio class implements `BasePortfolio`
- **THEN** it can manage positions and aggregate risk measures
- **AND** it can be used with the backtest engine

### Requirement: Base Hedge Executor Protocol

The system SHALL provide a `BaseHedgeExecutor` protocol that defines the interface for executing hedge trades.

The protocol MUST define methods for:
- Executing a hedge trade given size and pricing context
- Getting current hedge position
- Closing hedge positions
- Tracking hedge statistics

#### Scenario: Hedge Executor Protocol Implementation

- **GIVEN** any hedging instrument type (spot, futures, bond futures)
- **WHEN** a hedge executor implements `BaseHedgeExecutor`
- **THEN** it can execute hedges and track positions
- **AND** it returns standardized trade records

### Requirement: Base Backtest Engine Protocol

The system SHALL provide a `BaseBacktestEngine` protocol that defines the interface for running backtests.

The protocol MUST define methods for:
- Running the backtest simulation
- Stepping through time periods
- Updating pricing environments
- Recording state history
- Returning standardized results

#### Scenario: Backtest Engine Protocol Implementation

- **GIVEN** a portfolio, strategy, and market data
- **WHEN** an engine implements `BaseBacktestEngine`
- **THEN** it simulates the strategy over the time period
- **AND** it returns results compatible with visualization and reporting tools

### Requirement: Dynamic Transaction Cost Model
The system MUST model transaction costs as a combination of spread and market impact:
- Spread varies by asset liquidity tier and volatility bucket (configurable)
- Impact cost scales non-linearly with trade size (configurable)
- Toggleable via backtest configuration with parameters

#### Scenario: Apply dynamic spread by volatility
- **WHEN** trading in a high-volatility bucket
- **THEN** use the bucket’s spread parameter to compute spread cost

#### Scenario: Apply market impact by trade size
- **WHEN** trade size exceeds the configured threshold
- **THEN** add extra cost per the non-linear impact function

#### Scenario: Backtest configuration toggles model
- **WHEN** the dynamic cost model is enabled in backtest config
- **THEN** the engine uses v2 costs; when disabled, it falls back to the fixed-cost model

