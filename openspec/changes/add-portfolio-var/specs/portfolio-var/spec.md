## ADDED Requirements

### Requirement: VaR Configuration
The system SHALL provide a `VaRConfig` dataclass for configuring Value-at-Risk calculations with all necessary parameters.

#### Scenario: Default configuration
- **WHEN** a `VaRConfig` is instantiated without arguments
- **THEN** it SHALL default to 99% confidence level, 1-day holding period, 252-day lookback, and PARAMETRIC method

#### Scenario: Custom confidence level
- **WHEN** a `VaRConfig` is instantiated with `confidence_level=0.95`
- **THEN** VaR calculations SHALL use 95% confidence level

#### Scenario: Multi-day VaR configuration
- **WHEN** a `VaRConfig` is instantiated with `holding_period=10`
- **THEN** VaR calculations SHALL compute 10-day VaR using the configured scaling method

#### Scenario: Invalid confidence level validation
- **WHEN** a `VaRConfig` is instantiated with `confidence_level=1.5`
- **THEN** the system SHALL raise `ValidationError`

#### Scenario: Risk factor configuration for equity
- **WHEN** `EquityRiskFactorConfig` is provided with `include_spot=True, include_vol=True`
- **THEN** VaR calculations SHALL consider spot and volatility risk factors

#### Scenario: Risk factor configuration for fixed income
- **WHEN** `FIRiskFactorConfig` is provided with `include_key_rates=True`
- **THEN** VaR calculations SHALL consider key-rate risk factors at specified tenors

---

### Requirement: VaR Method Enumeration
The system SHALL provide a `VaRMethod` enum with three supported VaR calculation methods.

#### Scenario: Available VaR methods
- **WHEN** a user queries `VaRMethod`
- **THEN** it SHALL include `PARAMETRIC`, `HISTORICAL`, and `MONTE_CARLO` options

#### Scenario: Method selection in config
- **WHEN** `VaRConfig(var_method=VaRMethod.HISTORICAL)` is specified
- **THEN** the VaR engine SHALL use historical simulation with full revaluation

---

### Requirement: VaR Engine Protocol
The system SHALL define a `VaREngine` protocol that all VaR implementations MUST follow.

#### Scenario: Engine interface compliance
- **WHEN** a new VaR engine class is implemented
- **THEN** it MUST implement `calculate_var(portfolio, historical_data) -> VaRResult`

#### Scenario: Portfolio type support check
- **WHEN** `engine.supports_portfolio(portfolio)` is called
- **THEN** it SHALL return `True` if the engine can process the portfolio type, `False` otherwise

---

### Requirement: Parametric VaR Engine
The system SHALL provide a `ParametricVaREngine` implementing variance-covariance VaR using portfolio sensitivities.

#### Scenario: Equity portfolio parametric VaR
- **WHEN** `ParametricVaREngine.calculate_var()` is called with an `EquityPortfolio`
- **THEN** it SHALL compute VaR using delta, gamma, vega sensitivities and historical covariance matrix
- **AND** return a `VaRResult` with the VaR value

#### Scenario: Fixed income portfolio parametric VaR
- **WHEN** `ParametricVaREngine.calculate_var()` is called with an `FIPortfolio`
- **THEN** it SHALL compute VaR using DV01 and duration sensitivities and rate covariance matrix

#### Scenario: Covariance matrix construction
- **WHEN** historical data spans 252 days
- **THEN** the engine SHALL construct a covariance matrix from the risk factor returns

#### Scenario: Delta-gamma approximation for options
- **WHEN** the portfolio contains options (non-linear instruments)
- **THEN** the engine SHALL use delta-gamma approximation for P&L estimation

---

### Requirement: Historical VaR Engine
The system SHALL provide a `HistoricalVaREngine` implementing VaR via full portfolio revaluation on historical scenarios.

#### Scenario: Full revaluation under historical scenarios
- **WHEN** `HistoricalVaREngine.calculate_var()` is called
- **THEN** it SHALL create a stressed `PricingEnvironment` for each historical date
- **AND** reprice the entire portfolio under each scenario
- **AND** compute VaR from the empirical P&L distribution

#### Scenario: Historical VaR from MarketDataSet
- **WHEN** historical data is provided as a `MarketDataSet`
- **THEN** the engine SHALL extract risk factor changes from spot, vol, rate time series

#### Scenario: Historical VaR from DataFrame
- **WHEN** historical data is provided as a `pd.DataFrame` with columns ['spot_return', 'vol_change', 'rate_shift']
- **THEN** the engine SHALL construct scenarios directly from the DataFrame

#### Scenario: CVaR (Expected Shortfall) calculation
- **WHEN** VaR is calculated
- **THEN** the engine SHALL also compute CVaR as the mean of losses exceeding VaR

---

### Requirement: Monte Carlo VaR Engine
The system SHALL provide a `MonteCarloVaREngine` implementing VaR via simulated scenarios with full revaluation.

#### Scenario: Correlated scenario generation
- **WHEN** `MonteCarloVaREngine.calculate_var()` is called
- **THEN** it SHALL fit a multivariate distribution to historical risk factor changes
- **AND** generate correlated scenarios using Cholesky decomposition

#### Scenario: Configurable simulation count
- **WHEN** `VaRConfig(mc_num_simulations=50000)` is specified
- **THEN** the engine SHALL generate 50,000 simulated scenarios

#### Scenario: Reproducible simulations
- **WHEN** `VaRConfig(mc_seed=42)` is specified
- **THEN** repeated VaR calculations SHALL produce identical results

#### Scenario: Full revaluation per scenario
- **WHEN** simulating scenarios
- **THEN** the engine SHALL apply each scenario's risk factor shocks to the pricing environment
- **AND** reprice the portfolio for each scenario

---

### Requirement: VaR Result Structure
The system SHALL provide a `VaRResult` dataclass containing comprehensive VaR calculation outputs.

#### Scenario: Core VaR metrics
- **WHEN** a VaR calculation completes
- **THEN** the `VaRResult` SHALL contain `var`, `cvar`, `confidence_level`, `holding_period`, and `method`

#### Scenario: VaR as percentage of portfolio
- **WHEN** `VaRResult` is returned
- **THEN** it SHALL include `var_as_pct` computed as `var / portfolio_value`

#### Scenario: Component VaR by position
- **WHEN** `VaRConfig(calculate_component_var=True)`
- **THEN** `VaRResult.component_var` SHALL be a dict mapping `position_id` to its VaR contribution

#### Scenario: Marginal VaR by position
- **WHEN** `VaRConfig(calculate_marginal_var=True)`
- **THEN** `VaRResult.marginal_var` SHALL be a dict mapping `position_id` to its marginal VaR

#### Scenario: Factor VaR attribution
- **WHEN** `VaRConfig(calculate_factor_var=True)`
- **THEN** `VaRResult.factor_var` SHALL be a dict mapping risk factor names to their VaR contributions

#### Scenario: Scenario details for backtesting
- **WHEN** Historical or Monte Carlo method is used
- **THEN** `VaRResult.scenarios` SHALL contain a DataFrame of all scenario P&L values
- **AND** `VaRResult.worst_scenarios` SHALL contain details of the N worst scenarios

---

### Requirement: Multi-Day VaR Scaling
The system SHALL support scaling 1-day VaR to multi-day horizons.

#### Scenario: Square-root-of-time scaling
- **WHEN** `VaRConfig(holding_period=10, scaling_method="sqrt_t")`
- **THEN** VaR_10 SHALL equal VaR_1 multiplied by sqrt(10)

#### Scenario: Overlapping returns scaling
- **WHEN** `VaRConfig(holding_period=10, scaling_method="overlapping")`
- **THEN** the engine SHALL compute VaR directly from 10-day overlapping historical returns

---

### Requirement: Equity Risk Factors
The system SHALL support equity-specific risk factors for VaR calculation.

#### Scenario: Spot price risk factor
- **WHEN** `EquityRiskFactorConfig(include_spot=True)`
- **THEN** VaR calculation SHALL include spot price return as a risk factor

#### Scenario: Implied volatility risk factor
- **WHEN** `EquityRiskFactorConfig(include_vol=True)`
- **THEN** VaR calculation SHALL include implied volatility change as a risk factor

#### Scenario: Interest rate risk factor
- **WHEN** `EquityRiskFactorConfig(include_rate=True)`
- **THEN** VaR calculation SHALL include interest rate shift as a risk factor

#### Scenario: Dividend yield risk factor
- **WHEN** `EquityRiskFactorConfig(include_div_yield=True)`
- **THEN** VaR calculation SHALL include dividend yield change as a risk factor

---

### Requirement: Fixed Income Risk Factors
The system SHALL support fixed income-specific risk factors for VaR calculation.

#### Scenario: Parallel rate shift risk factor
- **WHEN** `FIRiskFactorConfig(include_parallel_shift=True)`
- **THEN** VaR calculation SHALL include parallel yield curve shift as a risk factor

#### Scenario: Key-rate risk factors
- **WHEN** `FIRiskFactorConfig(include_key_rates=True, key_rate_tenors=[2.0, 5.0, 10.0, 30.0])`
- **THEN** VaR calculation SHALL include separate risk factors for each tenor point

---

### Requirement: VaR Report Generation
The system SHALL provide a `VaRReportGenerator` for creating formatted VaR reports.

#### Scenario: Summary report generation
- **WHEN** `VaRReportGenerator.generate_summary(var_result)` is called
- **THEN** it SHALL return a formatted dictionary with key VaR metrics

#### Scenario: Position-level report
- **WHEN** `VaRReportGenerator.generate_position_report(var_result)` is called
- **THEN** it SHALL return a DataFrame with component and marginal VaR by position

#### Scenario: Risk factor report
- **WHEN** `VaRReportGenerator.generate_factor_report(var_result)` is called
- **THEN** it SHALL return a DataFrame with VaR contribution by risk factor

---

### Requirement: Input Validation
The system SHALL validate all inputs to VaR calculations.

#### Scenario: Empty portfolio validation
- **WHEN** `calculate_var()` is called with an empty portfolio
- **THEN** the system SHALL raise `ValidationError` with a descriptive message

#### Scenario: Insufficient historical data validation
- **WHEN** historical data has fewer days than `lookback_days` in config
- **THEN** the system SHALL raise `ValidationError` indicating insufficient data

#### Scenario: Missing risk factor data validation
- **WHEN** required risk factor data (e.g., volatility history) is missing
- **THEN** the system SHALL raise `MarketDataError` with details of missing data

#### Scenario: Unsupported portfolio type
- **WHEN** a VaR engine receives a portfolio type it does not support
- **THEN** the system SHALL raise `ValidationError` indicating unsupported portfolio

---

### Requirement: Stressed VaR Calculation
The system SHALL provide Stressed VaR (SVaR) calculation using a crisis period window.

#### Scenario: SVaR with user-specified crisis period
- **WHEN** `VaRConfig(calculate_stressed_var=True, stressed_period_start=date1, stressed_period_end=date2)` is specified
- **THEN** the engine SHALL compute VaR using only historical scenarios from the specified crisis period
- **AND** return `stressed_var` in the `VaRResult`

#### Scenario: SVaR with auto-detected crisis period
- **WHEN** `VaRConfig(calculate_stressed_var=True)` is specified without explicit period dates
- **THEN** the engine SHALL automatically identify the 12-month period with highest volatility
- **AND** use that period for SVaR calculation

#### Scenario: SVaR result structure
- **WHEN** SVaR calculation completes
- **THEN** `VaRResult` SHALL include `stressed_var`, `stressed_cvar`, and `stressed_period` metadata

#### Scenario: SVaR disabled
- **WHEN** `VaRConfig(calculate_stressed_var=False)`
- **THEN** `VaRResult.stressed_var` SHALL be `None`

---

### Requirement: VaR Backtesting
The system SHALL provide VaR backtesting capabilities with statistical tests for model validation.

#### Scenario: Backtest execution
- **WHEN** `VaRBacktester.run_backtest(portfolio, historical_data, var_engine)` is called
- **THEN** it SHALL compute VaR for each day in the backtest period
- **AND** compare predicted VaR against actual realized P&L
- **AND** record exceptions where actual loss exceeded VaR

#### Scenario: Kupiec POF test
- **WHEN** backtest completes
- **THEN** `VaRBacktestResult` SHALL include `kupiec_pof_statistic`, `kupiec_pof_pvalue`, and `kupiec_pof_pass`
- **AND** the test SHALL evaluate whether exception frequency matches expected rate

#### Scenario: Christoffersen test
- **WHEN** backtest completes
- **THEN** `VaRBacktestResult` SHALL include `christoffersen_statistic`, `christoffersen_pvalue`, and `christoffersen_pass`
- **AND** the test SHALL evaluate both coverage and independence of exceptions

#### Scenario: Basel traffic light zone
- **WHEN** backtest completes with 250 observations at 99% confidence
- **THEN** `VaRBacktestResult.basel_zone` SHALL be "green" for 0-4 exceptions
- **AND** "yellow" for 5-9 exceptions
- **AND** "red" for 10+ exceptions

#### Scenario: Exception details
- **WHEN** backtest identifies VaR breaches
- **THEN** `VaRBacktestResult.exceptions_dates` SHALL contain the dates of all exceptions
- **AND** `VaRBacktestResult.exception_details` SHALL contain P&L and VaR values for each breach

---

### Requirement: Incremental VaR
The system SHALL provide Incremental VaR (IVaR) to measure the impact of adding or removing positions.

#### Scenario: Incremental VaR calculation
- **WHEN** `VaRConfig(calculate_incremental_var=True)` is specified
- **THEN** the engine SHALL compute VaR with and without each position
- **AND** return `incremental_var` dict in `VaRResult` mapping `position_id` to its incremental VaR

#### Scenario: Incremental VaR interpretation
- **WHEN** `incremental_var[position_id]` is positive
- **THEN** the position adds risk to the portfolio
- **WHEN** `incremental_var[position_id]` is negative
- **THEN** the position provides diversification benefit

#### Scenario: Incremental VaR for single position query
- **WHEN** `engine.calculate_incremental_var(portfolio, position_id, historical_data)` is called
- **THEN** it SHALL return an `IncrementalVaRResult` for that specific position only

#### Scenario: Incremental VaR disabled
- **WHEN** `VaRConfig(calculate_incremental_var=False)`
- **THEN** `VaRResult.incremental_var` SHALL be `None`

