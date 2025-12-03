## 1. Core Infrastructure

- [x] 1.1 Create `var/` module directory structure
- [x] 1.2 Implement `VaRMethod` enum in `var/config.py`
- [x] 1.3 Implement `EquityRiskFactorConfig` dataclass
- [x] 1.4 Implement `FIRiskFactorConfig` dataclass
- [x] 1.5 Implement `VaRConfig` dataclass with validation
- [x] 1.6 Implement `VaREngine` protocol in `var/base.py`
- [x] 1.7 Implement `VaRResult` dataclass in `var/results/var_result.py`
- [x] 1.8 Create `var/__init__.py` with public API exports
- [x] 1.9 Write unit tests for config validation

## 2. Risk Factor Infrastructure

- [x] 2.1 Implement `RiskFactor` protocol in `var/risk_factors/base.py`
- [x] 2.2 Implement `SpotReturnFactor` in `var/risk_factors/equity_factors.py`
- [x] 2.3 Implement `VolChangeFactor` in `var/risk_factors/equity_factors.py`
- [x] 2.4 Implement `RateShiftFactor` in `var/risk_factors/equity_factors.py`
- [x] 2.5 Implement `DivYieldShiftFactor` in `var/risk_factors/equity_factors.py`
- [x] 2.6 Implement `ParallelShiftFactor` in `var/risk_factors/fi_factors.py`
- [x] 2.7 Implement `KeyRateShiftFactor` in `var/risk_factors/fi_factors.py`
- [x] 2.8 Write unit tests for risk factor extraction from historical data

## 3. Parametric VaR Engine

- [x] 3.1 Create `var/engines/parametric.py`
- [x] 3.2 Implement covariance matrix construction from historical data
- [x] 3.3 Implement sensitivity vector extraction for equity portfolios (delta, gamma, vega)
- [x] 3.4 Implement sensitivity vector extraction for FI portfolios (DV01, duration)
- [x] 3.5 Implement delta-gamma approximation for P&L distribution
- [x] 3.6 Implement VaR calculation from normal distribution quantile
- [x] 3.7 Implement CVaR calculation for parametric method
- [x] 3.8 Implement multi-day scaling (sqrt_t method)
- [x] 3.9 Write unit tests for parametric VaR with known analytical results

## 4. Historical VaR Engine

- [x] 4.1 Create `var/engines/historical.py`
- [x] 4.2 Implement scenario extraction from `MarketDataSet`
- [x] 4.3 Implement scenario extraction from `pd.DataFrame`
- [x] 4.4 Implement stressed `PricingEnvironment` creation for each scenario
- [x] 4.5 Implement full portfolio revaluation loop
- [x] 4.6 Implement empirical VaR from P&L distribution (percentile)
- [x] 4.7 Implement empirical CVaR (Expected Shortfall)
- [x] 4.8 Implement overlapping returns for multi-day VaR
- [x] 4.9 Store scenario details in result for backtesting
- [x] 4.10 Write unit tests for historical VaR with synthetic data

## 5. Monte Carlo VaR Engine

- [x] 5.1 Create `var/engines/monte_carlo.py`
- [x] 5.2 Implement multivariate distribution fitting to historical data
- [x] 5.3 Implement Cholesky decomposition for correlated scenario generation
- [x] 5.4 Implement scenario generator with configurable seed
- [x] 5.5 Implement full portfolio revaluation for each simulated scenario
- [x] 5.6 Implement VaR and CVaR from simulated P&L distribution
- [x] 5.7 Implement worst scenarios extraction
- [x] 5.8 Write unit tests for Monte Carlo VaR convergence

## 6. VaR Attribution

- [x] 6.1 Implement component VaR calculation (Euler allocation)
- [x] 6.2 Implement marginal VaR calculation (incremental contribution)
- [x] 6.3 Implement factor VaR attribution (risk factor decomposition)
- [x] 6.4 Add attribution results to `VaRResult`
- [x] 6.5 Write unit tests for VaR attribution (sum to total VaR)

## 7. Stressed VaR

- [x] 7.1 Add `calculate_stressed_var`, `stressed_period_start`, `stressed_period_end` to `VaRConfig`
- [x] 7.2 Implement auto-detection of highest volatility 12-month period
- [x] 7.3 Implement SVaR calculation using stressed period scenarios
- [x] 7.4 Add `stressed_var`, `stressed_cvar`, `stressed_period` to `VaRResult`
- [x] 7.5 Write unit tests for SVaR with known crisis periods

## 8. VaR Backtesting

- [x] 8.1 Create `var/backtest/` subdirectory structure
- [x] 8.2 Implement `VaRBacktestResult` dataclass
- [x] 8.3 Implement `VaRBacktester.run_backtest()` main loop
- [x] 8.4 Implement Kupiec POF (Proportion of Failures) test
- [x] 8.5 Implement Christoffersen conditional coverage test
- [x] 8.6 Implement Basel traffic light zone classification
- [x] 8.7 Implement exception date tracking and details
- [x] 8.8 Write unit tests for backtesting with synthetic exceptions

## 9. Incremental VaR

- [x] 9.1 Add `calculate_incremental_var` to `VaRConfig`
- [x] 9.2 Implement `IncrementalVaRResult` dataclass
- [x] 9.3 Implement incremental VaR calculation (full portfolio vs excluding position)
- [x] 9.4 Add `incremental_var` dict to `VaRResult`
- [x] 9.5 Implement single-position incremental VaR query method
- [x] 9.6 Write unit tests for incremental VaR (diversification scenarios)

## 10. Reporting and Integration

- [x] 10.1 Create `var/results/var_report.py`
- [x] 10.2 Implement `VaRReportGenerator.generate_summary()`
- [x] 10.3 Implement `VaRReportGenerator.generate_position_report()`
- [x] 10.4 Implement `VaRReportGenerator.generate_factor_report()`
- [x] 10.5 Implement `VaRReportGenerator.generate_backtest_report()`
- [x] 10.6 Add example script `example/portfolio_var_demo.py`
- [x] 10.7 Update `var/__init__.py` with complete exports

## 11. Documentation and Final Validation

- [x] 11.1 Add docstrings to all public classes and methods
- [x] 11.2 Create `var/README.md` with usage examples
- [x] 11.3 Run full test suite and verify coverage
- [x] 11.4 Validate against known benchmark VaR values
- [x] 11.5 Validate backtesting against published test cases