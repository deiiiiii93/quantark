## 1. Core Infrastructure

- [ ] 1.1 Create `var/` module directory structure
- [ ] 1.2 Implement `VaRMethod` enum in `var/config.py`
- [ ] 1.3 Implement `EquityRiskFactorConfig` dataclass
- [ ] 1.4 Implement `FIRiskFactorConfig` dataclass
- [ ] 1.5 Implement `VaRConfig` dataclass with validation
- [ ] 1.6 Implement `VaREngine` protocol in `var/base.py`
- [ ] 1.7 Implement `VaRResult` dataclass in `var/results/var_result.py`
- [ ] 1.8 Create `var/__init__.py` with public API exports
- [ ] 1.9 Write unit tests for config validation

## 2. Risk Factor Infrastructure

- [ ] 2.1 Implement `RiskFactor` protocol in `var/risk_factors/base.py`
- [ ] 2.2 Implement `SpotReturnFactor` in `var/risk_factors/equity_factors.py`
- [ ] 2.3 Implement `VolChangeFactor` in `var/risk_factors/equity_factors.py`
- [ ] 2.4 Implement `RateShiftFactor` in `var/risk_factors/equity_factors.py`
- [ ] 2.5 Implement `DivYieldShiftFactor` in `var/risk_factors/equity_factors.py`
- [ ] 2.6 Implement `ParallelShiftFactor` in `var/risk_factors/fi_factors.py`
- [ ] 2.7 Implement `KeyRateShiftFactor` in `var/risk_factors/fi_factors.py`
- [ ] 2.8 Write unit tests for risk factor extraction from historical data

## 3. Parametric VaR Engine

- [ ] 3.1 Create `var/engines/parametric.py`
- [ ] 3.2 Implement covariance matrix construction from historical data
- [ ] 3.3 Implement sensitivity vector extraction for equity portfolios (delta, gamma, vega)
- [ ] 3.4 Implement sensitivity vector extraction for FI portfolios (DV01, duration)
- [ ] 3.5 Implement delta-gamma approximation for P&L distribution
- [ ] 3.6 Implement VaR calculation from normal distribution quantile
- [ ] 3.7 Implement CVaR calculation for parametric method
- [ ] 3.8 Implement multi-day scaling (sqrt_t method)
- [ ] 3.9 Write unit tests for parametric VaR with known analytical results

## 4. Historical VaR Engine

- [ ] 4.1 Create `var/engines/historical.py`
- [ ] 4.2 Implement scenario extraction from `MarketDataSet`
- [ ] 4.3 Implement scenario extraction from `pd.DataFrame`
- [ ] 4.4 Implement stressed `PricingEnvironment` creation for each scenario
- [ ] 4.5 Implement full portfolio revaluation loop
- [ ] 4.6 Implement empirical VaR from P&L distribution (percentile)
- [ ] 4.7 Implement empirical CVaR (Expected Shortfall)
- [ ] 4.8 Implement overlapping returns for multi-day VaR
- [ ] 4.9 Store scenario details in result for backtesting
- [ ] 4.10 Write unit tests for historical VaR with synthetic data

## 5. Monte Carlo VaR Engine

- [ ] 5.1 Create `var/engines/monte_carlo.py`
- [ ] 5.2 Implement multivariate distribution fitting to historical data
- [ ] 5.3 Implement Cholesky decomposition for correlated scenario generation
- [ ] 5.4 Implement scenario generator with configurable seed
- [ ] 5.5 Implement full portfolio revaluation for each simulated scenario
- [ ] 5.6 Implement VaR and CVaR from simulated P&L distribution
- [ ] 5.7 Implement worst scenarios extraction
- [ ] 5.8 Write unit tests for Monte Carlo VaR convergence

## 6. VaR Attribution

- [ ] 6.1 Implement component VaR calculation (Euler allocation)
- [ ] 6.2 Implement marginal VaR calculation (incremental contribution)
- [ ] 6.3 Implement factor VaR attribution (risk factor decomposition)
- [ ] 6.4 Add attribution results to `VaRResult`
- [ ] 6.5 Write unit tests for VaR attribution (sum to total VaR)

## 7. Stressed VaR

- [ ] 7.1 Add `calculate_stressed_var`, `stressed_period_start`, `stressed_period_end` to `VaRConfig`
- [ ] 7.2 Implement auto-detection of highest volatility 12-month period
- [ ] 7.3 Implement SVaR calculation using stressed period scenarios
- [ ] 7.4 Add `stressed_var`, `stressed_cvar`, `stressed_period` to `VaRResult`
- [ ] 7.5 Write unit tests for SVaR with known crisis periods

## 8. VaR Backtesting

- [ ] 8.1 Create `var/backtest/` subdirectory structure
- [ ] 8.2 Implement `VaRBacktestResult` dataclass
- [ ] 8.3 Implement `VaRBacktester.run_backtest()` main loop
- [ ] 8.4 Implement Kupiec POF (Proportion of Failures) test
- [ ] 8.5 Implement Christoffersen conditional coverage test
- [ ] 8.6 Implement Basel traffic light zone classification
- [ ] 8.7 Implement exception date tracking and details
- [ ] 8.8 Write unit tests for backtesting with synthetic exceptions

## 9. Incremental VaR

- [ ] 9.1 Add `calculate_incremental_var` to `VaRConfig`
- [ ] 9.2 Implement `IncrementalVaRResult` dataclass
- [ ] 9.3 Implement incremental VaR calculation (full portfolio vs excluding position)
- [ ] 9.4 Add `incremental_var` dict to `VaRResult`
- [ ] 9.5 Implement single-position incremental VaR query method
- [ ] 9.6 Write unit tests for incremental VaR (diversification scenarios)

## 10. Reporting and Integration

- [ ] 10.1 Create `var/results/var_report.py`
- [ ] 10.2 Implement `VaRReportGenerator.generate_summary()`
- [ ] 10.3 Implement `VaRReportGenerator.generate_position_report()`
- [ ] 10.4 Implement `VaRReportGenerator.generate_factor_report()`
- [ ] 10.5 Implement `VaRReportGenerator.generate_backtest_report()`
- [ ] 10.6 Add example script `example/portfolio_var_demo.py`
- [ ] 10.7 Update `var/__init__.py` with complete exports

## 11. Documentation and Final Validation

- [ ] 11.1 Add docstrings to all public classes and methods
- [ ] 11.2 Create `var/README.md` with usage examples
- [ ] 11.3 Run full test suite and verify coverage
- [ ] 11.4 Validate against known benchmark VaR values
- [ ] 11.5 Validate backtesting against published test cases

