# Change: Add Portfolio Value-at-Risk (VaR) Calculation Module

## Why

Portfolio risk management requires robust Value-at-Risk (VaR) calculations to quantify potential losses under adverse market conditions. The existing codebase lacks a dedicated VaR module - the simple `value_at_risk()` method in backtest metrics only computes historical percentiles on P&L returns without proper full revaluation, risk factor decomposition, or Monte Carlo capabilities. A professional-grade VaR module is essential for regulatory compliance, risk limit monitoring, and portfolio optimization.

## What Changes

### New Module: `var/`

A complete VaR calculation framework with:

1. **VaRConfig** - Centralized configuration for:
   - Confidence levels (90%, 95%, 99%, 99.5%)
   - Holding periods (1-day, 2-day, 10-day VaR)
   - Lookback period for historical data
   - Risk factors per asset class
   - Scaling methodology (square-root-of-time, etc.)

2. **Three VaR Methods**:
   - **Parametric VaR**: Variance-covariance approach using portfolio sensitivities (Greeks for equity, DV01/duration for FI) and historical covariance matrix
   - **Historical VaR**: Full portfolio revaluation under each historical scenario (no distributional assumptions)
   - **Monte Carlo VaR**: Simulated scenarios from fitted distributions with full revaluation

3. **Risk Factors**:
   - **Equity**: Spot returns, volatility changes, interest rate shifts, dividend yield changes
   - **Fixed Income**: Parallel rate shifts, key-rate shifts (2Y, 5Y, 10Y, 30Y tenor points)

4. **Comprehensive Results**:
   - Total portfolio VaR and CVaR (Expected Shortfall)
   - Component VaR by position
   - Marginal VaR (incremental risk contribution)
   - Incremental VaR (impact of adding/removing positions)
   - VaR contribution by risk factor
   - Scenario details for backtesting

5. **Stressed VaR (SVaR)**:
   - VaR calculated using crisis period scenarios
   - Auto-detection of highest volatility period or user-specified dates
   - Basel III regulatory compliance

6. **VaR Backtesting**:
   - Kupiec POF test (Proportion of Failures)
   - Christoffersen conditional coverage test
   - Basel traffic light zone classification (green/yellow/red)
   - Exception tracking with dates and details

5. **Data Sources**:
   - Integration with existing `MarketDataAdapter` for live historical data
   - Direct `pd.DataFrame` input for ad-hoc analysis

## Impact

### Affected Specs
- New capability: `portfolio-var`

### Affected Code
- New module: `var/` (config, engines, results, risk_factors)
- Integration point: `portfolio/equity/portfolio.py` (read-only usage)
- Integration point: `portfolio/fi/portfolio.py` (read-only usage)
- Leverages: `util/marketdata/` for historical data
- Leverages: `priceenv/` for scenario revaluation

### Dependencies
- NumPy, SciPy (already in project)
- Existing portfolio classes (`EquityPortfolio`, `FIPortfolio`)
- Existing pricing engines for full revaluation

