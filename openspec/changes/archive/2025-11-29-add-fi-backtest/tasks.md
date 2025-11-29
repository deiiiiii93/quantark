## 1. Create Base Protocols

- [x] 1.1 Create `portfolio/base.py` with `BasePosition` and `BasePortfolio` protocols
- [x] 1.2 Create `backtest/base.py` with `BaseHedgeExecutor` and `BaseBacktestEngine` protocols
- [x] 1.3 Update `backtest/strategy/base_strategy.py` to support configurable hedging targets

## 2. Refactor Existing Equity Implementation into Subfolder

- [x] 2.1 Create `portfolio/equity/` directory structure
- [x] 2.2 Move `portfolio/position.py` → `portfolio/equity/position.py` (as `EquityPosition`)
- [x] 2.3 Move `portfolio/portfolio.py` → `portfolio/equity/portfolio.py` (as `EquityPortfolio`)
- [x] 2.4 Create `portfolio/equity/__init__.py` with exports
- [x] 2.5 Update `portfolio/__init__.py` with backward-compatible aliases
- [x] 2.6 Create `backtest/equity/` directory structure
- [x] 2.7 Move `backtest/engine.py` → `backtest/equity/engine.py` (as `EquityBacktestEngine`)
- [x] 2.8 Move `backtest/hedge_executor.py` → `backtest/equity/hedge_executor.py`
- [x] 2.9 Move `backtest/config.py` → `backtest/equity/config.py`
- [x] 2.10 Move `backtest/state.py` → `backtest/equity/state.py`
- [x] 2.11 Move `backtest/results.py` → `backtest/equity/results.py`
- [x] 2.12 Move `backtest/metrics.py` → `backtest/equity/metrics.py`
- [x] 2.13 Create `backtest/equity/__init__.py` with exports
- [x] 2.14 Update `backtest/__init__.py` with backward-compatible aliases

## 3. Implement Fixed Income Portfolio

- [x] 3.1 Create `portfolio/fi/__init__.py` with exports
- [x] 3.2 Implement `portfolio/fi/position.py` with `FIPosition` class
- [x] 3.3 Implement `portfolio/fi/portfolio.py` with `FIPortfolio` class
- [x] 3.4 Add FI risk measure calculations (DV01, convexity, modified duration)
- [x] 3.5 Update `portfolio/__init__.py` to export FI classes

## 4. Implement Fixed Income Hedging Strategies

- [x] 4.1 Create `backtest/strategy/dv01_neutral_strategy.py` with `DV01NeutralStrategy`
- [x] 4.2 Create `backtest/strategy/convexity_neutral_strategy.py` with `ConvexityNeutralStrategy`
- [x] 4.3 Update `backtest/strategy/__init__.py` with new strategy exports

## 5. Implement Fixed Income Backtest Components

- [x] 5.1 Create `backtest/fi/__init__.py` with exports
- [x] 5.2 Implement `backtest/fi/config.py` with `FIBacktestConfig`
- [x] 5.3 Implement `backtest/fi/state.py` with FI-specific state tracking
- [x] 5.4 Implement `backtest/fi/hedge_executor.py` with `FIHedgeExecutor` for bond futures
- [x] 5.5 Implement `backtest/fi/engine.py` with `FIBacktestEngine`
- [x] 5.6 Implement `backtest/fi/results.py` with `FIBacktestResults`
- [x] 5.7 Implement `backtest/fi/metrics.py` with FI-specific metrics

## 6. Extend Market Data Support

- [x] 6.1 Extend `MockMarketDataAdapter` to support FI market data (rate curves, bond prices)
- [ ] 6.2 Create FI-specific market data set class (optional - using existing MarketDataSet)

## 7. Create Fixed Income Backtest Example

- [x] 7.1 Create `backtest/examples/fi_dv01_hedge.py` with bond portfolio + futures hedging
- [x] 7.2 Include comparison between hedged and unhedged portfolios
- [ ] 7.3 Generate DV01 tracking visualizations (deferred to Phase 8)

## 8. Update Visualization and Reporting

- [x] 8.1 Extend `StaticVisualizer` to support FI metrics (DV01 tracking, yield curve sensitivity)
- [x] 8.2 Extend `InteractiveDashboard` for FI-specific plots
- [x] 8.3 Update `ReportGenerator` for FI backtest reports

## 9. Documentation and Tests

- [x] 9.1 Update `backtest/README.md` with FI backtest documentation
- [ ] 9.2 Create unit tests for FI portfolio components (future enhancement)
- [ ] 9.3 Create unit tests for FI backtest engine (future enhancement)
- [ ] 9.4 Create integration tests for full FI backtest workflow (future enhancement)

