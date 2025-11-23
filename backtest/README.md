# QuantArk Backtest Module

A comprehensive backtesting framework for delta-neutral hedging strategies with advanced features including transaction cost modeling, comprehensive logging, and rich visualizations.

## Overview

The backtest module allows you to simulate hedging strategies over historical or synthetic market data, providing detailed analytics on strategy performance, hedging effectiveness, and risk metrics.

### Key Features

- **Delta-Neutral Strategy**: Automated delta hedging with configurable parameters
- **Transaction Cost Modeling**: Multiple cost models (fixed, proportional, slippage, bid-ask spread)
- **Comprehensive Logging**: Multi-level logging for trades, state, events, and performance
- **Rich Visualizations**: Both static (matplotlib) and interactive (plotly) visualizations
- **Performance Metrics**: Sharpe ratio, max drawdown, VaR, CVaR, and hedging-specific metrics
- **Report Generation**: Automated HTML and text reports with embedded visualizations
- **Flexible Data Sources**: Support for historical data and synthetic (mock) data

## Architecture

The module follows a modular design with the following components:

```
backtest/
├── engine.py                   # Main backtest orchestrator
├── config.py                   # Configuration management
├── state.py                    # State tracking and history
├── hedge_executor.py           # Hedge execution logic
├── transaction_costs.py        # Cost modeling
├── logger.py                   # Logging infrastructure
├── results.py                  # Results container
├── metrics.py                  # Performance metrics
├── visualizer.py               # Static visualizations
├── dashboard.py                # Interactive dashboard
├── report_generator.py         # Report generation
└── strategy/
    ├── base_strategy.py        # Abstract strategy base
    └── delta_neutral_strategy.py  # Delta-neutral implementation
```

## Quick Start

### Basic Example

```python
from datetime import datetime
from backtest import (
    BacktestEngine,
    BacktestConfig,
    DeltaNeutralStrategy,
    ZeroCostModel
)
from portfolio import Position
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical import BlackScholesEngine
from util.enum import OptionType
from util.marketdata.adapter import MockMarketDataAdapter

# Create option position
option = EuropeanVanillaOption(
    strike=100.0,
    option_type=OptionType.CALL,
    maturity=1.0
)

initial_position = Position(
    product=option,
    quantity=100,
    entry_price=10.0,
    underlying="AAPL",
    engine=BlackScholesEngine(),
    entry_timestamp=datetime(2024, 1, 1)
)

# Configure delta-neutral strategy
strategy = DeltaNeutralStrategy(
    name="BasicDN",
    delta_threshold=50.0,
    rebalance_frequency='daily',
    hedge_instrument='spot'
)

# Configure backtest
config = BacktestConfig(
    strategy=strategy,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30),
    underlying="AAPL",
    initial_positions=[initial_position],
    market_data_adapter=MockMarketDataAdapter(seed=42),
    transaction_cost_model=ZeroCostModel()
)

# Run backtest
engine = BacktestEngine(config)
results = engine.run()

# Display results
print(f"Total P&L: ${results.get_total_pnl():,.2f}")
print(f"Total Return: {results.get_total_return():.2%}")
print(f"Sharpe Ratio: {results.metrics.sharpe_ratio():.3f}")
print(f"Max Drawdown: {results.metrics.max_drawdown():.2%}")
```

## Strategy Configuration

### Delta-Neutral Strategy

The `DeltaNeutralStrategy` monitors portfolio delta and triggers hedges based on configurable parameters:

```python
strategy = DeltaNeutralStrategy(
    name="MyStrategy",
    delta_threshold=100.0,          # Hedge when |delta| > 100
    rebalance_frequency='daily',    # 'daily', 'hourly', 'on_threshold', 'continuous'
    hedge_instrument='spot',         # 'spot' or 'futures'
    hedge_ratio=1.0,                 # Proportion to hedge (0-1)
    target_delta=0.0,                # Target delta after hedging
    min_time_between_hedges=None    # Optional minimum time between hedges
)
```

**Parameters:**

- `delta_threshold`: Absolute delta level that triggers a hedge
- `rebalance_frequency`: When to check for hedging opportunities
  - `'daily'`: Once per day
  - `'hourly'`: Once per hour
  - `'on_threshold'`: Only when threshold is breached
  - `'continuous'`: Check at every timestep
- `hedge_instrument`: Type of instrument to use for hedging
  - `'spot'`: Use spot/stock
  - `'futures'`: Use futures contracts
- `hedge_ratio`: Proportion of delta to hedge (1.0 = full hedge, 0.5 = half)
- `target_delta`: Target delta level after hedging (typically 0.0)
- `min_time_between_hedges`: Optional minimum time between hedges to avoid over-trading

## Transaction Cost Models

### Zero Cost (Frictionless)

```python
from backtest import ZeroCostModel

cost_model = ZeroCostModel()
```

### Fixed Commission

```python
from backtest import FixedCostModel

cost_model = FixedCostModel(
    commission_per_trade=2.0  # $2 per trade
)
```

### Proportional Commission

```python
from backtest import ProportionalCostModel

cost_model = ProportionalCostModel(
    commission_rate=0.0005  # 5 basis points
)
```

### Complete Cost Model

Combines all cost components:

```python
from backtest import CompleteCostModel

cost_model = CompleteCostModel(
    fixed_commission=2.0,           # $2 per trade
    proportional_rate=0.0005,       # 5 bps
    slippage_coefficient=0.0001,    # Slippage impact
    slippage_type='linear',         # 'linear' or 'sqrt'
    spread_bps=5.0                  # 5 bps bid-ask spread
)
```

## Results and Analysis

### Accessing Results

```python
# Run backtest
results = engine.run()

# Summary statistics
summary = results.get_summary()
print(f"Total P&L: ${summary['total_pnl']:,.2f}")
print(f"Number of Hedges: {summary['num_hedges']}")

# Time series data
pnl_series = results.get_pnl_series()
value_series = results.get_value_series()
delta_series = results.get_delta_series()

# Trade history
trades_df = results.get_hedge_trades()
```

### Performance Metrics

```python
metrics = results.metrics

# P&L metrics
sharpe = metrics.sharpe_ratio()
max_dd = metrics.max_drawdown()
win_rate = metrics.win_rate()

# Hedging metrics
hedge_freq = metrics.hedge_frequency()
delta_tracking_error = metrics.delta_tracking_error()

# Risk metrics
var_95 = metrics.value_at_risk(0.95)
cvar_95 = metrics.conditional_var(0.95)

# All metrics
all_metrics = metrics.calculate_all_metrics()
```

## Visualization

### Static Plots (Matplotlib)

```python
from backtest import StaticVisualizer

visualizer = StaticVisualizer(results, save_dir="plots")

# Individual plots
visualizer.plot_pnl_over_time(save=True)
visualizer.plot_delta_tracking(save=True)
visualizer.plot_greeks_evolution(save=True)
visualizer.plot_drawdown(save=True)

# Comprehensive dashboard
visualizer.create_summary_dashboard(save=True)

# Generate all plots
visualizer.generate_all_plots(save=True)
```

### Interactive Dashboard (Plotly)

```python
from backtest import InteractiveDashboard

dashboard = InteractiveDashboard(results, save_dir="plots/interactive")

# Interactive plots
dashboard.plot_pnl_interactive(save=True)
dashboard.plot_delta_tracking_interactive(save=True)
dashboard.plot_greeks_interactive(save=True)

# Comprehensive dashboard
dashboard.create_comprehensive_dashboard(save=True)

# Generate all interactive plots
dashboard.generate_all_interactive_plots(save=True)
```

## Report Generation

### HTML Report

```python
from backtest import ReportGenerator

report_gen = ReportGenerator(results, output_dir="reports")

# Generate comprehensive HTML report
html_path = report_gen.generate_html_report()
print(f"Report saved to: {html_path}")
```

### Text Report

```python
# Generate text report
text_path = report_gen.generate_text_report()
```

### Export Results

```python
# Export to Excel
results.export_to_excel("results.xlsx")

# Export to Parquet
results.export_to_parquet("results.parquet")
```

## Advanced Features

### Custom Strategy

Create your own strategy by extending `BaseStrategy`:

```python
from backtest.strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def should_hedge(self, current_time, portfolio_greeks, market_data, **kwargs):
        # Your logic here
        return True or False
    
    def calculate_hedge_size(self, current_time, portfolio_greeks, market_data, **kwargs):
        # Your logic here
        return hedge_size
    
    def get_parameters(self):
        return {'param1': value1, 'param2': value2}
```

### Market Data

Use mock (synthetic) data or implement your own adapter:

```python
from util.marketdata.adapter import MockMarketDataAdapter

# Mock data with custom configuration
adapter = MockMarketDataAdapter(seed=42)
adapter.set_asset_config(
    "AAPL",
    {
        'initial_spot': 150.0,
        'initial_vol': 0.25,
        'drift': 0.10,
        'vol_of_vol': 0.3
    }
)
```

## Performance Metrics Reference

### P&L Metrics

- `total_pnl()`: Total profit/loss
- `total_return()`: Total return as decimal
- `sharpe_ratio()`: Risk-adjusted return measure
- `max_drawdown()`: Maximum peak-to-trough decline
- `win_rate()`: Proportion of profitable periods
- `profit_factor()`: Gross profit / gross loss

### Hedging Metrics

- `hedge_frequency()`: Average hedges per day
- `average_hedge_cost()`: Average transaction cost per hedge
- `delta_tracking_error()`: RMSE of delta vs target
- `average_absolute_delta()`: Mean absolute delta

### Risk Metrics

- `value_at_risk(confidence)`: VaR at confidence level
- `conditional_var(confidence)`: CVaR (Expected Shortfall)
- `volatility()`: Annualized volatility
- `skewness()`: Distribution skewness
- `kurtosis()`: Distribution kurtosis

## Examples

See the `backtest/examples/` directory for complete examples:

- `basic_delta_hedge.py`: Simple delta-neutral hedging
- `advanced_backtest.py`: Advanced features with transaction costs

Run examples:

```bash
python backtest/examples/basic_delta_hedge.py
python backtest/examples/advanced_backtest.py
```

## Testing

Run unit tests:

```bash
pytest test/test_backtest.py -v
```

## Dependencies

The backtest module requires:

- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.14.0
- kaleido >= 0.2.1

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Best Practices

1. **Start Simple**: Begin with `ZeroCostModel` to understand strategy behavior
2. **Add Costs Gradually**: Introduce transaction costs incrementally
3. **Monitor Delta Tracking**: Keep an eye on `delta_tracking_error`
4. **Adjust Threshold**: Tune `delta_threshold` based on hedging frequency vs costs
5. **Use Logging**: Set `logging_level='DEBUG'` for detailed insights
6. **Save Results**: Always save results for later analysis
7. **Compare Strategies**: Run multiple backtests with different parameters

## Troubleshooting

### High Transaction Costs

- Increase `delta_threshold` to reduce hedge frequency
- Adjust `hedge_ratio` to partial hedge
- Set `min_time_between_hedges` to avoid over-trading

### Poor Delta Tracking

- Decrease `delta_threshold` for tighter control
- Increase `rebalance_frequency` to 'continuous'
- Check if `hedge_ratio` is too low

### Memory Issues

- Reduce backtest period
- Decrease data frequency (use weekly instead of daily)
- Set `save_snapshots=False` in config

## Contributing

The backtest module is designed to be extensible. Key extension points:

- Custom strategies: Extend `BaseStrategy`
- Custom cost models: Extend `TransactionCostModel`
- Custom visualizations: Use `results` data directly
- Custom metrics: Access `results.states_df` and `results.trades_df`

## License

Part of the QuantArk quantitative finance library.

## Support

For issues, questions, or contributions, please refer to the main QuantArk repository.

