# QuantArk Backtest Module

A comprehensive backtesting framework for hedging strategies across multiple asset classes, with advanced features including transaction cost modeling, comprehensive logging, and rich visualizations.

## Overview

The backtest module allows you to simulate hedging strategies over historical or synthetic market data, providing detailed analytics on strategy performance, hedging effectiveness, and risk metrics.

### Supported Asset Classes

- **Equity**: Delta-neutral hedging with spot/futures instruments
- **Fixed Income**: DV01/convexity-neutral hedging with bond futures

### Key Features

- **Multi-Asset Support**: Equity derivatives and Fixed Income bonds
- **Delta-Neutral Strategy** (Equity): Automated delta hedging with configurable parameters
- **DV01-Neutral Strategy** (FI): Automated DV01 hedging with bond futures
- **Transaction Cost Modeling**: Multiple cost models (fixed, proportional, slippage, bid-ask spread)
- **Comprehensive Logging**: Multi-level logging for trades, state, events, and performance
- **Rich Visualizations**: Both static (matplotlib) and interactive (plotly) visualizations
- **Performance Metrics**: Sharpe ratio, max drawdown, VaR, CVaR, and hedging-specific metrics
- **Report Generation**: Automated HTML and text reports with embedded visualizations
- **Flexible Data Sources**: Support for historical data and synthetic (mock) data

## Architecture

The module follows a modular design with asset-specific implementations:

```
backtest/
├── base.py                     # Base protocols for all backtests
├── transaction_costs.py        # Cost modeling (shared)
├── logger.py                   # Logging infrastructure (shared)
├── visualizer.py               # Static visualizations (shared)
├── dashboard.py                # Interactive dashboard (shared)
├── report_generator.py         # Report generation (shared)
├── strategy/
│   ├── base_strategy.py        # Abstract strategy base
│   ├── delta_neutral_strategy.py  # Equity: Delta-neutral
│   ├── dv01_neutral_strategy.py   # FI: DV01-neutral
│   └── convexity_neutral_strategy.py  # FI: Convexity-neutral
├── equity/                     # Equity-specific implementation
│   ├── engine.py              # Equity backtest engine
│   ├── config.py              # Equity configuration
│   ├── state.py               # Equity state tracking
│   ├── hedge_executor.py      # Spot/futures hedging
│   ├── results.py             # Equity results
│   └── metrics.py             # Equity metrics
└── fi/                        # Fixed Income implementation
    ├── engine.py              # FI backtest engine
    ├── config.py              # FI configuration
    ├── state.py               # FI state tracking (DV01, convexity)
    ├── hedge_executor.py      # Bond futures hedging
    ├── results.py             # FI results
    └── metrics.py             # FI metrics (DV01 tracking)
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

### Equity Examples
- `basic_delta_hedge.py`: Simple delta-neutral hedging for equity derivatives
- `advanced_backtest.py`: Advanced features with transaction costs

### Fixed Income Examples
- `fi_dv01_hedge.py`: DV01-neutral hedging for bond portfolios with bond futures

Run examples:

```bash
# Equity backtest
python backtest/examples/basic_delta_hedge.py
python backtest/examples/advanced_backtest.py

# Fixed Income backtest
python backtest/examples/fi_dv01_hedge.py
```

## Fixed Income Backtest

### DV01-Neutral Strategy

The `DV01NeutralStrategy` monitors portfolio DV01 and hedges using bond futures:

```python
from backtest.fi import FIBacktestEngine, FIBacktestConfig
from backtest.strategy import DV01NeutralStrategy
from portfolio.fi import FIPosition

# Configure DV01-neutral strategy
strategy = DV01NeutralStrategy(
    name="DV01_Neutral",
    dv01_threshold=50000.0,   # Hedge when |DV01| > $50,000
    rebalance_frequency='daily',
    hedge_instrument='bond_futures',
    hedge_ratio=1.0,
    target_dv01=0.0,
    futures_dv01=1000.0,      # $1,000 DV01 per futures contract
)

# Configure FI backtest
config = FIBacktestConfig(
    strategy=strategy,
    start_date=start_date,
    end_date=end_date,
    underlying="UST_10Y",
    initial_positions=[bond_position],
    market_data_adapter=adapter,
    transaction_cost_model=cost_model,
)

# Run FI backtest
engine = FIBacktestEngine(config)
results = engine.run()

# Access FI-specific metrics
dv01_series = results.get_dv01_series()
duration_series = results.get_duration_series()
print(f"DV01 Tracking Error: ${results.metrics.dv01_tracking_error():,.0f}")
```

### FI Metrics

FI-specific metrics include:

- `dv01_tracking_error()`: RMSE of DV01 vs target
- `average_absolute_dv01()`: Mean absolute DV01 exposure
- `max_dv01_exposure()`: Maximum absolute DV01
- `dv01_hedge_effectiveness()`: Hedge effectiveness ratio (0-1)
- `average_duration()`: Portfolio weighted-average duration

## Multi-leg futures carry hedging

A single futures contract can neutralise spot delta. It cannot neutralise
the *shape* of the carry curve, because a snowball's value depends on the
whole implied dividend term structure and one contract touches one point of
it. `FuturesBucketHedgeStrategy` sizes a position in every listed contract.

### Three distinct objectives

All three start from the exact bucket positions `h_i = -B_i / m_i`, where
`B_i` is the book's currency sensitivity to contract `i`.

| Objective | Spot delta | Parallel rhoq | Gross nodal rhoq |
|---|---|---|---|
| `nodes` | `D_F` | 0 | 0 |
| `spot_far` | 0 | `D_F S T_n` | `abs(D_F S T_n)` |
| `spot_parallel` | 0 | 0 | `2 abs(K)` |

`D_F` is the spot delta with every listed quote *pinned*. It is not a
tail artefact and it is generally not zero: on the first interval both
supported builders interpolate the forward log-linearly from spot to the
first quote, so a product monitoring tomorrow against a barrier keeps
`1 - t/T_1` of its spot delta outside what the futures span. A claim
observing tomorrow with the first contract 30 days out keeps 29/30 of it.

`spot_parallel` is the primary policy. It adds a two-tenor correction that
zeroes spot delta and parallel rhoq together, and concentrates the remaining
shape risk as `+K` and `-K` on the chosen pair, where
`K = D_F S T_a T_b / (T_b - T_a)`. It requires two distinct eligible tenors
even on a date whose `D_F` happens to be negligible; fewer nodes, or an
explicit pair that is not available, is a recorded infeasibility rather than
a silent fall back to another objective.

### A minimal configuration

```python
from quantark.backtest.replay import (
    AutocallableEngineConfig, CarryRiskSettings, FuturesBucketHedgeStrategy,
    ReplayBacktestConfig, ReplayBacktestEngine, ReplayProduct,
)

config = ReplayBacktestConfig(
    products=[ReplayProduct(product=snowball, quantity=-1.0,
                            position_id=0, has_lifecycle=True)],
    market_data=dataset,
    engine_config=AutocallableEngineConfig(
        dividend_source="futures_curve",        # ACTUAL quotes, required
        futures_curve_extrapolation="flat_q",   # or "flat_forward_carry"
    ),
    strategy=FuturesBucketHedgeStrategy(objective="spot_parallel"),
    carry_audit_mode="daily",
    carry_risk_settings=CarryRiskSettings(reference_notional=50_000_000.0),
)
results = ReplayBacktestEngine(config).run()
results.hedge_legs_df()          # one row per date and coordinate
results.hedge_attribution_df()   # one daily exposure and P&L row
results.hedge_stresses_df()      # finite unquoted-carry scenarios
```

`dividend_source="futures_curve"` is mandatory, with `flat_q` or
`flat_forward_carry`. A flat carry channel has no nodes to hedge, and
`surface_forwards` is an option-implied forward — a different instrument
from the future the hedge trades, so sizing against it would leave an
unreported cross-market basis. Both are rejected at configuration time.

### Units

Every Greek in `quantark.backtest.futures_risk` is a **currency**
sensitivity, never a hand count. "Per 1%" always means one absolute
percentage point of zero yield:

```text
spot_delta_hands  = delta / m_ref
rhoq_bp_per_1pct  = 100 * rhoq / N_ref
spot_1pct_bp      = 100 * delta * S / N_ref
```

The reporting notional is gross contractual, `sum |Q_p| * initial_price_p *
contract_multiplier_p`, so a long-and-short book does not report a zero
denominator and the measure does not shrink when a product knocks out.

### Parallel versus nodal

Zero parallel rhoq is a signed sum being zero. It is compatible with large
opposing nodal exposures, and folding risk onto a far node can *create*
parallel rhoq where there was none. Report gross nodal exposure against a
gross denominator, and never read the parallel column as "the carry risk is
gone".

### What the futures cannot span

Two stress families deliberately leave spot and every listed quote
unchanged, so any futures hedge P&L in them is exactly zero:

- **independent tail** — an extra instantaneous carry yield beyond the last
  listed tenor. A 1Y claim loses about 49.88 bp of its PV to a one
  percentage-point shift with the last node at six months.
- **interpolation shape** — a log-forward bump inside one interval that
  vanishes at both anchors, including `[0, T_1]`.

A product that moves under either is carrying risk that nodal rhoq says
nothing about. These are declared research scenarios, not one-standard-
deviation market moves.

### Execution residuals

Actual holdings are `h_i = eta * h_i* + e_i`, so

```text
D_actual   = (1 - eta) D + eta D* + sum_i m_i F_i e_i / S
R_actual,i = (1 - eta) R_i + eta R_i* - m_i F_i T_i e_i
```

The recorder splits `e_i` into a rounding error and a skipped-trade error,
so their sum is the total execution error against the scaled target. A
deliberate partial hedge is not an implementation bug, and neither is a
skipped trade inside the band.

### Independent audits

Sizing uses `D_F_derived = D - sum_i (F_i/S) B_i` and
`R_i_mapped = -F_i T_i B_i`. These are identities: they cancel against
themselves and cannot detect a wrong bucket. On a scheduled audit date the
book is **repriced** with its futures quantities held fixed, and the direct
measurement is compared with the algebraic prediction for those *actual*
holdings.

Numerical validity and objective neutrality are separate results. A correct
`nodes` book passes the audit while reporting `D_F` of net spot delta,
because that residual is the policy's intent. An unrun audit is
`not_measured` with NaN measurements, never a zero.

An audit failure calls for convergence analysis, not a wider tolerance. In
this repository the audit is sharp enough to see the pricing grid: on the
standard PDE grid an engine's own delta and a central difference of its own
price can differ by about 0.009 reference hands, a gap that does not shrink
with the audit bump but roughly halves on the finer grid.

### Failure categories

| Category | Meaning | Response |
|---|---|---|
| `missing_price` | no tradable mark for a held or targeted leg | fix the data |
| `infeasible_hedge` | fewer than two tenors, or a missing explicit pair | investigate, do not retry |
| `numeric_audit` | a direct measurement disagreed | convergence analysis |

A day's rebalance is planned, marked, validated and costed in full before
the ledger moves, so none of these leaves a partial position behind.

### Not in this version

Mixed spot/futures hedges, minimum-variance optimisation, covariance
fitting, order routing, stochastic-rate or futures-convexity modelling, and
option-implied forward coordinates. The simulated-path engine rejects the
strategy: it sizes one contract per day and carries no futures chain.

## Testing

Run unit tests:

```bash
pytest test/test_backtest.py -v
```

The multi-leg carry hedge has its own suites:

```bash
pytest -n0 -q test/test_futures_risk.py test/test_carry_curve_context.py \
  test/test_futures_carry_risk.py test/test_futures_carry_audit.py \
  test/test_futures_carry_stress.py test/test_futures_bucket_strategy.py \
  test/test_futures_hedge_book.py test/test_futures_bucket_config.py \
  test/test_futures_bucket_results.py test/test_futures_carry_recorder.py \
  test/test_replay_futures_buckets.py
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

