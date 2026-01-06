# Backtest Module - Developer Guide

## Overview

The backtest module is a comprehensive framework for simulating and evaluating hedging strategies across multiple asset classes. It provides transaction cost modeling, logging, visualizations, and performance analytics.

## Architecture

### Core Design Pattern: Protocol-Based Multi-Asset Implementation

```
backtest/
├── base.py                    # Base protocols (shared interface)
├── transaction_costs.py       # Cost models (shared)
├── logger.py                  # Logging infrastructure
├── visualizer.py              # Static matplotlib visualizations
├── dashboard.py               # Interactive plotly visualizations
├── report_generator.py        # HTML/text report generation
├── strategy/                  # Strategy implementations
│   ├── base_strategy.py       # Abstract base (shared)
│   ├── delta_neutral_strategy.py   # Equity: delta hedging
│   ├── dv01_neutral_strategy.py    # FI: DV01 hedging
│   └── convexity_neutral_strategy.py # FI: convexity hedging
├── equity/                    # Equity-specific implementation
│   ├── engine.py             # BacktestEngine
│   ├── config.py             # BacktestConfig
│   ├── state.py              # StateTracker, TradeRecord
│   ├── hedge_executor.py     # Spot/futures hedging
│   ├── results.py            # BacktestResults
│   └── metrics.py            # PerformanceMetrics
└── fi/                        # Fixed Income (lazy-loaded)
    ├── engine.py             # FIBacktestEngine
    ├── config.py             # FIBacktestConfig
    ├── state.py              # FIStateTracker
    ├── hedge_executor.py     # Bond futures hedging
    ├── results.py            # FIBacktestResults
    └── metrics.py            # FIPerformanceMetrics
```

### Exports

```python
# Base protocols
from backtest import BaseTradeRecord, BaseHedgeExecutor, BaseBacktestEngine

# Equity (backward-compatible aliases)
from backtest import BacktestEngine, BacktestConfig, BacktestResults
from backtest import HedgeExecutor, StateTracker, PerformanceMetrics

# Explicit equity names
from backtest import EquityBacktestEngine, EquityBacktestConfig

# FI (lazy-loaded via __getattr__)
from backtest import FIBacktestEngine, FIBacktestConfig, FIBacktestResults

# Transaction costs
from backtest import ZeroCostModel, FixedCostModel, ProportionalCostModel, CompleteCostModel

# Strategies
from backtest.strategy import DeltaNeutralStrategy, DV01NeutralStrategy, ConvexityNeutralStrategy

# Visualization
from backtest import StaticVisualizer, InteractiveDashboard, ReportGenerator, BacktestLogger
```

## Components

### 1. Strategies (`strategy/`)

**BaseStrategy** - Abstract base class for all hedging strategies:
```python
class BaseStrategy(ABC):
    def __init__(
        self,
        name: str,
        asset_class: AssetClass,          # EQUITY, FIXED_INCOME, GENERIC
        hedging_target: HedgingTarget,    # DELTA, DV01, GAMMA, VEGA, CONVEXITY
        hedge_instrument: str,            # spot, futures, bond_futures
    ): ...

    @abstractmethod
    def should_hedge(self, current_time, portfolio_greeks, market_data, **kwargs) -> bool: ...

    @abstractmethod
    def calculate_hedge_size(self, current_time, portfolio_greeks, market_data, **kwargs) -> float: ...
```

**Concrete Strategies:**

| Strategy | Asset Class | Target | Key Parameters |
|----------|-------------|--------|----------------|
| `DeltaNeutralStrategy` | Equity | DELTA | `delta_threshold`, `rebalance_frequency`, `hedge_instrument` |
| `DV01NeutralStrategy` | FI | DV01 | `dv01_threshold`, `futures_dv01` |
| `ConvexityNeutralStrategy` | FI | CONVEXITY | `convexity_threshold` |

### 2. Configuration

**BacktestConfig** (equity):
```python
@dataclass
class BacktestConfig:
    # Required
    strategy: BaseStrategy
    start_date: datetime
    end_date: datetime
    underlying: str
    initial_positions: List[Position]
    market_data_adapter: BaseMarketDataAdapter
    transaction_cost_model: TransactionCostModel

    # Optional
    frequency: str = "D"           # D, H, W, M
    calculate_greeks: bool = True
    greeks_method: str = "analytical"
    logging_level: str = "INFO"
    results_path: Optional[str] = None
    save_snapshots: bool = True
```

**FIBacktestConfig** - Similar structure for fixed income portfolios.

### 3. Transaction Cost Models

```
TransactionCostModel (Abstract Base)
├── ZeroCostModel           # No costs (testing)
├── FixedCostModel          # Flat fee per trade
├── ProportionalCostModel   # Percentage of notional
└── CompleteCostModel       # Combines: fixed + proportional + slippage + spread
```

**CompleteCostModel** parameters:
- `fixed_commission`: Flat fee per trade (e.g., $2)
- `proportional_rate`: Percentage of notional (e.g., 5 bps)
- `slippage_coefficient`: Market impact (linear or sqrt)
- `spread_bps`: Bid-ask spread

### 4. Engine Lifecycle

```python
class BacktestEngine:
    def run(self) -> BacktestResults:
        self._initialize()           # Setup portfolio, pricing env, hedge executor
        for timestamp in timestamps:
            self._step(timestamp)    # Update env, calc greeks, check/execute hedge
        return self._finalize()      # Generate results
```

**Key Integration Points:**
- Portfolio: `portfolio.Portfolio` (equity) / `portfolio.fi.FIPortfolio`
- Pricing: `priceenv.PricingEnvironment`
- Greeks: `asset.equity.riskmeasures.GreeksCalculator`
- Market Data: Adapter pattern (Mock, Database, etc.)

### 5. Results & Metrics

**BacktestResults:**
```python
results = engine.run()
results.get_summary()         # Dict with all summary stats
results.get_total_pnl()       # Total P&L
results.get_pnl_series()      # pd.Series of cumulative P&L
results.get_hedge_trades()    # pd.DataFrame of trades
results.states_df             # Full state history
results.metrics               # PerformanceMetrics instance
```

**PerformanceMetrics:**
- P&L: `sharpe_ratio()`, `sortino_ratio()`, `max_drawdown()`, `volatility()`
- Hedging: `hedge_frequency()`, `delta_tracking_error()`, `average_hedge_cost()`
- Risk: `value_at_risk(confidence)`, `conditional_var(confidence)`

**FI-specific metrics:** `dv01_tracking_error()`, `max_dv01_exposure()`, `dv01_hedge_effectiveness()`

### 6. Visualization

**StaticVisualizer** (matplotlib):
```python
viz = StaticVisualizer(results, save_dir="plots")
viz.plot_pnl_over_time(save=True)
viz.plot_delta_tracking(save=True)
viz.create_summary_dashboard(save=True)
```

**InteractiveDashboard** (plotly):
```python
dash = InteractiveDashboard(results)
dash.plot_pnl_interactive(save=True)
dash.create_comprehensive_dashboard(save=True)
```

**ReportGenerator**: Generate HTML reports with embedded visualizations.

## Usage Examples

### Basic Delta-Neutral Backtest

```python
from backtest import BacktestEngine, BacktestConfig, ZeroCostModel
from backtest.strategy import DeltaNeutralStrategy

strategy = DeltaNeutralStrategy(
    name="BasicDN",
    delta_threshold=100.0,
    rebalance_frequency='daily',
    hedge_instrument='spot'
)

config = BacktestConfig(
    strategy=strategy,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30),
    underlying="AAPL",
    initial_positions=[option_position],
    market_data_adapter=MockMarketDataAdapter(seed=42),
    transaction_cost_model=ZeroCostModel()
)

engine = BacktestEngine(config)
results = engine.run()

print(f"Total P&L: ${results.get_total_pnl():,.2f}")
print(f"Sharpe: {results.metrics.sharpe_ratio():.3f}")
```

### FI DV01-Neutral Backtest

```python
from backtest import FIBacktestEngine, FIBacktestConfig
from backtest.strategy import DV01NeutralStrategy

strategy = DV01NeutralStrategy(
    name="DV01_Neutral",
    dv01_threshold=50000.0,
    futures_dv01=1000.0
)

config = FIBacktestConfig(
    strategy=strategy,
    start_date=start_date,
    end_date=end_date,
    underlying="UST_10Y",
    initial_positions=[bond_position],
    market_data_adapter=fi_data_adapter,
    transaction_cost_model=CompleteCostModel(fixed_commission=2.0, proportional_rate=0.0005)
)

engine = FIBacktestEngine(config)
results = engine.run()
```

### Custom Strategy

```python
from backtest.strategy import BaseStrategy, AssetClass, HedgingTarget

class MyCustomStrategy(BaseStrategy):
    def __init__(self, name: str, vol_threshold: float):
        super().__init__(
            name=name,
            asset_class=AssetClass.EQUITY,
            hedging_target=HedgingTarget.DELTA,
            hedge_instrument='spot'
        )
        self.vol_threshold = vol_threshold

    def should_hedge(self, current_time, portfolio_greeks, market_data, **kwargs) -> bool:
        delta = portfolio_greeks.get('delta', 0.0)
        vol = market_data.get('volatility', 0.0)
        return abs(delta) > 100.0 and vol > self.vol_threshold

    def calculate_hedge_size(self, current_time, portfolio_greeks, market_data, **kwargs) -> float:
        return -0.9 * portfolio_greeks.get('delta', 0.0)
```

## Testing

```bash
# All backtest tests
python -m pytest test/test_backtest.py -v

# Specific category
python -m pytest test/test_backtest.py::TestDeltaNeutralStrategy -v

# With coverage
python -m pytest test/test_backtest.py --cov=backtest
```

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Start date must be before end date" | Invalid date order | Check date parameters |
| "Invalid frequency" | Wrong frequency string | Use 'D', 'H', 'M', or 'W' |
| "Insufficient market data" | Data doesn't cover range | Check adapter date coverage |
| Strategy/engine mismatch | FI strategy with equity engine | Use matching engine type |

## Summary

- **Asset Classes**: Equity (delta/gamma), Fixed Income (DV01/convexity)
- **Strategies**: 3 built-in + custom via BaseStrategy
- **Cost Models**: 4 models from zero to complete
- **Outputs**: Results + metrics + visualizations + reports
- **FI Note**: FI components lazy-loaded via `__getattr__` to avoid circular imports
