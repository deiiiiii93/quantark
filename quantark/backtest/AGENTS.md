# Backtest Module - AI Agent Guide

## Purpose

This guide is specifically for AI agents working with the QuantArk backtest module. It provides targeted guidance on common tasks, patterns, and pitfalls to help you work effectively with the backtesting framework.

## Quick Start for AI Agents

### Understanding the Backtest Module Structure

```
backtest/
├── base.py                     # Protocol interfaces (shared)
├── transaction_costs.py        # Cost models (shared)
├── logger.py                   # Logging (shared)
├── visualizer.py               # Static plots (shared)
├── dashboard.py                # Interactive plots (shared)
├── report_generator.py         # Report generation (shared)
├── strategy/                   # Hedging strategies
│   ├── base_strategy.py        # Abstract base
│   ├── delta_neutral_strategy.py   # Equity: Delta hedging
│   ├── dv01_neutral_strategy.py    # FI: DV01 hedging
│   └── convexity_neutral_strategy.py
├── examples/                   # Backtest examples
│   ├── basic_delta_hedge.py
│   ├── advanced_backtest.py
│   └── fi_dv01_hedge.py
├── equity/                     # Equity implementation
│   ├── engine.py              # BacktestEngine
│   ├── config.py              # BacktestConfig
│   ├── state.py               # State tracking
│   ├── hedge_executor.py      # Spot/futures hedging
│   ├── results.py             # BacktestResults
│   └── metrics.py             # Performance metrics
├── fi/                         # Fixed Income implementation
│   ├── engine.py              # FIBacktestEngine
│   ├── config.py              # FIBacktestConfig
│   ├── state.py               # FI state tracking
│   ├── hedge_executor.py      # Bond futures hedging
│   ├── results.py             # FIBacktestResults
│   └── metrics.py             # FI metrics
└── README.md                   # User-facing documentation
```

### Common Imports

```python
# Core engine and config
from backtest.equity import BacktestEngine, BacktestConfig
from backtest.fi import FIBacktestEngine, FIBacktestConfig

# Strategies
from backtest.strategy import (
    BaseStrategy,
    DeltaNeutralStrategy,
    DV01NeutralStrategy,
    ConvexityNeutralStrategy
)

# Transaction costs
from backtest.transaction_costs import (
    TransactionCostModel,
    ZeroCostModel,
    FixedCostModel,
    ProportionalCostModel,
    SlippageModel,
    BidAskSpreadModel,
    CompleteCostModel
)

# Logging
from backtest import BacktestLogger

# Visualization and reporting
from backtest import StaticVisualizer, InteractiveDashboard, ReportGenerator

# Results and metrics
from backtest.equity.results import BacktestResults
from backtest.equity.metrics import BacktestMetrics
```

## Task-Oriented Guidance

### Task 1: Adding a New Hedging Strategy

**When**: When you need to implement a custom hedging strategy (e.g., gamma hedging, vega hedging, dynamic hedging).

**Steps**:

1. **Create strategy class** in `backtest/strategy/your_strategy.py`:

```python
from backtest.strategy import BaseStrategy, AssetClass, HedgingTarget
from typing import Dict, Any, Optional
from datetime import datetime

class YourHedgingStrategy(BaseStrategy):
    """
    Your custom hedging strategy.

    Follow the BaseStrategy interface.
    """

    def __init__(
        self,
        name: str,
        your_param: float,  # Custom parameter
        asset_class: AssetClass = AssetClass.EQUITY,
        hedging_target: HedgingTarget = HedgingTarget.DELTA,
        hedge_instrument: str = "spot",
    ):
        """Initialize strategy with custom parameters."""
        super().__init__(
            name=name,
            asset_class=asset_class,
            hedging_target=hedging_target,
            hedge_instrument=hedge_instrument
        )
        self.your_param = your_param

    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ) -> bool:
        """
        Determine if hedging should be performed.

        Args:
            current_time: Current timestamp
            portfolio_greeks: Portfolio risk measures (delta, gamma, vega, etc.)
            market_data: Current market data (spot, vol, rate)
            **kwargs: Additional context

        Returns:
            True if hedge should be executed, False otherwise
        """
        # Your logic here
        # Example: Hedge when gamma exceeds threshold
        gamma = portfolio_greeks.get('gamma', 0.0)
        return abs(gamma) > self.your_param

    def calculate_hedge_size(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ) -> float:
        """
        Calculate hedge size.

        Args:
            current_time: Current timestamp
            portfolio_greeks: Portfolio risk measures
            market_data: Current market data
            **kwargs: Additional context

        Returns:
            Hedge size (positive=buy, negative=sell)
        """
        # Your logic here
        # Example: Hedge 100% of gamma exposure
        gamma = portfolio_greeks.get('gamma', 0.0)
        target_gamma = 0.0
        return -(gamma - target_gamma)

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'name': self.name,
            'your_param': self.your_param,
            'asset_class': self.asset_class.value,
            'hedging_target': self.hedging_target.value,
            'hedge_instrument': self.hedge_instrument
        }
```

2. **Export strategy** in `backtest/strategy/__init__.py`:

```python
from backtest.strategy.your_strategy import YourHedgingStrategy

__all__ = [
    "BaseStrategy",
    "DeltaNeutralStrategy",
    "DV01NeutralStrategy",
    "ConvexityNeutralStrategy",
    "YourHedgingStrategy",  # Add here
]
```

3. **Add to main exports** in `backtest/__init__.py`:

```python
from backtest.strategy import YourHedgingStrategy

__all__ = [
    # ... existing exports
    "YourHedgingStrategy",
]
```

4. **Add tests** in `test/test_backtest.py`:

```python
def test_your_hedging_strategy():
    """Test your custom strategy."""
    strategy = YourHedgingStrategy(
        name="Test",
        your_param=100.0
    )

    # Test should_hedge
    assert strategy.should_hedge(
        datetime.now(),
        {'gamma': 150.0},  # Above threshold
        {}
    ) == True

    assert strategy.should_hedge(
        datetime.now(),
        {'gamma': 50.0},  # Below threshold
        {}
    ) == False

    # Test calculate_hedge_size
    hedge_size = strategy.calculate_hedge_size(
        datetime.now(),
        {'gamma': 200.0},
        {}
    )
    assert hedge_size == -200.0  # Full gamma hedge

    # Test get_parameters
    params = strategy.get_parameters()
    assert params['name'] == 'Test'
    assert params['your_param'] == 100.0
```

**Key Requirements**:
- ✅ Extend `BaseStrategy` (don't modify it)
- ✅ Implement `should_hedge()` and `calculate_hedge_size()`
- ✅ Return float from `calculate_hedge_size()` (positive=buy, negative=sell)
- ✅ Support all asset classes if applicable
- ✅ Add comprehensive tests
- ✅ Update all `__init__.py` exports

### Task 2: Adding a New Asset Class

**When**: When you need to support a new asset class (e.g., Crypto, Commodities, FX).

**Steps**:

1. **Create asset-specific directory**:
   ```bash
   mkdir -p backtest/crypto
   ```

2. **Create config** in `backtest/crypto/config.py`:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Any, Optional, Dict
from backtest.base import BaseBacktestConfig

@dataclass
class CryptoBacktestConfig(BaseBacktestConfig):
    """Configuration for crypto backtests."""

    strategy: Any  # Crypto strategy
    start_date: datetime
    end_date: datetime
    underlying: str
    initial_positions: List[Any]
    market_data_adapter: Any
    transaction_cost_model: Any

    frequency: str = "D"
    currency: str = "USD"
    logging_level: str = "INFO"
    results_path: Optional[str] = None
    save_snapshots: bool = True
    calculate_greeks: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""
        # Add validation logic
        pass

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "strategy": self.strategy.__class__.__name__,
            "underlying": self.underlying,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }
```

3. **Create state tracking** in `backtest/crypto/state.py`:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
from backtest.base import BaseTradeRecord

@dataclass
class CryptoTradeRecord(BaseTradeRecord):
    """Crypto trade record."""
    # Add crypto-specific fields
    funding_rate: Optional[float] = None
    exchange: Optional[str] = None

class CryptoStateTracker:
    """Track crypto backtest state."""

    def __init__(self):
        self.states_df = None  # Will be DataFrame

    def record_state(
        self,
        timestamp: datetime,
        portfolio_value: float,
        greeks: Dict[str, float],
        hedge_positions: Dict[str, float],
        transaction_costs: float,
        **kwargs
    ):
        """Record state at timestamp."""
        # Implementation
        pass
```

4. **Create hedge executor** in `backtest/crypto/hedge_executor.py`:

```python
from backtest.base import BaseHedgeExecutor
from backtest.transaction_costs import TransactionCostModel

class CryptoHedgeExecutor(BaseHedgeExecutor):
    """Execute crypto hedge trades."""

    def __init__(
        self,
        transaction_cost_model: TransactionCostModel,
        logger: Any,
    ):
        self.transaction_cost_model = transaction_cost_model
        self.logger = logger
        self.hedge_positions = {}

    def execute_hedge(...) -> CryptoTradeRecord:
        """Execute hedge trade."""
        # Implementation
        pass

    def get_hedge_position(...) -> Optional[Any]:
        """Get current hedge position."""
        # Implementation
        pass
```

5. **Create engine** in `backtest/crypto/engine.py`:

```python
from backtest.base import BaseBacktestEngine
from backtest.logger import BacktestLogger

class CryptoBacktestEngine(BaseBacktestEngine):
    """Crypto backtest engine."""

    def __init__(self, config: CryptoBacktestConfig):
        self.config = config
        self.logger = BacktestLogger(...)

    def run(self) -> 'CryptoBacktestResults':
        """Execute crypto backtest."""
        # Implementation following equity/FI pattern
        pass

    def _initialize(self) -> None:
        """Initialize crypto backtest."""
        # Setup portfolio, pricing env, hedge executor
        pass

    def _step(self, timestamp: datetime) -> None:
        """Execute single timestep."""
        # Update, hedge, record state
        pass

    def _finalize(self) -> 'CryptoBacktestResults':
        """Generate results."""
        # Create CryptoBacktestResults
        pass
```

6. **Create results** in `backtest/crypto/results.py`:

```python
from backtest.base import BaseBacktestResults

class CryptoBacktestResults(BaseBacktestResults):
    """Crypto backtest results."""

    def __init__(
        self,
        config: CryptoBacktestConfig,
        states_df: Any,
        trades_df: Any,
        logger: Any,
    ):
        self.config = config
        self.states_df = states_df
        self.trades_df = trades_df
        self.logger = logger
        self.metrics = CryptoBacktestMetrics(self)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary."""
        # Implementation
        pass

    def get_total_pnl(self) -> float:
        """Get total P&L."""
        # Implementation
        pass
```

7. **Create metrics** in `backtest/crypto/metrics.py`:

```python
class CryptoBacktestMetrics:
    """Crypto-specific performance metrics."""

    def __init__(self, results: CryptoBacktestResults):
        self.results = results

    def funding_cost_rate(self) -> float:
        """Calculate average funding cost rate."""
        # Crypto-specific metric
        pass
```

8. **Create __init__.py** in `backtest/crypto/__init__.py`:

```python
from backtest.crypto.config import CryptoBacktestConfig
from backtest.crypto.engine import CryptoBacktestEngine
from backtest.crypto.results import CryptoBacktestResults
from backtest.crypto.state import CryptoStateTracker, CryptoTradeRecord

__all__ = [
    "CryptoBacktestConfig",
    "CryptoBacktestEngine",
    "CryptoBacktestResults",
    "CryptoStateTracker",
    "CryptoTradeRecord",
]
```

9. **Add to main exports** in `backtest/__init__.py`:

```python
from backtest.crypto import (
    CryptoBacktestConfig,
    CryptoBacktestEngine,
    CryptoBacktestResults,
)

__all__ = [
    # ... existing exports
    "CryptoBacktestConfig",
    "CryptoBacktestEngine",
    "CryptoBacktestResults",
]
```

### Task 3: Adding New Transaction Cost Models

**When**: When you need to model specific transaction cost structures (e.g., tiered commissions, volume-based pricing, market impact models).

**Steps**:

1. **Create cost model class** in `backtest/transaction_costs.py`:

```python
class YourCostModel(TransactionCostModel):
    """
    Your custom transaction cost model.

    Implements TransactionCostModel interface.
    """

    def __init__(
        self,
        param1: float,
        param2: float,
    ):
        """Initialize with custom parameters."""
        self.param1 = param1
        self.param2 = param2

    def calculate_cost(
        self,
        quantity: float,
        price: float,
        notional: float,
        instrument_type: str,
        trade_type: str,
        **kwargs
    ) -> float:
        """
        Calculate transaction cost.

        Args:
            quantity: Trade quantity
            price: Execution price
            notional: Total notional
            instrument_type: 'spot', 'futures', 'option', etc.
            trade_type: 'open', 'close', 'hedge'
            **kwargs: Additional parameters

        Returns:
            Total cost (always positive)
        """
        # Your cost calculation logic
        # Example: Tiered commission
        if notional < 100000:  # Less than $100k
            return self.param1
        else:  # $100k or more
            return self.param2

    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'param1': self.param1,
            'param2': self.param2,
        }

    def __repr__(self) -> str:
        return f"YourCostModel(param1={self.param1}, param2={self.param2})"
```

2. **Export in __init__.py**:

```python
from backtest.transaction_costs import YourCostModel

__all__ = [
    # ... existing models
    "YourCostModel",
]
```

3. **Add tests**:

```python
def test_your_cost_model():
    """Test your cost model."""
    model = YourCostModel(param1=1.0, param2=2.0)

    # Test tiered pricing
    assert model.calculate_cost(
        quantity=100, price=100, notional=50000,
        instrument_type='spot', trade_type='hedge'
    ) == 1.0  # Small trade

    assert model.calculate_cost(
        quantity=1000, price=100, notional=100000,
        instrument_type='spot', trade_type='hedge'
    ) == 2.0  # Large trade
```

**Advanced Example: Volume-Based Market Impact**:

```python
class VolumeBasedCostModel(TransactionCostModel):
    """
    Market impact model based on volume.

    Uses square root model:
    cost = impact_coefficient * notional * sqrt(quantity / avg_volume)
    """

    def __init__(
        self,
        impact_coefficient: float = 0.0001,
        avg_daily_volume: float = 1000000,
    ):
        self.impact_coefficient = impact_coefficient
        self.avg_daily_volume = avg_daily_volume

    def calculate_cost(...) -> float:
        """Calculate cost with market impact."""
        # Base proportional cost
        base_cost = 0.0005 * notional  # 5 bps

        # Market impact (square root model)
        volume_ratio = abs(quantity) / self.avg_daily_volume
        impact = self.impact_coefficient * notional * np.sqrt(volume_ratio)

        return base_cost + impact
```

### Task 4: Adding New Visualization Types

**When**: When you need to create custom plots or dashboards for specific metrics.

**Steps**:

1. **Add plot method** in `backtest/visualizer.py`:

```python
def plot_your_metric(
    self,
    figsize: Tuple[int, int] = (14, 6),
    save: bool = False,
    filename: str = "your_metric.png",
) -> plt.Figure:
    """
    Plot your custom metric.

    Args:
        figsize: Figure size
        save: Whether to save plot
        filename: Filename if saving

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get your metric time series
    metric_series = self.results.states_df['your_metric']

    # Create plot
    ax.plot(metric_series.index, metric_series.values, linewidth=2)

    # Customize
    ax.set_xlabel("Date")
    ax.set_ylabel("Your Metric")
    ax.set_title("Your Metric Over Time")
    ax.grid(True, alpha=0.3)

    if save:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')

    return fig
```

2. **Add to comprehensive dashboard**:

```python
def create_comprehensive_dashboard(self, save: bool = True) -> plt.Figure:
    """Create 3x2 dashboard with all plots."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))

    # Row 1
    self.plot_pnl_over_time(ax=axes[0, 0], save=False)
    self.plot_your_metric(ax=axes[0, 1], save=False)

    # Row 2
    self.plot_delta_tracking(ax=axes[1, 0], save=False)
    self.plot_drawdown(ax=axes[1, 1], save=False)

    # Row 3
    self.plot_returns_distribution(ax=axes[2, 0], save=False)
    self.plot_hedge_frequency(ax=axes[2, 1], save=False)

    fig.tight_layout()

    if save:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.save_dir / "comprehensive_dashboard.png",
                   dpi=300, bbox_inches='tight')

    return fig
```

### Task 5: Adding New Performance Metrics

**When**: When you need to calculate custom performance or risk metrics.

**Steps**:

1. **Add metric method** in `backtest/equity/metrics.py`:

```python
def your_custom_metric(self) -> float:
    """
    Calculate your custom metric.

    Returns:
        Metric value
    """
    # Your calculation logic
    # Example: Calculate something from P&L
    pnl_series = self.results.get_pnl_series()
    returns = pnl_series.pct_change().dropna()

    # Calculate your metric
    metric_value = returns.mean() / returns.std() * np.sqrt(252)

    return metric_value

def calculate_all_metrics(self) -> Dict[str, float]:
    """Calculate all metrics including custom."""
    metrics = {
        'total_pnl': self.total_pnl(),
        'sharpe_ratio': self.sharpe_ratio(),
        # ... existing metrics
        'your_custom_metric': self.your_custom_metric(),  # Add here
    }
    return metrics
```

2. **Export metric** in `backtest/equity/__init__.py`:

```python
from backtest.equity.metrics import BacktestMetrics

__all__ = ["BacktestMetrics"]
```

### Task 6: Fixing Bugs in Backtest Execution

**When**: When you encounter incorrect hedging, state tracking, or results issues.

**Debugging Steps**:

1. **Check configuration**:

```python
# Verify config is valid
try:
    config.validate()
except ValidationError as e:
    print(f"Config error: {e}")

# Check strategy parameters
print(f"Strategy: {config.strategy.name}")
print(f"Delta threshold: {config.strategy.delta_threshold}")
```

2. **Check market data**:

```python
# Verify data availability
market_data = config.market_data_adapter.get_market_data_set(
    asset_name=config.underlying,
    start_date=config.start_date,
    end_date=config.end_date
)
print(f"Data points: {len(market_data.spot_data)}")
```

3. **Check state tracking**:

```python
# Verify states DataFrame
states_df = results.states_df
print(f"States shape: {states_df.shape}")
print(f"States columns: {states_df.columns.tolist()}")

# Check for NaN values
if states_df.isnull().any().any():
    print("WARNING: NaN values in states")
```

4. **Check hedge execution**:

```python
# Verify trades
trades_df = results.get_hedge_trades()
print(f"Number of trades: {len(trades_df)}")

if len(trades_df) > 0:
    print(f"First trade:\n{trades_df.iloc[0]}")
    print(f"Last trade:\n{trades_df.iloc[-1]}")
```

5. **Validate results**:

```python
# Basic validation
assert results.get_total_pnl() > -1e6, "Unreasonable P&L"
assert results.get_total_pnl() < 1e8, "Unreasonable P&L"

# Check delta tracking
if 'delta' in results.states_df.columns:
    max_abs_delta = results.states_df['delta'].abs().max()
    print(f"Max absolute delta: {max_abs_delta}")
```

**Common Bugs and Solutions**:

1. **Bug**: No hedges executed
   - **Cause**: Delta never exceeds threshold
   - **Fix**: Lower `delta_threshold` or check `should_hedge()` logic

2. **Bug**: Excessive transaction costs
   - **Cause**: Threshold too low (too many trades)
   - **Fix**: Increase threshold or add `min_time_between_hedges`

3. **Bug**: Delta doesn't converge to target
   - **Cause**: Hedge ratio too low
   - **Fix**: Increase `hedge_ratio` to 1.0

4. **Bug**: State tracking has gaps
   - **Cause**: Missing data or failed Greeks calculation
   - **Fix**: Check `calculate_greeks` and market data quality

### Task 7: Performance Optimization

**When**: When backtests are too slow for production or testing.

**Optimization Strategies**:

1. **Reduce Greeks Calculation Frequency**:

```python
# Only calculate Greeks every N steps
class OptimizedStrategy(BaseStrategy):
    def __init__(self, name: str, greeks_frequency: int = 1):
        super().__init__(name, ...)
        self.greeks_frequency = greeks_frequency
        self._step_count = 0

    def should_hedge(self, current_time, portfolio_greeks, market_data, **kwargs):
        self._step_count += 1
        if self._step_count % self.greeks_frequency != 0:
            # Skip this step
            return False
        # ... normal logic
```

2. **Disable Unnecessary Features**:

```python
config = BacktestConfig(
    # ... other parameters
    save_snapshots=False,        # Disable state snapshots
    calculate_greeks=False,      # Skip Greeks (if not needed)
    logging_level='WARNING',     # Reduce logging overhead
)
```

3. **Use Mock Data for Development**:

```python
# Fast synthetic data
adapter = MockMarketDataAdapter(seed=42)

# Don't use real data during development
# adapter = RealMarketDataAdapter(...)  # Slower
```

4. **Optimize Strategy Check Frequency**:

```python
strategy = DeltaNeutralStrategy(
    name="Optimized",
    rebalance_frequency='daily',  # vs 'continuous'
    min_time_between_hedges=timedelta(hours=1)  # Rate limiting
)
```

5. **Vectorized Operations** (for custom code):

```python
# Bad: Python loop
pnl_list = []
for i in range(len(timestamps)):
    pnl = calculate_pnl(i)
    pnl_list.append(pnl)

# Good: Vectorized
pnl_array = calculate_pnl_vectorized(timestamps)  # 10-100x faster
```

### Task 8: Adding Custom Market Data Adapters

**When**: When you need to integrate with specific data sources (databases, APIs, files).

**Steps**:

1. **Create adapter class**:

```python
from util.marketdata.adapter import BaseMarketDataAdapter

class YourDataAdapter(BaseMarketDataAdapter):
    """
    Your custom market data adapter.

    Implements BaseMarketDataAdapter interface.
    """

    def __init__(self, connection_params: Dict[str, Any]):
        """Initialize with connection parameters."""
        self.connection_params = connection_params
        self.connection = self._connect()

    def _connect(self) -> Any:
        """Establish connection to data source."""
        # Your connection logic
        pass

    def get_market_data_set(
        self,
        asset_name: str,
        start_date: datetime,
        end_date: datetime,
        currency: str = "USD",
        frequency: str = "D",
    ) -> MarketDataSet:
        """Fetch market data."""
        # Fetch data from your source
        spot_data = self._fetch_spot_data(asset_name, start_date, end_date)
        vol_data = self._fetch_vol_data(asset_name, start_date, end_date)
        rate_data = self._fetch_rate_data(currency, start_date, end_date)

        # Return MarketDataSet
        return MarketDataSet(
            spot_data=spot_data,
            vol_data=vol_data,
            rate_data=rate_data,
        )
```

2. **Use in backtest**:

```python
# Configure with your adapter
adapter = YourDataAdapter(
    connection_params={
        'host': 'localhost',
        'port': 5432,
        'database': 'market_data',
    }
)

config = BacktestConfig(
    strategy=strategy,
    # ... other parameters
    market_data_adapter=adapter,
)
```

## Common Patterns and Anti-Patterns

### ✅ DO: Follow These Patterns

1. **Extend BaseStrategy**:

```python
# Good: Extend BaseStrategy
class YourStrategy(BaseStrategy):
    def should_hedge(self, current_time, greeks, data, **kwargs):
        # Implementation
        pass

    def calculate_hedge_size(self, current_time, greeks, data, **kwargs):
        # Implementation
        pass
```

2. **Use Protocol Interfaces**:

```python
# Good: Implement BaseHedgeExecutor protocol
class YourExecutor(BaseHedgeExecutor):
    def execute_hedge(self, underlying, size, context, time, reason):
        # Implementation
        pass
```

3. **Validate Configuration**:

```python
# Good: Validate in _validate()
def _validate(self):
    if self.delta_threshold < 0:
        raise ValidationError("Threshold must be non-negative")
```

4. **Use Type Hints**:

```python
# Good: Full type hints
from typing import Dict, Any, Optional
from datetime import datetime

def should_hedge(
    self,
    current_time: datetime,
    portfolio_greeks: Dict[str, float],
    market_data: Dict[str, float],
    **kwargs: Any
) -> bool:
    pass
```

5. **Document Strategy Logic**:

```python
def should_hedge(self, current_time, greeks, data, **kwargs):
    """
    Hedge when delta exceeds threshold.

    Uses simple threshold-based approach:
    - Hedge when |delta| > threshold
    - No time-based or volatility filters
    """
    delta = greeks.get('delta', 0.0)
    return abs(delta) > self.delta_threshold
```

### ❌ DON'T: Avoid These Anti-Patterns

1. **Don't Modify BaseStrategy**:

```python
# Bad: Don't do this!
class BaseStrategy:
    def should_hedge(self, current_time, greeks, data, **kwargs):
        # This will break all existing strategies

# Good: Extend instead
class YourStrategy(BaseStrategy):
    def should_hedge(self, current_time, greeks, data, **kwargs):
        # Your implementation
```

2. **Don't Assume Asset Class**:

```python
# Bad: Assume equity
def calculate_hedge_size(self, current_time, greeks, data, **kwargs):
    assert 'delta' in greeks, "Must have delta"

# Good: Check asset class
if self.asset_class == AssetClass.EQUITY:
    delta = greeks.get('delta', 0.0)
elif self.asset_class == AssetClass.FIXED_INCOME:
    dv01 = greeks.get('dv01', 0.0)
```

3. **Don't Ignore Transaction Costs**:

```python
# Bad: Assume zero costs
def execute_hedge(self, ...):
    # No cost calculation
    return trade

# Good: Calculate costs
cost = self.transaction_cost_model.calculate_cost(...)
trade.transaction_cost = cost
```

4. **Don't Use Loops for Vectorizable Operations**:

```python
# Bad: Python loop
values = []
for timestamp in timestamps:
    value = self.portfolio.value(timestamp)
    values.append(value)

# Good: Vectorized
values = self.portfolio.value_vectorized(timestamps)
```

5. **Don't Skip Result Validation**:

```python
# Bad: No validation
results = engine.run()
return results

# Good: Validate results
results = engine.run()
assert results.get_total_pnl() > -1e6
return results
```

## Quick Reference

### Engine Selection Guide

| Asset Class | Engine | Strategy | Hedge Instrument |
|-------------|--------|----------|------------------|
| Equity | BacktestEngine | DeltaNeutralStrategy | spot or futures |
| Fixed Income | FIBacktestEngine | DV01NeutralStrategy | bond_futures |
| Fixed Income | FIBacktestEngine | ConvexityNeutralStrategy | bond_futures |

### Strategy Selection

| Hedging Target | Strategy | Asset Class | Best For |
|---------------|----------|-------------|----------|
| Delta | DeltaNeutralStrategy | Equity | Options, equity derivatives |
| DV01 | DV01NeutralStrategy | Fixed Income | Bond portfolios |
| Convexity | ConvexityNeutralStrategy | Fixed Income | Long duration portfolios |

### Transaction Cost Model Selection

| Model | Use Case | Example |
|-------|----------|---------|
| ZeroCostModel | Theoretical analysis | No trading costs |
| FixedCostModel | Fixed commission brokers | $2 per trade |
| ProportionalCostModel | Percentage-based fees | 5 bps of notional |
| CompleteCostModel | Realistic trading | Full cost modeling |

### Configuration Best Practices

| Parameter | Recommended Value | Reason |
|-----------|-------------------|--------|
| frequency | 'D' (daily) | Balance of detail vs speed |
| logging_level | 'INFO' (prod), 'DEBUG' (dev) | Control log verbosity |
| save_snapshots | True (dev), False (prod large) | Memory vs debugging |
| calculate_greeks | True (default) | Required for most strategies |

## Testing Strategy

### Test Categories

1. **Unit Tests** (test individual components)
   - Configuration validation
   - Strategy logic (should_hedge, calculate_hedge_size)
   - Transaction cost calculations
   - Hedge execution

2. **Integration Tests** (test workflows)
   - End-to-end backtest execution
   - State tracking accuracy
   - Results generation

3. **Performance Tests** (test scalability)
   - Large portfolio performance
   - Memory usage
   - Calculation time

### Example Test Structure

```python
class TestDeltaNeutralStrategy:
    """Test suite for DeltaNeutralStrategy."""

    def test_should_hedge_above_threshold(self):
        """Test hedge decision when delta exceeds threshold."""
        strategy = DeltaNeutralStrategy(
            name="Test",
            delta_threshold=100.0
        )

        assert strategy.should_hedge(
            datetime.now(),
            {'delta': 150.0},  # Above threshold
            {}
        ) == True

    def test_should_hedge_below_threshold(self):
        """Test hedge decision when delta within threshold."""
        strategy = DeltaNeutralStrategy(
            name="Test",
            delta_threshold=100.0
        )

        assert strategy.should_hedge(
            datetime.now(),
            {'delta': 50.0},  # Below threshold
            {}
        ) == False

    def test_calculate_hedge_size(self):
        """Test hedge size calculation."""
        strategy = DeltaNeutralStrategy(
            name="Test",
            target_delta=0.0,
            hedge_ratio=1.0
        )

        hedge_size = strategy.calculate_hedge_size(
            datetime.now(),
            {'delta': 200.0},
            {}
        )

        # Should hedge 100% of delta to reach target of 0
        assert hedge_size == -200.0
```

## Integration with Other Modules

### Portfolio Module

```python
from portfolio import Portfolio, Position
from asset.equity.product.option import EuropeanVanillaOption

# Create option position
option = EuropeanVanillaOption(
    strike=100.0,
    option_type=OptionType.CALL,
    maturity=1.0
)

position = Position(
    product=option,
    quantity=100,
    entry_price=10.0,
    underlying="AAPL"
)

# Use in backtest
config = BacktestConfig(
    strategy=strategy,
    initial_positions=[position],
    # ... other parameters
)
```

### PriceEnv Module

```python
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve

# Engine creates PricingEnvironment internally
# Access for custom logic:
pricing_env = engine.pricing_env
spot_price = pricing_env.get_spot("AAPL")
```

### Market Data Module

```python
from util.marketdata.adapter import MockMarketDataAdapter

# Mock data (fast)
adapter = MockMarketDataAdapter(seed=42)

# Configure mock data
adapter.set_asset_config("AAPL", {
    'initial_spot': 150.0,
    'initial_vol': 0.25,
    'drift': 0.10,
    'vol_of_vol': 0.3
})

# Use in backtest
config.market_data_adapter = adapter
```

## Working with Results

### Basic Results Access

```python
results = engine.run()

# Summary
summary = results.get_summary()
print(f"Total P&L: ${summary['total_pnl']:,.2f}")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.3f}")

# Time series
pnl_series = results.get_pnl_series()
delta_series = results.states_df['delta']

# Trade history
trades_df = results.get_hedge_trades()
print(f"Number of trades: {len(trades_df)}")
```

### Advanced Analysis

```python
# Calculate custom metric
returns = results.get_pnl_series().pct_change().dropna()
volatility = returns.std() * np.sqrt(252)
skewness = returns.skew()
kurtosis = returns.kurtosis()

print(f"Annualized Volatility: {volatility:.2%}")
print(f"Skewness: {skewness:.3f}")
print(f"Kurtosis: {kurtosis:.3f}")

# Analyze trades
if len(trades_df) > 0:
    avg_trade_size = trades_df['notional'].mean()
    total_costs = trades_df['transaction_cost'].sum()
    print(f"Average trade size: ${avg_trade_size:,.2f}")
    print(f"Total transaction costs: ${total_costs:,.2f}")
```

### Visualization

```python
# Static plots
visualizer = StaticVisualizer(results, save_dir="plots")
visualizer.plot_pnl_over_time(save=True)
visualizer.plot_delta_tracking(save=True)

# Interactive plots
dashboard = InteractiveDashboard(results, save_dir="plots/interactive")
dashboard.plot_pnl_interactive(save=True)

# Comprehensive dashboard
visualizer.create_summary_dashboard(save=True)
dashboard.create_comprehensive_dashboard(save=True)
```

## Debugging Tips

### Enable Verbose Logging

```python
config = BacktestConfig(
    logging_level='DEBUG',
    results_path='logs/debug_run'
)

# This will log:
# - Each backtest step
# - Hedge decisions
# - Trade execution
# - State updates
```

### Check Strategy Decisions

```python
# Inspect states DataFrame
states_df = results.states_df

# See when hedges occurred
hedge_events = states_df[states_df['transaction_costs'] > 0]
print(f"Hedges occurred at:\n{hedge_events.index.tolist()}")

# Check delta evolution
delta_series = states_df['delta']
print(f"Delta range: {delta_series.min():.2f} to {delta_series.max():.2f}")
```

### Validate Greeks Calculation

```python
# Check Greeks time series
for greek in ['delta', 'gamma', 'vega']:
    if greek in states_df.columns:
        series = states_df[greek]
        print(f"{greek.capitalize()} stats:")
        print(f"  Mean: {series.mean():.4f}")
        print(f"  Std: {series.std():.4f}")
        print(f"  Min: {series.min():.4f}")
        print(f"  Max: {series.max():.4f}")
```

### Analyze Transaction Costs

```python
trades_df = results.get_hedge_trades()

if len(trades_df) > 0:
    # Cost breakdown
    print("Transaction cost breakdown:")
    print(f"  Fixed commission: ${trades_df['fixed_commission'].sum():.2f}")
    print(f"  Proportional: ${trades_df['proportional_commission'].sum():.2f}")
    print(f"  Slippage: ${trades_df['slippage'].sum():.2f}")
    print(f"  Spread: ${trades_df['spread_cost'].sum():.2f}")

    # Cost per trade
    avg_cost = trades_df['transaction_cost'].mean()
    print(f"Average cost per trade: ${avg_cost:.2f}")
```

## Common Issues and Solutions

### Issue: "No trades executed"

**Error**: Strategy never triggers hedges.

**Solution**:
```python
# Check threshold
print(f"Delta threshold: {config.strategy.delta_threshold}")
print(f"Max delta: {results.states_df['delta'].abs().max()}")

# Lower threshold if needed
strategy = DeltaNeutralStrategy(
    name="Test",
    delta_threshold=50.0  # Lower from 100.0
)
```

### Issue: "Excessive transaction costs"

**Error**: Too many trades with high cumulative costs.

**Solution**:
```python
# Increase threshold
strategy = DeltaNeutralStrategy(
    delta_threshold=200.0  # Increase from 100.0
)

# Add minimum time between hedges
strategy = DeltaNeutralStrategy(
    min_time_between_hedges=timedelta(hours=4)  # Rate limiting
)
```

### Issue: "Delta doesn't converge to target"

**Error**: Delta oscillates or doesn't reach target.

**Solution**:
```python
# Check hedge ratio
strategy = DeltaNeutralStrategy(
    hedge_ratio=1.0  # Full hedge (not 0.5 or 0.8)
)

# Check target delta
strategy = DeltaNeutralStrategy(
    target_delta=0.0  # Should be 0 for neutral
)
```

### Issue: "State tracking has gaps"

**Error**: Missing timestamps or NaN values.

**Solution**:
```python
# Check data frequency
print(f"Expected frequency: {config.frequency}")
print(f"Actual data points: {len(results.states_df)}")

# Ensure market data covers full period
adapter = config.market_data_adapter
data = adapter.get_market_data_set(...)
if len(data.spot_data) < 100:
    print("WARNING: Insufficient market data")
```

### Issue: "Slow execution"

**Error**: Backtest takes too long.

**Solutions**:
1. **Reduce frequency**: Use 'D' instead of 'H'
2. **Disable snapshots**: `save_snapshots=False`
3. **Skip Greeks**: `calculate_greeks=False`
4. **Use mock data**: `MockMarketDataAdapter`

## Performance Monitoring

### Benchmark Function

```python
import time

def benchmark_backtest(config):
    """Benchmark backtest execution time."""
    engine = BacktestEngine(config)

    start = time.time()
    results = engine.run()
    elapsed = time.time() - start

    print(f"Backtest completed in {elapsed:.2f}s")
    print(f"  States: {len(results.states_df)}")
    print(f"  Trades: {len(results.get_hedge_trades())}")
    print(f"  Time per step: {elapsed / len(results.states_df) * 1000:.1f}ms")

    return results
```

### Performance Regression Testing

```python
@pytest.mark.performance
def test_large_backtest_performance():
    """Ensure backtest completes within time limit."""
    # Create large portfolio
    positions = create_large_portfolio(100)

    config = BacktestConfig(
        strategy=strategy,
        start_date=datetime(2024, 1, 1),
        end_date=2024,
        initial_positions=positions,
        save_snapshots=False  # Disable for speed
    )

    start = time.time()
    results = BacktestEngine(config).run()
    elapsed = time.time() - start

    # Must complete within 30 seconds
    assert elapsed < 30.0, f"Backtest took {elapsed:.2f}s (>30s limit)"
```

## Documentation Guidelines

### For New Features

When adding new functionality, document:

1. **Purpose**: What problem does this solve?
2. **Usage**: How to use it (with examples)?
3. **Configuration**: What parameters control it?
4. **Performance**: What's the performance impact?
5. **Testing**: How is correctness verified?

### For Strategy Changes

Document:
1. **Logic**: Why this decision logic?
2. **Parameters**: What do they control?
3. **Examples**: When to use this strategy?
4. **Caveats**: What are the limitations?

### Code Comments

Add comments for:

1. **Complex algorithms** (why, not just what)
2. **Non-obvious optimizations**
3. **Asset-specific logic** (equity vs FI)
4. **Integration points** (how components interact)

```python
# Good: Explains why
# Hedge 90% of delta to avoid over-hedging in volatile markets
# Full hedging can lead to excessive transaction costs
hedge_size = -0.9 * (current_delta - target_delta)

# Good: Explains asset-specific logic
# For equity: use delta
# For FI: use DV01
if self.asset_class == AssetClass.EQUITY:
    risk_measure = greeks.get('delta', 0.0)
else:  # FIXED_INCOME
    risk_measure = greeks.get('dv01', 0.0)
```

## Checklist for AI Agents

Before submitting code changes:

- [ ] New strategies extend `BaseStrategy` (don't modify it)
- [ ] All abstract methods implemented
- [ ] Type hints added to all functions
- [ ] Configuration validated in `_validate()`
- [ ] Transaction costs properly calculated
- [ ] State tracking works correctly
- [ ] Results include all expected data
- [ ] Unit tests added for new functionality
- [ ] Integration tests verify end-to-end workflows
- [ ] Performance impact assessed and documented
- [ ] Documentation updated (CLAUDE.md, AGENTS.md)
- [ ] Examples added to README.md if user-facing
- [ ] Backwards compatibility maintained
- [ ] Error handling comprehensive
- [ ] Code follows project style guidelines

## Summary

This guide provides AI agents with targeted guidance for working with the backtest module. Key takeaways:

1. **Always extend BaseStrategy** - Don't modify the base class
2. **Follow protocol interfaces** - Ensures consistency across asset classes
3. **Validate configuration** - Prevents silent errors
4. **Model transaction costs realistically** - Critical for practical results
5. **Use appropriate strategies** - Match strategy to asset class
6. **Test thoroughly** - Backtesting is critical for risk management
7. **Document complex logic** - Future maintainers will thank you

For detailed implementation guidance, see `backtest/CLAUDE.md`.
For user-facing documentation, see `backtest/README.md`.
