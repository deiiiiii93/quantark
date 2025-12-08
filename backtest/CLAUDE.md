# Backtest Module Developer Guide

## Overview

The QuantArk backtest module is a comprehensive, production-grade framework for simulating and evaluating hedging strategies across multiple asset classes. It provides sophisticated transaction cost modeling, extensive logging, rich visualizations, and detailed performance analytics.

## Architecture

### Core Design Pattern: Modular Asset-Specific Implementation

The module follows a **dual-track architecture** with shared infrastructure and asset-specific implementations:

```
backtest/
├── base.py                    # Base protocols (shared interface)
├── transaction_costs.py       # Cost models (shared)
├── logger.py                  # Logging infrastructure (shared)
├── visualizer.py              # Static visualizations (shared)
├── dashboard.py               # Interactive visualizations (shared)
├── report_generator.py        # Report generation (shared)
├── strategy/                  # Strategy implementations
│   ├── base_strategy.py       # Abstract base (shared)
│   ├── delta_neutral_strategy.py   # Equity strategies
│   ├── dv01_neutral_strategy.py    # FI strategies
│   └── convexity_neutral_strategy.py
├── equity/                    # Equity-specific implementation
│   ├── engine.py             # BacktestEngine
│   ├── config.py             # BacktestConfig
│   ├── state.py              # State tracking
│   ├── hedge_executor.py     # Spot/futures hedging
│   ├── results.py            # BacktestResults
│   └── metrics.py            # Performance metrics
└── fi/                        # Fixed Income implementation
    ├── engine.py             # FIBacktestEngine
    ├── config.py             # FIBacktestConfig
    ├── state.py              # FI state tracking
    ├── hedge_executor.py     # Bond futures hedging
    ├── results.py            # FIBacktestResults
    └── metrics.py            # FI-specific metrics
```

### 1. Base Protocols (`base.py`)

Defines the **Protocol interface** for all backtest components, ensuring consistency across asset classes.

#### Key Protocols:

##### **BaseHedgeExecutor**
```python
@runtime_checkable
class BaseHedgeExecutor(Protocol):
    """Interface for hedge execution across asset classes."""

    def execute_hedge(
        self,
        underlying: str,
        hedge_size: float,
        pricing_context: Any,
        current_time: datetime,
        reason: str = "hedge"
    ) -> BaseTradeRecord:
        """Execute hedge trade."""

    def get_hedge_position(self, underlying: str) -> Optional[Any]:
        """Get current hedge position."""

    def close_hedge_position(...) -> Optional[BaseTradeRecord]:
        """Close hedge position."""
```

**Implementation Pattern**:
- Equity: `HedgeExecutor` (handles spot/futures)
- FI: `FIHedgeExecutor` (handles bond futures)
- Both implement the same protocol interface

##### **BaseBacktestEngine**
```python
@runtime_checkable
class BaseBacktestEngine(Protocol):
    """Interface for backtest engines."""

    def run(self) -> Any:
        """Execute backtest and return results."""

    def _initialize(self) -> None:
        """Initialize portfolio and environment."""

    def _step(self, timestamp: datetime) -> None:
        """Execute single timestep."""
```

**Implementation Pattern**:
- Equity: `BacktestEngine`
- FI: `FIBacktestEngine`
- Both follow the same lifecycle: `_initialize()` → loop `_step()` → `_finalize()`

##### **BaseBacktestResults**
```python
@runtime_checkable
class BaseBacktestResults(Protocol):
    """Interface for backtest results."""

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""

    def get_total_pnl(self) -> float:
        """Get total P&L."""

    def get_pnl_series(self) -> Any:
        """Get P&L time series."""
```

##### **BaseTradeRecord**
```python
@dataclass
class BaseTradeRecord:
    """Common trade record structure."""
    timestamp: datetime
    trade_type: str
    instrument_type: str
    underlying: str
    quantity: float
    price: float
    notional: float
    transaction_cost: float
    reason: str
```

### 2. Strategy Layer (`strategy/`)

#### **BaseStrategy** (Abstract Base)

```python
class BaseStrategy(ABC):
    """
    Abstract base for all hedging strategies.

    Defines core strategy interface:
    - should_hedge(): When to hedge
    - calculate_hedge_size(): How much to hedge
    """

    def __init__(
        self,
        name: str,
        asset_class: AssetClass = AssetClass.GENERIC,
        hedging_target: HedgingTarget = HedgingTarget.DELTA,
        hedge_instrument: str = "spot",
    ):
```

**Key Attributes**:
- `asset_class`: EQUITY, FIXED_INCOME, or GENERIC
- `hedging_target`: DELTA, DV01, GAMMA, VEGA, CONVEXITY, DURATION
- `hedge_instrument`: spot, futures, bond_futures

**Abstract Methods**:

```python
@abstractmethod
def should_hedge(
    self,
    current_time: datetime,
    portfolio_greeks: Dict[str, float],
    market_data: Dict[str, float],
    **kwargs,
) -> bool:
    """Determine if hedging should be performed."""
    ...

@abstractmethod
def calculate_hedge_size(
    self,
    current_time: datetime,
    portfolio_greeks: Dict[str, float],
    market_data: Dict[str, float],
    **kwargs,
) -> float:
    """Calculate hedge size."""
    ...
```

#### **Concrete Strategies**

##### DeltaNeutralStrategy (`delta_neutral_strategy.py`)
- **Asset Class**: EQUITY
- **Target**: DELTA hedging
- **Instruments**: Spot or futures

```python
class DeltaNeutralStrategy(BaseStrategy):
    """
    Delta-neutral hedging strategy.

    Monitors portfolio delta and executes hedges when threshold breached.
    """

    def __init__(
        self,
        name: str,
        delta_threshold: float = 100.0,
        rebalance_frequency: str = 'daily',
        hedge_instrument: str = 'spot',
        hedge_ratio: float = 1.0,
        target_delta: float = 0.0,
        min_time_between_hedges: Optional[timedelta] = None,
    ):
```

**Key Parameters**:
- `delta_threshold`: Hedge when |delta| > threshold
- `rebalance_frequency`: 'daily', 'hourly', 'on_threshold', 'continuous'
- `hedge_instrument`: 'spot' or 'futures'
- `hedge_ratio`: 0.0 to 1.0 (proportion to hedge)
- `target_delta`: Target delta after hedging (typically 0.0)

##### DV01NeutralStrategy (`dv01_neutral_strategy.py`)
- **Asset Class**: FIXED_INCOME
- **Target**: DV01 hedging
- **Instruments**: Bond futures

```python
class DV01NeutralStrategy(BaseStrategy):
    """
    DV01-neutral hedging strategy.

    Monitors portfolio DV01 and hedges with bond futures.
    """

    def __init__(
        self,
        name: str,
        dv01_threshold: float = 50000.0,
        rebalance_frequency: str = 'daily',
        hedge_instrument: str = 'bond_futures',
        hedge_ratio: float = 1.0,
        target_dv01: float = 0.0,
        futures_dv01: float = 1000.0,
    ):
```

**Key Parameters**:
- `dv01_threshold`: Hedge when |DV01| > threshold
- `futures_dv01`: DV01 per futures contract (typically $1,000)
- Other parameters similar to DeltaNeutralStrategy

##### ConvexityNeutralStrategy (`convexity_neutral_strategy.py`)
- **Asset Class**: FIXED_INCOME
- **Target**: CONVEXITY hedging
- **Purpose**: Second-order rate risk hedging

### 3. Engine Layer

#### **Equity BacktestEngine** (`equity/engine.py`)

The main orchestrator for equity backtests:

```python
class BacktestEngine:
    """
    Main equity backtest engine.

    Execution flow:
    1. _initialize(): Setup portfolio, pricing env, hedge executor
    2. Loop through timestamps:
       - Update pricing environment
       - Calculate portfolio Greeks
       - Query strategy for hedge decision
       - Execute hedge if needed
       - Record state
    3. _finalize(): Generate results
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = BacktestLogger(...)
        self.greeks_calculator = GreeksCalculator()
        self.state_tracker = StateTracker()
```

**Core Methods**:

```python
def run(self) -> "BacktestResults":
    """Execute backtest lifecycle."""
    self._initialize()

    # Get market data
    self.market_data_set = self.config.market_data_adapter.get_market_data_set(...)

    # Main simulation loop
    for timestamp in timestamps:
        self._step(timestamp)

    return self._finalize()

def _step(self, timestamp: datetime):
    """Execute single timestep."""
    # 1. Update pricing environment
    self._update_pricing_environment(timestamp)

    # 2. Calculate Greeks
    if self.config.calculate_greeks:
        portfolio_greeks = self._calculate_greeks()

    # 3. Check if hedge needed
    if self.config.strategy.should_hedge(timestamp, portfolio_greeks, market_data):
        hedge_size = self.config.strategy.calculate_hedge_size(...)
        self._execute_hedge(hedge_size, timestamp)

    # 4. Record state
    self.state_tracker.record_state(...)

def _initialize(self):
    """Initialize all components."""
    # 1. Create portfolio
    self.portfolio = Portfolio(positions=self.config.initial_positions)

    # 2. Create pricing environment
    self.pricing_env = PricingEnvironment(...)

    # 3. Create hedge executor
    self.hedge_executor = HedgeExecutor(
        transaction_cost_model=self.config.transaction_cost_model,
        logger=self.logger
    )

    # 4. Store initial value
    self._initial_portfolio_value = self.portfolio.value(self.pricing_env)
```

**Integration Points**:
- **Portfolio**: `portfolio.Portfolio` (equity positions)
- **Pricing**: `priceenv.PricingEnvironment` (market data container)
- **Greeks**: `asset.equity.riskmeasures.GreeksCalculator`
- **Market Data**: Adapter pattern (Mock, Real, Database)

#### **FI BacktestEngine** (`fi/engine.py`)

Similar structure to equity engine but focused on fixed income:

```python
class FIBacktestEngine:
    """
    Fixed Income backtest engine.

    Differences from equity engine:
    - Tracks DV01 and convexity instead of Greeks
    - Uses FIHedgeExecutor (bond futures)
    - Calculates duration metrics
    """

    def __init__(self, config: FIBacktestConfig):
        self.config = config
        self.logger = BacktestLogger(...)
        self.state_tracker = FIStateTracker()
```

**Key Differences**:
- Portfolio type: `FIPortfolio` (bond positions)
- Hedge executor: `FIHedgeExecutor` (bond futures)
- Risk metrics: DV01, convexity, duration (not delta, gamma, vega)

### 4. Hedge Executors

#### **Equity HedgeExecutor** (`equity/hedge_executor.py`)

```python
class HedgeExecutor:
    """
    Executes hedge trades for equity strategies.

    Responsibilities:
    - Track hedge positions
    - Calculate hedge sizes
    - Execute trades (spot/futures)
    - Calculate transaction costs
    - Record trades
    """

    def __init__(
        self,
        transaction_cost_model: TransactionCostModel,
        logger: BacktestLogger,
    ):
        self.transaction_cost_model = transaction_cost_model
        self.logger = logger
        self.hedge_positions: Dict[str, Any] = {}
```

**Key Methods**:

```python
def execute_hedge(
    self,
    underlying: str,
    hedge_size: float,
    pricing_context: Any,
    current_time: datetime,
    reason: str = "hedge"
Record:
    """Execute hedge trade."""

) -> Trade1. Get current position
    current_qty = self.get_hedge_quantity(    # underlying)

    # 2. Calculate new position
    new_qty = current_qty + hedge_size

    # 3. Calculate trade quantity
    trade_qty = new_qty - current_qty

    # 4. Execute if non-zero
    if abs(trade_qty) > self.min_trade_size:
        price = pricing_context.get_price(underlying)
        notional = abs(trade_qty * price)
        cost = self.transaction_cost_model.calculate_cost(...)

        # Record trade
        trade = TradeRecord(...)
        self.trades.append(trade)

        # Update position
        self.hedge_positions[underlying] = new_qty

        return trade
```

#### **FI HedgeExecutor** (`fi/hedge_executor.py`)

Similar to equity executor but for bond futures:

```python
class FIHedgeExecutor:
    """
    Executes bond futures hedges for FI strategies.

    Key differences:
    - Tracks futures contracts (not spot)
    - Calculates DV01 per contract
    - Handles futures-specific costs
    """
```

### 5. Transaction Cost Models (`transaction_costs.py`)

Hierarchical cost model design with **Strategy pattern**:

```
TransactionCostModel (Abstract Base)
├── ZeroCostModel
├── FixedCostModel
├── ProportionalCostModel
└── CompleteCostModel
```

#### **TransactionCostModel** (Abstract Base)

```python
class TransactionCostModel(ABC):
    """Abstract base for all cost models."""

    @abstractmethod
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        notional: float,
        instrument_type: str,
        trade_type: str,
        **kwargs
    ) -> float:
        """Calculate transaction cost."""
```

#### **CompleteCostModel** (Most Comprehensive)

```python
class CompleteCostModel(TransactionCostModel):
    """
    Complete transaction cost model.

    Combines:
    - Fixed commission (per trade)
    - Proportional commission (% of notional)
    - Slippage (market impact)
    - Bid-ask spread
    """

    def calculate_cost(...) -> float:
        # 1. Fixed commission
        fixed = self.fixed_commission

        # 2. Proportional commission
        proportional = self.proportional_rate * notional

        # 3. Slippage
        slippage = self._calculate_slippage(notional, quantity)

        # 4. Bid-ask spread
        spread = self._calculate_spread(notional, instrument_type)

        return fixed + proportional + slippage + spread
```

**Cost Components**:

1. **Fixed Commission**: Flat fee per trade (e.g., $2)
2. **Proportional Commission**: Percentage of notional (e.g., 5 bps)
3. **Slippage**: Market impact (linear or sqrt model)
   ```python
   def _calculate_slippage(self, notional: float, quantity: float) -> float:
       if self.slippage_type == 'linear':
           return self.slippage_coefficient * notional
       else:  # sqrt model (more realistic for large trades)
           return self.slippage_coefficient * notional * np.sqrt(abs(quantity))
   ```
4. **Bid-Ask Spread**: Difference between bid and ask (e.g., 5 bps)

### 6. Configuration

#### **BacktestConfig** (`equity/config.py`)

```python
@dataclass
class BacktestConfig:
    """Configuration for equity backtests."""

    # Required parameters
    strategy: BaseStrategy
    start_date: datetime
    end_date: datetime
    underlying: str
    initial_positions: List[Position]
    market_data_adapter: BaseMarketDataAdapter
    transaction_cost_model: TransactionCostModel

    # Optional parameters
    frequency: str = "D"  # Daily, Hourly, Weekly, Monthly
    currency: str = "USD"
    logging_level: str = "INFO"
    results_path: Optional[str] = None
    save_snapshots: bool = True
    calculate_greeks: bool = True
    greeks_method: str = "analytical"
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Validation** (`_validate()`):
- Start date must be before end date
- Frequency must be valid ('D', 'H', 'M', 'W')
- Logging level must be valid
- Greeks method must be valid

#### **FIBacktestConfig** (`fi/config.py`)

Similar to equity config but for fixed income:
```python
@dataclass
class FIBacktestConfig:
    """Configuration for FI backtests."""
    # Similar structure but for FI portfolio and data
```

### 7. State Tracking

#### **Equity StateTracker** (`equity/state.py`)

```python
class StateTracker:
    """
    Tracks backtest state over time.

    Records:
    - Portfolio value
    - Greeks (delta, gamma, vega, theta, rho)
    - P&L
    - Hedge positions
    - Transaction costs
    """

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
        self.states_df.loc[timestamp] = {
            'portfolio_value': portfolio_value,
            'delta': greeks.get('delta', 0.0),
            'gamma': greeks.get('gamma', 0.0),
            'vega': greeks.get('vega', 0.0),
            'theta': greeks.get('theta', 0.0),
            'rho': greeks.get('rho', 0.0),
            'hedge_positions': hedge_positions,
            'transaction_costs': transaction_costs,
            'cumulative_costs': self._cumulative_costs + transaction_costs,
        }
```

**Output**: DataFrame with timestamps as index, metrics as columns

#### **FI StateTracker** (`fi/state.py`)

Tracks FI-specific metrics:
- DV01 (instead of delta)
- Convexity (instead of gamma)
- Duration (additional metric)

### 8. Results

#### **BacktestResults** (`equity/results.py`)

```python
class BacktestResults:
    """
    Complete backtest results.

    Provides:
    - Summary statistics
    - Time series accessors
    - Trade history
    - Performance metrics
    """

    def __init__(
        self,
        config: BacktestConfig,
        states_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        logger: BacktestLogger,
    ):
        self.config = config
        self.states_df = states_df
        self.trades_df = trades_df
        self.logger = logger
        self.metrics = BacktestMetrics(self)
```

**Key Accessors**:

```python
def get_summary(self) -> Dict[str, Any]:
    """Get summary statistics."""
    return {
        'total_pnl': self.get_total_pnl(),
        'total_return': self.get_total_return(),
        'num_hedges': len(self.trades_df),
        'total_transaction_costs': self.trades_df['transaction_cost'].sum(),
        'sharpe_ratio': self.metrics.sharpe_ratio(),
        'max_drawdown': self.metrics.max_drawdown(),
        # ... more metrics
    }

def get_pnl_series(self) -> pd.Series:
    """Get cumulative P&L time series."""
    initial_value = self.states_df['portfolio_value'].iloc[0]
    return self.states_df['portfolio_value'] - initial_value

def get_hedge_trades(self) -> pd.DataFrame:
    """Get trade history as DataFrame."""
    return self.trades_df
```

#### **FIBacktestResults** (`fi/results.py`)

Similar to equity results but with FI-specific methods:
- `get_dv01_series()`: DV01 time series
- `get_duration_series()`: Duration time series
- `get_convexity_series()`: Convexity time series

### 9. Metrics

#### **BacktestMetrics** (`equity/metrics.py`)

```python
class BacktestMetrics:
    """
    Calculate performance metrics.

    Categories:
    - P&L metrics (Sharpe, max drawdown, etc.)
    - Hedging metrics (hedge frequency, delta tracking error)
    - Risk metrics (VaR, CVaR)
    """

    def __init__(self, results: BacktestResults):
        self.results = results

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        returns = self.results.get_pnl_series().pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        pnl_series = self.results.get_pnl_series()
        peak = pnl_series.expanding().max()
        drawdown = (pnl_series - peak) / peak
        return drawdown.min()

    def hedge_frequency(self) -> float:
        """Calculate average hedges per day."""
        if len(self.results.trades_df) == 0:
            return 0.0
        days = (self.results.states_df.index[-1] - self.results.states_df.index[0]).days
        return len(self.results.trades_df) / days

    def delta_tracking_error(self) -> float:
        """Calculate RMSE of delta vs target."""
        delta_series = self.results.states_df['delta']
        target_delta = self.results.config.strategy.target_delta
        tracking_error = delta_series - target_delta
        return np.sqrt(np.mean(tracking_error ** 2))
```

**Metric Categories**:

1. **P&L Metrics**:
   - `total_pnl()`, `total_return()`
   - `sharpe_ratio()`, `sortino_ratio()`
   - `max_drawdown()`, `volatility()`
   - `win_rate()`, `profit_factor()`

2. **Hedging Metrics**:
   - `hedge_frequency()`: Hedges per day
   - `average_hedge_cost()`: Cost per hedge
   - `delta_tracking_error()`: RMSE of delta vs target
   - `average_absolute_delta()`: Mean abs(delta)

3. **Risk Metrics**:
   - `value_at_risk(confidence)`: VaR
   - `conditional_var(confidence)`: CVaR
   - `skewness()`, `kurtosis()`

#### **FIBacktestMetrics** (`fi/metrics.py`)

FI-specific metrics:
- `dv01_tracking_error()`: RMSE of DV01 vs target
- `average_absolute_dv01()`: Mean abs(DV01)
- `max_dv01_exposure()`: Maximum absolute DV01
- `dv01_hedge_effectiveness()`: Hedge effectiveness ratio
- `average_duration()`: Weighted-average duration

### 10. Visualization

#### **StaticVisualizer** (`visualizer.py`)

```python
class StaticVisualizer:
    """
    Create static plots using matplotlib.

    Plot types:
    - P&L over time
    - Delta/DV01 tracking
    - Greeks evolution
    - Drawdown chart
    - Returns distribution
    - Hedge frequency
    """

    def __init__(self, results: BacktestResults, save_dir: Optional[str] = None):
        self.results = results
        self.save_dir = Path(save_dir) if save_dir else Path("plots")
```

**Key Methods**:

```python
def plot_pnl_over_time(
    self,
    figsize: Tuple[int, int] = (14, 6),
    save: bool = False,
) -> plt.Figure:
    """Plot cumulative P&L."""
    fig, ax = plt.subplots(figsize=figsize)
    pnl_series = self.results.get_pnl_series()

    ax.plot(pnl_series.index, pnl_series.values, linewidth=2, color="steelblue")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.fill_between(pnl_series.index, 0, pnl_series.values, ...)

    ax.set_xlabel("Date")
    ax.set_ylabel("P&L ($)")
    ax.set_title(f"Portfolio P&L Over Time - {self.results.config.underlying}")
    return fig

def plot_delta_tracking(
    self,
    figsize: Tuple[int, int] = (14, 6),
    save: bool = False,
) -> plt.Figure:
    """Plot delta vs target."""
    fig, ax = plt.subplots(figsize=figsize)

    delta_series = self.results.states_df['delta']
    target_delta = self.results.config.strategy.target_delta

    ax.plot(delta_series.index, delta_series.values, label='Actual Delta')
    ax.axhline(y=target_delta, color='red', linestyle='--', label='Target Delta')
    ax.fill_between(delta_series.index, 0, delta_series.values, alpha=0.3)

    return fig

def create_summary_dashboard(self, save: bool = True) -> plt.Figure:
    """Create comprehensive dashboard."""
    # 2x2 subplot grid:
    # - P&L over time
    # - Delta tracking
    # - Drawdown
    # - Returns distribution
```

#### **InteractiveDashboard** (`dashboard.py`)

Plotly-based interactive visualizations:

```python
class InteractiveDashboard:
    """
    Create interactive plots using plotly.

    Features:
    - Zoom and pan
    - Hover tooltips
    - Multiple linked views
    - Export to HTML
    """

    def plot_pnl_interactive(self, save: bool = False) -> go.Figure:
        """Interactive P&L plot."""
        import plotly.graph_objects as go

        pnl_series = self.results.get_pnl_series()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pnl_series.index,
            y=pnl_series.values,
            mode='lines',
            name='P&L',
            line=dict(color='steelblue', width=2)
        ))

        fig.update_layout(
            title=f"Portfolio P&L - {self.results.config.underlying}",
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            hovermode='x unified'
        )

        return fig
```

### 11. Report Generation (`report_generator.py`)

```python
class ReportGenerator:
    """
    Generate comprehensive HTML and text reports.

    Report contents:
    - Executive summary
    - Strategy parameters
    - Performance metrics
    - Embedded visualizations
    - Trade history
    """

    def __init__(self, results: BacktestResults, output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)

    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report."""
        html = self._build_html_template()

        # Add summary section
        html += self._add_summary_section()

        # Add visualizations
        html += self._add_visualizations()

        # Add trade history
        html += self._add_trade_history()

        # Save report
        output_path = self.output_dir / f"{self.results.config.underlying}_report.html"
        output_path.write_text(html)

        return str(output_path)
```

### 12. Logging (`logger.py`)

```python
class BacktestLogger:
    """
    Comprehensive logging for backtests.

    Log levels:
    - DEBUG: Detailed execution info
    - INFO: General progress updates
    - WARNING: Non-fatal issues
    - ERROR: Fatal errors

    Log targets:
    - Console
    - File (rotating log files)
    """

    def __init__(
        self,
        log_dir: str,
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
        backtest_name: str = "backtest",
    ):
        self.logger = logging.getLogger(backtest_name)
        self.logger.setLevel(getattr(logging, log_level))

        # Configure handlers
        if enable_console:
            self._add_console_handler()

        if enable_file:
            self._add_file_handler(log_dir, backtest_name)
```

**Log Events**:
- Backtest start/end
- Hedge execution
- State updates
- Performance metrics
- Errors and warnings

## Usage Patterns

### Pattern 1: Basic Delta-Neutral Backtest

```python
from backtest import BacktestEngine, BacktestConfig, DeltaNeutralStrategy, ZeroCostModel

# Create strategy
strategy = DeltaNeutralStrategy(
    name="BasicDN",
    delta_threshold=100.0,
    rebalance_frequency='daily',
    hedge_instrument='spot'
)

# Configure backtest
config = BacktestConfig(
    strategy=strategy,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30),
    underlying="AAPL",
    initial_positions=[option_position],
    market_data_adapter=MockMarketDataAdapter(seed=42),
    transaction_cost_model=ZeroCostModel()
)

# Run backtest
engine = BacktestEngine(config)
results = engine.run()

# Analyze results
print(f"Total P&L: ${results.get_total_pnl():,.2f}")
print(f"Sharpe Ratio: {results.metrics.sharpe_ratio():.3f}")
```

### Pattern 2: Advanced Backtest with Transaction Costs

```python
from backtest import CompleteCostModel

# Configure realistic transaction costs
cost_model = CompleteCostModel(
    fixed_commission=2.0,           # $2 per trade
    proportional_rate=0.0005,       # 5 bps
    slippage_coefficient=0.0001,    # Linear slippage
    slippage_type='linear',
    spread_bps=5.0                  # 5 bps bid-ask spread
)

config = BacktestConfig(
    strategy=strategy,
    # ... other parameters
    transaction_cost_model=cost_model,
    logging_level='DEBUG',          # Detailed logs
    results_path='results/run1',    # Save results
    save_snapshots=True             # Save state snapshots
)
```

### Pattern 3: FI DV01-Neutral Backtest

```python
from backtest.fi import FIBacktestEngine, FIBacktestConfig
from backtest.strategy import DV01NeutralStrategy

# Configure DV01-neutral strategy
strategy = DV01NeutralStrategy(
    name="DV01_Neutral",
    dv01_threshold=50000.0,   # $50,000 DV01 threshold
    rebalance_frequency='daily',
    futures_dv01=1000.0       # $1,000 DV01 per contract
)

# Configure FI backtest
config = FIBacktestConfig(
    strategy=strategy,
    start_date=start_date,
    end_date=end_date,
    underlying="UST_10Y",
    initial_positions=[bond_position],
    market_data_adapter=future_data_adapter,
    transaction_cost_model=cost_model
)

# Run FI backtest
engine = FIBacktestEngine(config)
results = engine.run()

# Access FI-specific metrics
dv01_error = results.metrics.dv01_tracking_error()
print(f"DV01 Tracking Error: ${dv01_error:,.0f}")
```

### Pattern 4: Custom Strategy Implementation

```python
from backtest.strategy import BaseStrategy, AssetClass, HedgingTarget

class MyCustomStrategy(BaseStrategy):
    """Custom hedging strategy."""

    def __init__(self, name: str, custom_param: float):
        super().__init__(
            name=name,
            asset_class=AssetClass.EQUITY,
            hedging_target=HedgingTarget.DELTA,
            hedge_instrument='spot'
        )
        self.custom_param = custom_param

    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ) -> bool:
        """Custom hedge decision logic."""
        delta = portfolio_greeks.get('delta', 0.0)
        vol = market_data.get('volatility', 0.0)

        # Hedge if delta exceeds threshold AND volatility is high
        return abs(delta) > self.custom_param and vol > 0.3

    def calculate_hedge_size(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ) -> float:
        """Custom hedge size calculation."""
        delta = portfolio_greeks.get('delta', 0.0)
        target_delta = 0.0

        # Hedge 90% of delta
        return -0.9 * (delta - target_delta)
```

### Pattern 5: Visualization and Reporting

```python
from backtest import StaticVisualizer, InteractiveDashboard, ReportGenerator

# Static visualizations
visualizer = StaticVisualizer(results, save_dir="plots")
visualizer.plot_pnl_over_time(save=True)
visualizer.plot_delta_tracking(save=True)
visualizer.create_summary_dashboard(save=True)

# Interactive dashboard
dashboard = InteractiveDashboard(results, save_dir="plots/interactive")
dashboard.plot_pnl_interactive(save=True)
dashboard.create_comprehensive_dashboard(save=True)

# Generate report
report_gen = ReportGenerator(results, output_dir="reports")
html_path = report_gen.generate_html_report()
print(f"Report saved to: {html_path}")
```

### Pattern 6: Multi-Strategy Comparison

```python
strategies = [
    ("LowThreshold", DeltaNeutralStrategy(name="Low", delta_threshold=50.0)),
    ("HighThreshold", DeltaNeutralStrategy(name="High", delta_threshold=200.0)),
    ("SpotHedge", DeltaNeutralStrategy(name="Spot", hedge_instrument='spot')),
    ("FuturesHedge", DeltaNeutralStrategy(name="Futures", hedge_instrument='futures')),
]

results_comparison = {}
for name, strategy in strategies:
    config.strategy = strategy
    engine = BacktestEngine(config)
    results = engine.run()
    results_comparison[name] = results

# Compare results
for name, results in results_comparison.items():
    print(f"{name:15} P&L: ${results.get_total_pnl():>8,.0f}  "
          f"Sharpe: {results.metrics.sharpe_ratio():>6.3f}  "
          f"Hedges: {len(results.trades_df):>4d}")
```

## Performance Considerations

### Engine Performance

| Scenario | Expected Time | Notes |
|----------|---------------|-------|
| 6 months, daily data, 10 positions | 5-10 seconds | Basic equity backtest |
| 1 year, hourly data, 50 positions | 30-60 seconds | Higher frequency |
| With Greeks calculation | +50% overhead | Analytical Greeks faster |
| With transaction costs | +10% overhead | Cost calculation minimal |
| With snapshots saved | +20% overhead | I/O bound |

### Optimization Strategies

1. **Reduce Greeks Calculation Frequency**:
   ```python
   config = BacktestConfig(
       calculate_greeks=True,
       greeks_frequency='daily'  # Calculate every N steps
   )
   ```

2. **Disable Snapshots for Large Backtests**:
   ```python
   config = BacktestConfig(
       save_snapshots=False  # Reduces memory and I/O
   )
   ```

3. **Use Mock Data for Development**:
   ```python
   # Fast synthetic data
   adapter = MockMarketDataAdapter(seed=42)
   ```

4. **Optimize Strategy Check Frequency**:
   ```python
   strategy = DeltaNeutralStrategy(
       rebalance_frequency='daily',  # vs 'continuous'
       min_time_between_hedges=timedelta(hours=1)  # Rate limiting
   )
   ```

### Memory Usage

- **States DataFrame**: ~1MB per month (daily data, 20 columns)
- **Trades DataFrame**: ~1KB per 100 trades
- **Snapshots**: ~10MB per month (if enabled)

**Optimization**:
- Disable snapshots for large backtests
- Use daily instead of hourly data
- Limit backtest duration for testing

## Testing

### Test Structure

```
test/test_backtest.py
├── TestBacktestConfig          # Configuration validation
├── TestBacktestEngine          # Engine lifecycle
├── TestDeltaNeutralStrategy    # Strategy logic
├── TestTransactionCostModels   # Cost calculations
├── TestHedgeExecutor           # Hedge execution
└── TestBacktestResults         # Results access
```

### Running Tests

```bash
# All backtest tests
python -m pytest test/test_backtest.py -v

# Specific test category
python -m pytest test/test_backtest.py::TestDeltaNeutralStrategy -v

# With coverage
python -m pytest test/test_backtest.py --cov=backtest --cov-report=html
```

### Test Categories

1. **Unit Tests**: Individual components
   - Configuration validation
   - Strategy decision logic
   - Cost model calculations
   - Hedge execution

2. **Integration Tests**: Full workflows
   - End-to-end backtest execution
   - State tracking accuracy
   - Results generation

3. **Performance Tests**: Scalability
   - Large portfolio performance
   - Memory usage
   - Calculation time

### Example Test

```python
def test_delta_neutral_strategy_should_hedge():
    """Test strategy hedge decision logic."""
    strategy = DeltaNeutralStrategy(
        name="Test",
        delta_threshold=100.0,
        target_delta=0.0
    )

    # Should hedge when delta exceeds threshold
    assert strategy.should_hedge(
        datetime.now(),
        {'delta': 150.0},  # Above threshold
        {}
    ) == True

    # Should not hedge when delta within threshold
    assert strategy.should_hedge(
        datetime.now(),
        {'delta': 50.0},  # Below threshold
        {}
    ) == False
```

## Error Handling

### Exception Hierarchy

```
QuantArkException (base)
├── ValidationError       # Invalid configuration/inputs
├── MarketDataError       # Missing/invalid market data
├── NumericalError        # Numerical issues
└── BacktestError         # General backtest errors
```

### Common Errors and Solutions

1. **ValidationError: "Start date must be before end date"**
   - Check date ordering
   - Ensure timezone consistency

2. **ValidationError: "Invalid frequency"**
   - Must be 'D', 'H', 'M', or 'W'
   - Check frequency parameter

3. **MarketDataError: "Insufficient market data"**
   - Ensure data covers full date range
   - Check market data adapter

4. **NumericalError: "Greeks calculation failed"**
   - Check option parameters (strike, maturity)
   - Verify market data (spot, vol, rate)

### Debugging Tips

1. **Enable DEBUG Logging**:
   ```python
   config = BacktestConfig(
       logging_level='DEBUG',
       results_path='logs/debug_run'
   )
   ```

2. **Check Trade History**:
   ```python
   trades_df = results.get_hedge_trades()
   print(trades_df.head())
   ```

3. **Validate State Tracking**:
   ```python
   states_df = results.states_df
   print(states_df.describe())
   ```

4. **Inspect Greeks**:
   ```python
   delta_series = results.states_df['delta']
   print(delta_series.describe())
   ```

## Integration with Other Modules

### Portfolio Module

```python
from portfolio import Portfolio, Position
from portfolio.fi import FIPortfolio, FIPosition

# Equity portfolio
portfolio = Portfolio(positions=[option_pos1, option_pos2])

# FI portfolio
fi_portfolio = FIPortfolio(positions=[bond_pos1, bond_pos2])
```

### PriceEnv Module

```python
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve

# Create pricing environment
env = PricingEnvironment(
    spot=SpotQuote(symbol="AAPL", price=150.0),
    vol_surface=FlatVolSurface(surface={"AAPL": 0.25}),
    rate_curve=FlatRateCurve(curve={"USD": 0.05}),
    dividend_yield=ContinuousDividendYield(symbol="AAPL", rate=0.02)
)

# Portfolio uses this for valuation
portfolio_value = portfolio.value(env)
```

### Market Data Module

```python
from util.marketdata.adapter import MockMarketDataAdapter, RealMarketDataAdapter

# Mock (synthetic) data
adapter = MockMarketDataAdapter(seed=42)

# Real data from source
adapter = DatabaseMarketDataAdapter(
    connection_string="postgresql://...",
    query_config={...}
)
```

## Best Practices

### 1. Start Simple

```python
# Good: Start with zero costs
cost_model = ZeroCostModel()

# Then add costs
cost_model = CompleteCostModel(
    fixed_commission=2.0,
    proportional_rate=0.0005,
    # ... other costs
)
```

### 2. Validate Configuration

```python
config = BacktestConfig(...)
try:
    config.validate()
except ValidationError as e:
    print(f"Invalid config: {e}")
```

### 3. Use Appropriate Frequency

```python
# Good: Daily for most backtests
frequency = 'D'

# Only use hourly for high-frequency strategies
frequency = 'H'
```

### 4. Monitor Transaction Costs

```python
results = engine.run()
total_costs = results.trades_df['transaction_cost'].sum()
print(f"Total transaction costs: ${total_costs:,.2f}")
```

### 5. Save Results

```python
config = BacktestConfig(
    results_path='results/run_20241208',
    save_snapshots=True
)

# Results saved for later analysis
results.export_to_excel('results.xlsx')
```

### 6. Compare Strategies

```python
# Always compare multiple strategies
strategies = [strategy1, strategy2, strategy3]
for strategy in strategies:
    config.strategy = strategy
    results = BacktestEngine(config).run()
    # Compare results
```

### 7. Use Mock Data for Development

```python
# Fast iteration with mock data
adapter = MockMarketDataAdapter(seed=42)

# Validate with real data before production
adapter = RealMarketDataAdapter(...)
```

## Common Pitfalls

### 1. Mismatched Asset Classes

```python
# Wrong: FI strategy with equity backtest
strategy = DV01NeutralStrategy(...)  # FI strategy
config = BacktestConfig(...)  # Equity config
engine = BacktestEngine(config)  # Error!

# Correct: Match strategy and engine
strategy = DeltaNeutralStrategy(...)  # Equity strategy
config = BacktestConfig(...)
engine = BacktestEngine(config)  # Works!
```

### 2. Ignoring Transaction Costs

```python
# Wrong: Assuming zero costs
cost_model = ZeroCostModel()  # Unrealistic

# Correct: Use realistic costs
cost_model = CompleteCostModel(
    fixed_commission=2.0,
    proportional_rate=0.0005,
    spread_bps=5.0
)
```

### 3. Inappropriate Greeks Method

```python
# Wrong: Numerical Greeks for all positions
config = BacktestConfig(greeks_method='numerical')  # Slow

# Correct: Use analytical when possible
config = BacktestConfig(greeks_method='analytical')  # Fast
```

### 4. Not Validating Results

```python
# Wrong: Trusting results blindly
results = engine.run()
print(f"P&L: {results.get_total_pnl()}")

#
results = engine.get_total_pnl() > -1e6, "Unreasonable Correct: Validate results len(results.trades_df) > 0, "No trades executed"
.run()
assert results P&L"
assert```

## Future Enhancements (Potential TODOs)

 Optimization**: Mean1. **Portfolio-variance optimization for hedge selection
2. **Multi-Asset Strategies**: Cross-asset hedging (equity + FI)
3. **Machine Learning**: ML-based hedge timing and sizing
4. **Real-Time Backtesting**: Streaming market data integration
5. **Options Strategies**: Multi-leg option strategies
6. **Credit Risk**: Credit spread and default risk modeling
7. **FX Hedging**: Multi-currency portfolios
8. **Stress Testing**: Historical scenario replay
9. **Optimization Framework**: Parameter tuning and optimization
10. **Cloud Integration**: Distributed backtesting on cloud platforms

## API Reference

### Core Classes

```python
# Engine
class BacktestEngine:
    def __init__(self, config: BacktestConfig)
    def run(self) -> BacktestResults

class FIBacktestEngine:
    def __init__(self, config: FIBacktestConfig)
    def run(self) -> FIBacktestResults

# Strategies
class BaseStrategy(ABC):
    def should_hedge(...) -> bool
    def calculate_hedge_size(...) -> float

class DeltaNeutralStrategy(BaseStrategy)
class DV01NeutralStrategy(BaseStrategy)
class ConvexityNeutralStrategy(BaseStrategy)

# Configuration
@dataclass
class BacktestConfig:
    strategy: BaseStrategy
    start_date: datetime
    end_date: datetime
    underlying: str
    initial_positions: List[Position]
    market_data_adapter: BaseMarketDataAdapter
    transaction_cost_model: TransactionCostModel

@dataclass
class FIBacktestConfig:
    # Similar structure for FI

# Results
class BacktestResults:
    def get_summary(self) -> Dict[str, Any]
    def get_total_pnl(self) -> float
    def get_pnl_series(self) -> pd.Series
    def get_hedge_trades(self) -> pd.DataFrame

# Metrics
class BacktestMetrics:
    def sharpe_ratio(self) -> float
    def max_drawdown(self) -> float
    def hedge_frequency(self) -> float
    def delta_tracking_error(self) -> float
```

### Transaction Cost Models

```python
class TransactionCostModel(ABC):
    def calculate_cost(...) -> float

class ZeroCostModel(TransactionCostModel)
class FixedCostModel(TransactionCostModel)
class ProportionalCostModel(TransactionCostModel)
class CompleteCostModel(TransactionCostModel)
```

### Visualization

```python
class StaticVisualizer:
    def __init__(self, results: BacktestResults, save_dir: Optional[str] = None)
    def plot_pnl_over_time(...) -> plt.Figure
    def plot_delta_tracking(...) -> plt.Figure
    def create_summary_dashboard(...) -> plt.Figure

class InteractiveDashboard:
    def __init__(self, results: BacktestResults, save_dir: Optional[str] = None)
    def plot_pnl_interactive(...) -> go.Figure
    def create_comprehensive_dashboard(...) -> go.Figure
```

## References

### Academic Papers

1. Black, F., and Scholes, M. "The Pricing of Options and Corporate Liabilities"
2. Merton, R. C. "Theory of Rational Option Pricing"
3. Hull, J. "Options, Futures, and Other Derivatives"
4. Choudhry, M. "The Bond and Money Markets"

### Industry Resources

1. CFA Institute. "Derivatives and Alternative Investments"
2. ISDA. "Credit Support Documents"
3. CME Group. "Fixed Income Futures Documentation"

## Support and Resources

- **GitHub Issues**: QuantArk repository
- **Documentation**: QuantArk Docs
- **Email**: quantark-support@example.com
- **Internal Wiki**: [Internal Backtest Documentation]

---

**Note**: This module is actively maintained. For significant changes or new features, create an OpenSpec proposal following the project guidelines.
