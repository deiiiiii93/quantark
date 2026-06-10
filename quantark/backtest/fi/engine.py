"""
Fixed Income backtest engine for simulating hedging strategies.
"""
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
from copy import deepcopy

from .config import FIBacktestConfig
from .state import FIBacktestState, FIStateTracker, FITradeRecord
from .hedge_executor import FIHedgeExecutor
from backtest.logger import BacktestLogger
from portfolio.fi import FIPortfolio
from priceenv import PricingEnvironment
from param import FlatRateCurve
from util.exceptions import ValidationError


class FIBacktestEngine:
    """
    Fixed Income backtest engine for simulating hedging strategies.
    
    Orchestrates the backtest by:
    1. Loading market data time series (rate curves)
    2. Stepping through time
    3. Updating pricing environments
    4. Calculating portfolio DV01/convexity
    5. Querying strategy for hedge decisions
    6. Executing hedges using bond futures
    7. Recording state and performance
    
    Attributes:
        config: FI backtest configuration
        portfolio: FI portfolio being managed
        state_tracker: Tracks state history
        logger: Backtest logger
        hedge_executor: Executes hedge trades
    """
    
    def __init__(self, config: FIBacktestConfig):
        """
        Initialize FI backtest engine.
        
        Args:
            config: FI backtest configuration
        """
        self.config = config
        
        # Initialize logger
        self.logger = BacktestLogger(
            log_dir=config.results_path if config.results_path else "logs",
            log_level=config.logging_level,
            enable_console=True,
            enable_file=True,
            backtest_name=f"FI_{config.underlying}_{config.strategy.name}"
        )
        
        # State tracker
        self.state_tracker = FIStateTracker()
        
        # Will be initialized in run()
        self.portfolio: Optional[FIPortfolio] = None
        self.hedge_executor: Optional[FIHedgeExecutor] = None
        self.pricing_env: Optional[PricingEnvironment] = None
        self.market_data_set = None
        
        # Performance tracking
        self._initial_portfolio_value: float = 0.0
        self._cumulative_transaction_costs: float = 0.0
        self._num_hedges_executed: int = 0
    
    def run(self) -> 'FIBacktestResults':
        """
        Execute the FI backtest.
        
        Returns:
            FIBacktestResults with complete history and metrics
        """
        # Log start
        self.logger.log_backtest_start(self.config.get_summary())
        
        # Initialize
        self._initialize()
        
        # Get market data time series
        self.market_data_set = self.config.market_data_adapter.get_market_data_set(
            asset_name=self.config.underlying,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            currency=self.config.currency,
            frequency=self.config.frequency
        )
        
        self.logger.logger.info(
            f"Loaded FI market data: {len(self.market_data_set.rate_data)} data points"
        )
        
        # Get timestamps from rate data
        timestamps = self.market_data_set.rate_data.data.index
        
        self.logger.logger.info(
            f"Backtesting from {timestamps[0]} to {timestamps[-1]} "
            f"({len(timestamps)} steps)"
        )
        
        # Main backtest loop
        for i, timestamp in enumerate(timestamps):
            self._step(timestamp)
            
            # Log progress
            if (i + 1) % max(1, len(timestamps) // 10) == 0:
                progress = (i + 1) / len(timestamps) * 100
                self.logger.logger.info(f"Progress: {progress:.1f}% ({i+1}/{len(timestamps)})")
        
        # Finalize
        results = self._finalize()
        
        # Log completion
        self.logger.log_backtest_end(results.get_summary())
        self.logger.save_structured_logs()
        
        return results
    
    def _initialize(self):
        """Initialize portfolio, pricing environment, and hedge executor."""
        # Create initial pricing environment with default rate
        from param import SpotQuote, FlatVolSurface, ContinuousDividendYield
        
        self.pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0, asset_name=self.config.underlying),
            vol_surface=FlatVolSurface(volatility=0.01),  # Low vol for bonds
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.0),
            valuation_date=self.config.start_date
        )
        
        # Create FI portfolio with pricing environment
        self.portfolio = FIPortfolio(
            portfolio_name=f"FI_Backtest_{self.config.underlying}",
            pricing_environments={self.config.underlying: self.pricing_env},
            creation_date=self.config.start_date
        )
        
        # Add initial positions
        for position in self.config.initial_positions:
            self.portfolio.add_position(
                product=position.product,
                quantity=position.quantity,
                entry_price=position.entry_price,
                underlying=position.underlying,
                engine=position.engine,
                entry_timestamp=position.entry_timestamp,
                notional_per_unit=getattr(position, 'notional_per_unit', 100.0)
            )
        
        self.logger.logger.info(
            f"Initialized FI portfolio with {len(self.config.initial_positions)} positions"
        )
        
        # Record initial portfolio value
        self._initial_portfolio_value = self.portfolio.get_portfolio_value()
        
        # Create hedge executor
        futures_dv01 = self.config.strategy.futures_dv01 if hasattr(self.config.strategy, 'futures_dv01') else 1000.0
        
        self.hedge_executor = FIHedgeExecutor(
            portfolio=self.portfolio,
            transaction_cost_model=self.config.transaction_cost_model,
            futures_spec=self.config.hedge_futures_spec,
            futures_dv01=futures_dv01
        )
        
        # Reset strategy
        self.config.strategy.reset()
        
        self.logger.logger.info("FI initialization complete")
    
    def _step(self, timestamp: datetime):
        """Execute a single backtest step."""
        # Update pricing environment with current market data
        self._update_pricing_environment(timestamp)
        
        # Calculate portfolio risk measures
        portfolio_risk_measures = {}
        if self.config.calculate_risk_measures:
            portfolio_risk_measures = self.portfolio.get_portfolio_risk_measures()
            # Add hedge DV01 to get net exposure
            hedge_dv01 = self.hedge_executor.get_hedge_dv01()
            portfolio_risk_measures['dv01'] = portfolio_risk_measures.get('dv01', 0) + hedge_dv01
        
        # Get market data
        market_data = self._get_current_market_data(timestamp)
        
        # Call strategy on_step
        self.config.strategy.on_step(
            current_time=timestamp,
            portfolio_greeks=portfolio_risk_measures,  # Named 'greeks' for compatibility
            market_data=market_data
        )
        
        # Check if hedging is needed
        should_hedge = self.config.strategy.should_hedge(
            current_time=timestamp,
            portfolio_greeks=portfolio_risk_measures,
            market_data=market_data
        )
        
        # Log hedge decision
        current_dv01 = portfolio_risk_measures.get('dv01', 0.0)
        self.logger.log_hedge_decision(
            timestamp=timestamp,
            should_hedge=should_hedge,
            current_delta=current_dv01,  # Using delta field for DV01
            threshold=self.config.strategy.dv01_threshold,
            reason="dv01_threshold_check"
        )
        
        # Execute hedge if needed
        trade_records = []
        if should_hedge:
            trade_record = self._execute_hedge(
                timestamp=timestamp,
                portfolio_risk_measures=portfolio_risk_measures,
                market_data=market_data
            )
            if trade_record:
                trade_records.append(trade_record)
        
        # Record state
        self._record_state(
            timestamp=timestamp,
            portfolio_risk_measures=portfolio_risk_measures,
            market_data=market_data,
            trade_records=trade_records
        )
    
    def _update_pricing_environment(self, timestamp: datetime):
        """Update pricing environment with current market data."""
        # Get current rate data
        rate_data = self.market_data_set.rate_data.get_at_date(timestamp)
        
        # Update rate curve
        self.pricing_env.rate_curve = FlatRateCurve(
            rate=rate_data['rate']
        )
        
        self.pricing_env.valuation_date = timestamp
    
    def _get_current_market_data(self, timestamp: datetime) -> Dict[str, float]:
        """Get current market data as dictionary."""
        rate = self.pricing_env.get_rate(1.0)
        return {
            'rate': rate,
            'yield_10y': rate,  # Simplified
            'timestamp': timestamp
        }
    
    def _execute_hedge(
        self,
        timestamp: datetime,
        portfolio_risk_measures: Dict[str, float],
        market_data: Dict[str, float]
    ) -> Optional[FITradeRecord]:
        """Execute hedge trade."""
        # Calculate hedge size (number of futures contracts)
        hedge_size = self.config.strategy.calculate_hedge_size(
            current_time=timestamp,
            portfolio_greeks=portfolio_risk_measures,
            market_data=market_data
        )
        
        if abs(hedge_size) < 0.5:  # Less than half a contract
            return None
        
        # Execute hedge
        trade_record = self.hedge_executor.execute_hedge(
            underlying=self.config.underlying,
            hedge_size=hedge_size,
            pricing_env=self.pricing_env,
            current_time=timestamp,
            reason="dv01_hedge"
        )
        
        # Update strategy state
        self.config.strategy.on_hedge_executed(
            current_time=timestamp,
            hedge_size=hedge_size,
            hedge_price=trade_record.price
        )
        
        # Update tracking
        self._cumulative_transaction_costs += trade_record.transaction_cost
        if abs(hedge_size) >= 0.5:
            self._num_hedges_executed += 1
        
        # Log trade
        self.logger.log_trade(
            timestamp=timestamp,
            trade_type=trade_record.trade_type,
            underlying=trade_record.underlying,
            quantity=trade_record.quantity,
            price=trade_record.price,
            notional=trade_record.notional,
            transaction_cost=trade_record.transaction_cost,
            reason=trade_record.reason
        )
        
        return trade_record
    
    def _record_state(
        self,
        timestamp: datetime,
        portfolio_risk_measures: Dict[str, float],
        market_data: Dict[str, float],
        trade_records: list
    ):
        """Record current state."""
        # Calculate portfolio value and P&L
        portfolio_value = self.portfolio.get_portfolio_value()
        portfolio_pnl = self.portfolio.get_portfolio_pnl()
        
        # Add hedge P&L
        hedge_stats = self.hedge_executor.get_statistics()
        hedge_pnl = hedge_stats.get('cumulative_hedge_pnl', 0)
        
        # Calculate net P&L (after transaction costs)
        net_pnl = portfolio_pnl + hedge_pnl - self._cumulative_transaction_costs
        
        # Create state
        state = FIBacktestState(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            pnl=net_pnl,
            bond_pnl=portfolio_pnl,
            hedge_pnl=hedge_pnl,
            transaction_costs=self._cumulative_transaction_costs,
            num_positions=len(self.portfolio.positions),
            num_hedges=1 if abs(self.hedge_executor.get_hedge_quantity(self.config.underlying)) > 0 else 0,
            risk_measures=portfolio_risk_measures,
            market_data=market_data,
            trades=trade_records
        )
        
        # Add to tracker
        self.state_tracker.add_state(state)
        
        # Log state periodically
        if len(self.state_tracker) % 10 == 0 or len(trade_records) > 0:
            self.logger.log_state(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                pnl=net_pnl,
                delta=portfolio_risk_measures.get('dv01', 0.0)
            )
    
    def _finalize(self) -> 'FIBacktestResults':
        """Finalize backtest and create results."""
        from .results import FIBacktestResults
        
        # Create results object
        results = FIBacktestResults(
            config=self.config,
            state_tracker=self.state_tracker,
            initial_value=self._initial_portfolio_value,
            final_value=self.portfolio.get_portfolio_value(),
            num_hedges=self._num_hedges_executed,
            total_transaction_costs=self._cumulative_transaction_costs
        )
        
        return results
    
    def __repr__(self) -> str:
        return (
            f"FIBacktestEngine("
            f"strategy={self.config.strategy.name}, "
            f"underlying={self.config.underlying})"
        )

