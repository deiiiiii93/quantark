"""
Main equity backtest engine for simulating hedging strategies.
"""

from typing import Optional, Dict, Any, Union
from datetime import datetime
import pandas as pd
from copy import deepcopy

from .config import BacktestConfig
from .state import BacktestState, StateTracker, TradeRecord
from .hedge_executor import HedgeExecutor
from .multi_hedge_executor import MultiInstrumentHedgeExecutor
from quantark.backtest.logger import BacktestLogger
from quantark.backtest.strategy.multi_greek_strategy import MultiGreekHedgeStrategy
from quantark.portfolio import Portfolio
from quantark.portfolio.equity.swap_position import EquitySwapPosition
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.asset.equity.lifecycle import PortfolioLifecycleManager
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero


class BacktestEngine:
    """
    Main backtest engine for simulating hedging strategies.

    Orchestrates the backtest by:
    1. Loading market data time series
    2. Stepping through time
    3. Updating pricing environments
    4. Calculating portfolio Greeks
    5. Querying strategy for hedge decisions
    6. Executing hedges
    7. Recording state and performance

    Attributes:
        config: Backtest configuration
        portfolio: Portfolio being managed
        state_tracker: Tracks state history
        logger: Backtest logger
        hedge_executor: Executes hedge trades
        greeks_calculator: Calculates portfolio Greeks
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config

        # Initialize logger
        self.logger = BacktestLogger(
            log_dir=config.results_path if config.results_path else "logs",
            log_level=config.logging_level,
            enable_console=True,
            enable_file=True,
            backtest_name=f"{config.underlying}_{config.strategy.name}",
        )

        # Initialize Greeks calculator
        self.greeks_calculator = GreeksCalculator()

        # State tracker
        self.state_tracker = StateTracker()

        # Will be initialized in run()
        self.portfolio: Optional[Portfolio] = None
        self.hedge_executor: Optional[
            Union[HedgeExecutor, MultiInstrumentHedgeExecutor]
        ] = None
        self.pricing_env: Optional[PricingEnvironment] = None
        self.market_data_set = None

        # Lifecycle event handling (Snowball/Phoenix/barrier-family)
        self.lifecycle_manager: Optional[PortfolioLifecycleManager] = None
        self._terminated_lifecycle_cost_basis: float = 0.0
        self._lifecycle_events_today: list = []

        # Performance tracking
        self._initial_portfolio_value: float = 0.0
        self._cumulative_transaction_costs: float = 0.0
        self._num_hedges_executed: int = 0

    def run(self) -> "BacktestResults":
        """
        Execute the backtest.

        Returns:
            BacktestResults with complete history and metrics
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
            frequency=self.config.frequency,
        )

        self.logger.logger.info(
            f"Loaded market data: {len(self.market_data_set.spot_data)} data points"
        )

        # Get common timestamps
        timestamps = self.market_data_set.spot_data.data.index

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
                self.logger.logger.info(
                    f"Progress: {progress:.1f}% ({i+1}/{len(timestamps)})"
                )

        # Finalize
        results = self._finalize()

        # Log completion
        self.logger.log_backtest_end(results.get_summary())
        self.logger.save_structured_logs()

        return results

    def _initialize(self):
        """Initialize portfolio, pricing environment, and hedge executor."""
        # Create initial pricing environment (will be updated each step)
        self.pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0, asset_name=self.config.underlying),
            vol_surface=FlatVolSurface(volatility=0.25),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
            valuation_date=self.config.start_date,
        )

        # Create portfolio with pricing environment
        self.portfolio = Portfolio(
            portfolio_name=f"Backtest_{self.config.underlying}",
            pricing_environments={self.config.underlying: self.pricing_env},
            creation_date=self.config.start_date,
        )

        # Add initial positions. TRS swap positions price through their cashflow
        # valuator (no separate engine), so register them via add_swap_position;
        # payoff-on-spot positions carry an engine and use add_position.
        for position in self.config.initial_positions:
            if isinstance(position, EquitySwapPosition):
                self.portfolio.add_swap_position(
                    product=position.product,
                    quantity=position.quantity,
                    underlying=position.underlying,
                    entry_price=position.entry_price,
                    funding_rate_ref=position.funding_rate_ref,
                    entry_timestamp=position.entry_timestamp,
                )
            else:
                self.portfolio.add_position(
                    product=position.product,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    underlying=position.underlying,
                    engine=position.engine,
                    entry_timestamp=position.entry_timestamp,
                )

        self.logger.logger.info(
            f"Initialized portfolio with {len(self.config.initial_positions)} positions"
        )

        # Record initial portfolio value
        self._initial_portfolio_value = self.portfolio.get_portfolio_value()

        # Attach lifecycle trackers to lifecycle-bearing positions. Registered
        # once, before stepping, so hedge instruments added later stay
        # untracked. Snowball/Phoenix and barrier-family products get realized
        # KO/KI/coupon/maturity/expiry handling; everything else is a no-op.
        if self.config.handle_lifecycle_events:
            self.lifecycle_manager = PortfolioLifecycleManager(
                base_date=self.config.start_date
            )
            self.lifecycle_manager.register_positions(self.portfolio)
            if self.lifecycle_manager.num_tracked > 0:
                self.logger.logger.info(
                    f"Lifecycle tracking enabled for "
                    f"{self.lifecycle_manager.num_tracked} position(s)"
                )

        # Create hedge executor (multi-instrument strategies trade a basket
        # of instruments; single-target strategies keep the spot/futures path)
        if isinstance(self.config.strategy, MultiGreekHedgeStrategy):
            self.hedge_executor = MultiInstrumentHedgeExecutor(
                portfolio=self.portfolio,
                transaction_cost_model=self.config.transaction_cost_model,
                instruments=self.config.strategy.hedge_instruments,
                greeks_calculator=self.greeks_calculator,
            )
        else:
            self.hedge_executor = HedgeExecutor(
                portfolio=self.portfolio,
                transaction_cost_model=self.config.transaction_cost_model,
                hedge_instrument_type=self.config.strategy.hedge_instrument,
            )

        # Reset strategy
        self.config.strategy.reset()

        self.logger.logger.info("Initialization complete")

    def _step(self, timestamp: datetime):
        """
        Execute a single backtest step.

        Args:
            timestamp: Current timestamp
        """
        # Update pricing environment with current market data
        self._update_pricing_environment(timestamp)

        # Process realized lifecycle events on this day's close. Determined
        # cashflows remain as receivables until their payment dates; knocked-in
        # barriers reprice as their European equivalent. Processed before
        # Greeks so the hedge sizes against the surviving contingent portfolio.
        self._lifecycle_events_today = []
        if self.lifecycle_manager is not None:
            self._process_lifecycle(timestamp)

        # Roll expiring hedge contracts before computing Greeks, so the
        # hedge solve never sizes against a contract about to vanish
        trade_records = []
        if isinstance(self.hedge_executor, MultiInstrumentHedgeExecutor):
            roll_records = self.hedge_executor.process_rolls(
                underlying=self.config.underlying,
                pricing_env=self.pricing_env,
                current_time=timestamp,
            )
            for record in roll_records:
                self._cumulative_transaction_costs += record.transaction_cost
                self.logger.log_trade(
                    timestamp=timestamp,
                    trade_type=record.trade_type,
                    underlying=record.underlying,
                    quantity=record.quantity,
                    price=record.price,
                    notional=record.notional,
                    transaction_cost=record.transaction_cost,
                    reason=record.reason,
                )
            trade_records.extend(roll_records)

        # Calculate portfolio Greeks
        portfolio_greeks = {}
        if self.config.calculate_greeks:
            portfolio_greeks = self.portfolio.get_portfolio_greeks(
                greeks_calculator=self.greeks_calculator,
                use_analytical=(self.config.greeks_method == "analytical"),
            )

        # Strategies may need portfolio-level measures beyond Greeks
        # (e.g. scenario P&L); merge them into the decision dict
        if isinstance(self.config.strategy, MultiGreekHedgeStrategy):
            extra_measures = self.config.strategy.compute_portfolio_measures(
                self.portfolio, self.config.underlying, self.pricing_env
            )
            if extra_measures:
                portfolio_greeks = {**portfolio_greeks, **extra_measures}

        # Get market data
        market_data = self._get_current_market_data(timestamp)

        # Call strategy on_step
        self.config.strategy.on_step(
            current_time=timestamp,
            portfolio_greeks=portfolio_greeks,
            market_data=market_data,
        )

        # Check if hedging is needed
        should_hedge = self.config.strategy.should_hedge(
            current_time=timestamp,
            portfolio_greeks=portfolio_greeks,
            market_data=market_data,
        )

        # Log hedge decision
        current_delta = portfolio_greeks.get("delta", 0.0)
        self.logger.log_hedge_decision(
            timestamp=timestamp,
            should_hedge=should_hedge,
            current_delta=current_delta,
            threshold=getattr(self.config.strategy, "delta_threshold", 0.0),
            reason="threshold_check",
        )

        # Execute hedge if needed
        if should_hedge:
            trade_records.extend(
                self._execute_hedge(
                    timestamp=timestamp,
                    portfolio_greeks=portfolio_greeks,
                    market_data=market_data,
                )
            )

        # Record state
        self._record_state(
            timestamp=timestamp,
            portfolio_greeks=portfolio_greeks,
            market_data=market_data,
            trade_records=trade_records,
        )

    def _process_lifecycle(self, timestamp: datetime):
        """
        Detect and apply realized lifecycle events for this day's close.

        Delegates event detection and position mutation/removal to the shared
        ``PortfolioLifecycleManager`` registers immutable determined
        cashflows, retains their pending PV until payment, and removes
        terminated contingent positions. This method preserves the cost basis
        of removed positions for P&L attribution.

        Args:
            timestamp: Current timestamp (the day's close).
        """
        # Capture cost basis before the manager mutates/removes positions, so
        # realized P&L matches the (price - entry) * quantity convention used
        # by Portfolio.get_portfolio_pnl for still-live positions.
        pre_costs = {
            pid: pos.entry_price * pos.quantity
            for pid, pos in self.portfolio.positions.items()
        }
        events = self.lifecycle_manager.process_day(
            self.portfolio, day_index=0, day_date=timestamp
        )
        for item in events:
            if item.event.terminates_position:
                self._terminated_lifecycle_cost_basis += pre_costs.get(
                    item.position_id,
                    0.0,
                )
        self._lifecycle_events_today = events

    def _update_pricing_environment(self, timestamp: datetime):
        """Update pricing environment with current market data."""
        # Get current market data point
        spot_data = self.market_data_set.spot_data.get_at_date(timestamp)
        vol_data = self.market_data_set.vol_data.get_at_date(timestamp)
        rate_data = self.market_data_set.rate_data.get_at_date(timestamp)

        div_data = None
        if self.market_data_set.div_yield_data is not None:
            div_data = self.market_data_set.div_yield_data.get_at_date(timestamp)

        # Update pricing environment
        self.pricing_env.spot_quote = SpotQuote(
            spot=spot_data["spot"], asset_name=self.config.underlying
        )
        self.pricing_env.vol_surface = FlatVolSurface(volatility=vol_data["volatility"])
        self.pricing_env.rate_curve = FlatRateCurve(rate=rate_data["rate"])

        if div_data:
            self.pricing_env.div_yield = ContinuousDividendYield(
                div_yield=div_data["div_yield"]
            )

        self.pricing_env.valuation_date = timestamp

    def _get_current_market_data(self, timestamp: datetime) -> Dict[str, float]:
        """Get current market data as dictionary."""
        return {
            "spot": self.pricing_env.spot,
            "volatility": self.pricing_env.get_vol(self.pricing_env.spot, 1.0),
            "rate": self.pricing_env.get_rate(1.0),
            "div_yield": self.pricing_env.get_div_yield(1.0),
            "timestamp": timestamp,
        }

    def _execute_hedge(
        self,
        timestamp: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
    ) -> list:
        """
        Execute hedge trade(s).

        Single-target strategies trade one instrument; multi-Greek
        strategies solve for and trade a basket.

        Args:
            timestamp: Current timestamp
            portfolio_greeks: Portfolio Greeks
            market_data: Market data

        Returns:
            List of TradeRecords (possibly empty)
        """
        if isinstance(self.config.strategy, MultiGreekHedgeStrategy):
            return self._execute_multi_hedge(timestamp, portfolio_greeks, market_data)

        # Calculate hedge size
        hedge_size = self.config.strategy.calculate_hedge_size(
            current_time=timestamp,
            portfolio_greeks=portfolio_greeks,
            market_data=market_data,
        )

        if is_zero(hedge_size):
            return []

        # Execute hedge
        trade_record = self.hedge_executor.execute_hedge(
            underlying=self.config.underlying,
            hedge_size=hedge_size,
            pricing_env=self.pricing_env,
            current_time=timestamp,
            reason="delta_hedge",
        )

        # Update strategy state
        self.config.strategy.on_hedge_executed(
            current_time=timestamp,
            hedge_size=hedge_size,
            hedge_price=trade_record.price,
        )

        # Update tracking
        self._cumulative_transaction_costs += trade_record.transaction_cost
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
            reason=trade_record.reason,
        )

        return [trade_record]

    def _execute_multi_hedge(
        self,
        timestamp: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
    ) -> list:
        """
        Execute a multi-instrument hedge rebalance.

        Asks the executor for the per-unit Greeks of each tradeable
        contract, solves for quantities via the strategy, and executes.

        Args:
            timestamp: Current timestamp
            portfolio_greeks: Portfolio Greeks
            market_data: Market data

        Returns:
            List of TradeRecords (possibly empty)
        """
        instrument_greeks = self.hedge_executor.get_instrument_measures(
            underlying=self.config.underlying,
            pricing_env=self.pricing_env,
            current_time=timestamp,
            measure_provider=self.config.strategy,
        )

        quantities = self.config.strategy.calculate_hedge_quantities(
            current_time=timestamp,
            portfolio_greeks=portfolio_greeks,
            market_data=market_data,
            instrument_greeks=instrument_greeks,
        )

        records = self.hedge_executor.execute_hedges(
            underlying=self.config.underlying,
            quantities=quantities,
            pricing_env=self.pricing_env,
            current_time=timestamp,
        )

        if records:
            self._num_hedges_executed += 1
            self.config.strategy.on_hedges_executed(timestamp, quantities)

        for record in records:
            self._cumulative_transaction_costs += record.transaction_cost
            self.logger.log_trade(
                timestamp=timestamp,
                trade_type=record.trade_type,
                underlying=record.underlying,
                quantity=record.quantity,
                price=record.price,
                notional=record.notional,
                transaction_cost=record.transaction_cost,
                reason=record.reason,
            )

        return records

    def _record_state(
        self,
        timestamp: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        trade_records: list,
    ):
        """Record current state."""
        # Fixed receivables and paid lifecycle cash stay in portfolio value, so
        # value (and P&L) is continuous across determination and payment.
        pending_receivable_pv = (
            self.lifecycle_manager.pending_receivable_pv
            if self.lifecycle_manager is not None
            else 0.0
        )
        paid_cash = (
            self.lifecycle_manager.paid_cash
            if self.lifecycle_manager is not None
            else 0.0
        )
        portfolio_value = (
            self.portfolio.get_portfolio_value()
            + pending_receivable_pv
            + paid_cash
        )
        portfolio_pnl = self.portfolio.get_portfolio_pnl()

        # Calculate net P&L (after transaction costs), including realized
        # P&L from hedge contracts that were closed or rolled and from
        # settled/coupon-paying lifecycle positions
        realized_pnl = getattr(self.hedge_executor, "realized_pnl", 0.0)
        net_pnl = (
            portfolio_pnl
            + pending_receivable_pv
            + paid_cash
            - self._terminated_lifecycle_cost_basis
            + realized_pnl
            - self._cumulative_transaction_costs
        )

        # Create state
        state = BacktestState(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            pnl=net_pnl,
            transaction_costs=self._cumulative_transaction_costs,
            num_positions=len(self.portfolio.positions),
            greeks=portfolio_greeks,
            market_data=market_data,
            trades=trade_records,
            cash=paid_cash,
            pending_receivable_pv=pending_receivable_pv,
            paid_cash=paid_cash,
            realized_cash=paid_cash,
            lifecycle_events=self._lifecycle_events_today,
        )

        # Add to tracker
        self.state_tracker.add_state(state)

        # Log state (periodically to avoid spam)
        if len(self.state_tracker) % 10 == 0 or len(trade_records) > 0:
            self.logger.log_state(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                pnl=net_pnl,
                delta=portfolio_greeks.get("delta", 0.0),
            )

    def _finalize(self) -> "BacktestResults":
        """Finalize backtest and create results."""
        from .results import BacktestResults

        # Pending receivables and paid lifecycle cash remain in final value
        # after terminated contingent positions have been removed.
        pending_receivable_pv = (
            self.lifecycle_manager.pending_receivable_pv
            if self.lifecycle_manager is not None
            else 0.0
        )
        paid_cash = (
            self.lifecycle_manager.paid_cash
            if self.lifecycle_manager is not None
            else 0.0
        )

        # Create results object
        results = BacktestResults(
            config=self.config,
            state_tracker=self.state_tracker,
            initial_value=self._initial_portfolio_value,
            final_value=(
                self.portfolio.get_portfolio_value()
                + pending_receivable_pv
                + paid_cash
            ),
            num_hedges=self._num_hedges_executed,
            total_transaction_costs=self._cumulative_transaction_costs,
        )

        return results

    def __repr__(self) -> str:
        return (
            f"BacktestEngine("
            f"strategy={self.config.strategy.name}, "
            f"underlying={self.config.underlying})"
        )


# Alias for explicit naming
EquityBacktestEngine = BacktestEngine
