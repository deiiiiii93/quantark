"""
Core dynamic scenario analysis engine.

This module contains the main engine for running dynamic scenario simulations
with time evolution and optional hedging strategies.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import time
from copy import deepcopy

from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.param import (
    SpotQuote,
    FlatVolSurface,
    FlatRateCurve,
    ContinuousDividendYield,
    FlatBasisYield,
    TermStructureDividendYield,
    TermStructureBasisYield,
)
from quantark.param.basis.basis_yield import (
    calculate_basis_from_rate_dividend,
    calculate_dividend_from_rate_basis,
)
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.equity.product.deltaone import SpotInstrument, Futures
from quantark.asset.equity.engine.analytical import DeltaOneEngine

from quantark.backtest.strategy.base_strategy import BaseStrategy
from quantark.backtest.transaction_costs import TransactionCostModel, ZeroCostModel

from quantark.dynamicscenario.config import DynamicScenarioConfig
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.dynamicscenario.lifecycle_manager import LifecycleManager
from quantark.dynamicscenario.results.dynamic_results import (
    DynamicScenarioResults, DayResult, LifecycleEventSnapshot, PositionSnapshot,
    TradeSnapshot, MarketState
)
from quantark.stresstest.stress.stress_types import (
    StressType,
    StressLevel,
    BasisDividendRelationshipMode,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import pnl_pct_of_abs_baseline


class DynamicScenarioEngine:
    """
    Engine for executing dynamic scenario analysis.
    
    This is the main entry point for running multi-day scenario simulations.
    It handles:
    - Applying day-by-day market changes to pricing environments
    - Calculating portfolio value and Greeks at each step
    - Optionally executing hedging strategies
    - Recording day-by-day state evolution
    
    Example:
        >>> config = DynamicScenarioConfig(calculate_greeks=True)
        >>> engine = DynamicScenarioEngine(config)
        >>> 
        >>> # Create a 5-day rally path
        >>> path = PathLibrary.consecutive_rally(days=5, daily_pct=0.02)
        >>> 
        >>> # Run simulation
        >>> results = engine.run(portfolio, path)
        >>> print(results.get_summary())
        
        >>> # Run with hedging
        >>> strategy = DeltaNeutralStrategy(delta_threshold=50)
        >>> results = engine.run(portfolio, path, hedge_strategy=strategy)
    """
    
    def __init__(self, config: Optional[DynamicScenarioConfig] = None):
        """
        Initialize dynamic scenario engine.
        
        Args:
            config: Configuration for execution
        """
        self.config = config or DynamicScenarioConfig()
        self.greeks_calculator = GreeksCalculator() if self.config.calculate_greeks else None
        self._deltaone_engine = DeltaOneEngine()
    
    def run(
        self,
        portfolio: Portfolio,
        day_path: DayPath,
        hedge_strategy: Optional[BaseStrategy] = None,
        transaction_cost_model: Optional[TransactionCostModel] = None
    ) -> DynamicScenarioResults:
        """
        Run dynamic scenario simulation.
        
        Simulates portfolio evolution through the day path, optionally
        applying hedging strategies at each step.
        
        Args:
            portfolio: Portfolio to simulate
            day_path: Day path defining market evolution
            hedge_strategy: Optional hedging strategy (from backtest module)
            transaction_cost_model: Optional transaction cost model
            
        Returns:
            DynamicScenarioResults with day-by-day evolution
            
        Raises:
            ValidationError: If portfolio or path is invalid
        """
        # Validate inputs
        if not portfolio or len(portfolio) == 0:
            raise ValidationError("Portfolio must contain at least one position")
        
        if not day_path or day_path.num_days == 0:
            raise ValidationError("Day path must have at least one day")
        
        start_time = time.time()
        print(f"Starting dynamic scenario: {day_path.name}")
        print(f"  Days: {day_path.num_days}")
        print(f"  Hedging: {'Yes' if hedge_strategy else 'No'}")
        
        # Use zero cost model if none provided
        if transaction_cost_model is None:
            transaction_cost_model = ZeroCostModel()
        
        # Create working copy of portfolio
        working_portfolio = self._clone_portfolio(portfolio)
        
        # Track state
        baseline_value = working_portfolio.get_portfolio_value()
        cumulative_transaction_costs = 0.0
        total_hedges = 0
        day_results: List[DayResult] = []
        previous_value = baseline_value
        
        # Reset strategy if provided
        if hedge_strategy:
            hedge_strategy.reset()
        
        # Track hedge positions
        hedge_positions: Dict[str, str] = {}  # underlying -> position_id

        # Lifecycle event handling
        lifecycle_manager: Optional[LifecycleManager] = None
        if self.config.handle_lifecycle_events:
            base_date = day_path.start_date
            if base_date is None:
                first_env = next(
                    iter(working_portfolio.pricing_environments.values()), None
                )
                base_date = first_env.valuation_date if first_env else None
            if base_date is None:
                raise ValidationError(
                    "Lifecycle event handling requires day_path.start_date or a "
                    "pricing environment valuation_date to anchor observation "
                    "schedules (or set handle_lifecycle_events=False)."
                )
            lifecycle_manager = LifecycleManager(base_date=base_date)
            lifecycle_manager.register_positions(working_portfolio)

        use_analytical = (self.config.greeks_method == 'analytical')

        # Run each day
        for day_step in day_path:
            print(f"  Processing Day {day_step.day_index}...")
            
            # Get date for this day
            day_date = day_path.get_date_for_day(day_step.day_index)
            
            # Apply day's market changes
            self._apply_day_changes(working_portfolio, day_step)
            
            # Update valuation date if we have dates
            if day_date:
                for env in working_portfolio.pricing_environments.values():
                    env.valuation_date = day_date

            # Process lifecycle events on this day's close
            lifecycle_events_today: List[LifecycleEventSnapshot] = []
            if lifecycle_manager is not None:
                lifecycle_events_today = lifecycle_manager.process_day(
                    working_portfolio, day_step.day_index, day_date
                )
            realized_cash = (
                lifecycle_manager.realized_cash if lifecycle_manager else 0.0
            )

            # Calculate portfolio value and Greeks
            portfolio_value = working_portfolio.get_portfolio_value() + realized_cash
            daily_pnl = portfolio_value - previous_value
            cumulative_pnl = portfolio_value - baseline_value
            
            portfolio_greeks = {}
            if self.config.calculate_greeks and self.greeks_calculator:
                portfolio_greeks = working_portfolio.get_portfolio_greeks(
                    self.greeks_calculator,
                    use_analytical=use_analytical
                )
            
            # Execute hedging if strategy provided
            trades_today: List[TradeSnapshot] = []
            transaction_costs_today = 0.0
            
            if hedge_strategy and len(working_portfolio) > 0:
                # Get market data for strategy
                market_data = self._get_market_data(working_portfolio)
                
                # Call strategy on_step
                current_time = day_date or datetime.now()
                hedge_strategy.on_step(
                    current_time=current_time,
                    portfolio_greeks=portfolio_greeks,
                    market_data=market_data
                )
                
                # Check if hedging needed
                should_hedge = hedge_strategy.should_hedge(
                    current_time=current_time,
                    portfolio_greeks=portfolio_greeks,
                    market_data=market_data
                )
                
                if should_hedge:
                    # Calculate hedge size
                    hedge_size = hedge_strategy.calculate_hedge_size(
                        current_time=current_time,
                        portfolio_greeks=portfolio_greeks,
                        market_data=market_data
                    )
                    
                    if abs(hedge_size) > 1e-10:
                        # Execute hedge for each underlying
                        for underlying in working_portfolio.pricing_environments.keys():
                            trade, cost = self._execute_hedge(
                                portfolio=working_portfolio,
                                underlying=underlying,
                                hedge_size=hedge_size,
                                transaction_cost_model=transaction_cost_model,
                                hedge_positions=hedge_positions,
                                current_time=current_time
                            )
                            
                            if trade:
                                trades_today.append(trade)
                                transaction_costs_today += cost
                                total_hedges += 1
                        
                        # Update strategy
                        hedge_strategy.on_hedge_executed(
                            current_time=current_time,
                            hedge_size=hedge_size,
                            hedge_price=market_data.get('spot', 0.0)
                        )
                        
                        # Recalculate Greeks after hedge
                        if self.config.calculate_greeks and self.greeks_calculator:
                            portfolio_greeks = working_portfolio.get_portfolio_greeks(
                                self.greeks_calculator,
                                use_analytical=use_analytical
                            )
            
            cumulative_transaction_costs += transaction_costs_today
            net_pnl = cumulative_pnl - cumulative_transaction_costs
            
            # Capture position snapshots
            position_snapshots = self._capture_position_snapshots(working_portfolio)
            
            # Capture market state
            market_state = self._capture_market_state(working_portfolio)
            
            # Create day result
            day_result = DayResult(
                day_index=day_step.day_index,
                date=day_date,
                label=day_step.label,
                portfolio_value=portfolio_value,
                daily_pnl=daily_pnl,
                cumulative_pnl=cumulative_pnl,
                transaction_costs_today=transaction_costs_today,
                cumulative_transaction_costs=cumulative_transaction_costs,
                net_pnl=net_pnl,
                greeks=portfolio_greeks,
                positions=position_snapshots,
                trades=trades_today,
                market_state=market_state,
                lifecycle_events=lifecycle_events_today,
                realized_cash=realized_cash,
            )
            day_results.append(day_result)
            
            # Update previous value for next iteration
            previous_value = portfolio_value
        
        total_time = time.time() - start_time
        
        # Build final results
        final_value = working_portfolio.get_portfolio_value() + (
            lifecycle_manager.realized_cash if lifecycle_manager else 0.0
        )
        
        results = DynamicScenarioResults(
            path_name=day_path.name,
            baseline_value=baseline_value,
            final_value=final_value,
            day_results=day_results,
            total_pnl=final_value - baseline_value,
            total_pnl_pct=pnl_pct_of_abs_baseline(final_value - baseline_value, baseline_value),
            total_transaction_costs=cumulative_transaction_costs,
            net_pnl=final_value - baseline_value - cumulative_transaction_costs,
            total_hedges=total_hedges,
            total_execution_time=total_time,
            config_summary=self.config.get_summary(),
            metadata={
                'path_description': day_path.description,
                'hedge_strategy': hedge_strategy.name if hedge_strategy else None,
            }
        )
        
        print(f"\nDynamic scenario completed in {total_time:.2f} seconds")
        print(f"  Final P&L: ${results.total_pnl:,.2f} ({results.total_pnl_pct:+.2f}%)")
        
        return results
    
    def _clone_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """Create a deep copy of portfolio with cloned pricing environments."""
        # Clone pricing environments
        cloned_envs = {}
        for underlying, env in portfolio.pricing_environments.items():
            cloned_envs[underlying] = PricingEnvironment(
                rate_curve=deepcopy(env.rate_curve),
                valuation_date=env.valuation_date,
                spot_quote=deepcopy(env.spot_quote) if env.spot_quote else None,
                vol_surface=deepcopy(env.vol_surface) if env.vol_surface else None,
                div_yield=deepcopy(env.div_yield) if env.div_yield else None,
                basis_yield=deepcopy(env.basis_yield) if env.basis_yield else None,
                day_count_convention=env.day_count_convention,
                bus_days_in_year=env.bus_days_in_year,
            )
        
        # Create new portfolio with cloned environments
        cloned_portfolio = Portfolio(
            portfolio_name=portfolio.portfolio_name + "_simulation",
            pricing_environments=cloned_envs,
            creation_date=portfolio.creation_date,
        )
        
        # Deep copy positions
        cloned_portfolio.positions = deepcopy(portfolio.positions)
        
        return cloned_portfolio
    
    def _apply_day_changes(self, portfolio: Portfolio, day_step: DayStep) -> None:
        """Apply a day's market changes to portfolio pricing environments."""
        for change in day_step.changes:
            if change.level == StressLevel.PORTFOLIO:
                # Apply to all underlyings
                for underlying in portfolio.pricing_environments.keys():
                    self._apply_parameter_change(
                        portfolio.pricing_environments[underlying],
                        change,
                        portfolio,
                        underlying,
                    )
            elif change.level == StressLevel.UNDERLYING:
                # Apply to specific underlying
                if change.target in portfolio.pricing_environments:
                    self._apply_parameter_change(
                        portfolio.pricing_environments[change.target],
                        change,
                        portfolio,
                        change.target,
                    )
            # POSITION level would need position-specific environments
    
    def _apply_parameter_change(
        self,
        env: PricingEnvironment,
        change: ParameterChange,
        portfolio: Portfolio,
        underlying: str,
    ) -> None:
        """Apply a single parameter change to a pricing environment."""
        param = change.parameter.lower()
        
        if param == "spot":
            if env.spot_quote:
                current = env.spot_quote.spot
                new_value = change.apply(current)
                if new_value <= 0:
                    raise ValidationError(f"Spot cannot be <= 0, got {new_value}")
                env.spot_quote = SpotQuote(
                    spot=new_value,
                    timestamp=env.spot_quote.timestamp,
                    asset_name=env.spot_quote.asset_name,
                )
        
        elif param in ["volatility", "vol"]:
            if env.vol_surface and isinstance(env.vol_surface, FlatVolSurface):
                current = env.vol_surface.volatility
                new_value = change.apply(current)
                if new_value <= 0:
                    raise ValidationError(f"Volatility cannot be <= 0, got {new_value}")
                env.vol_surface = FlatVolSurface(volatility=new_value)
        
        elif param == "rate":
            if env.rate_curve and isinstance(env.rate_curve, FlatRateCurve):
                current = env.rate_curve.get_rate(1.0)
                new_value = change.apply(current)
                env.rate_curve = FlatRateCurve(rate=new_value)
        
        elif param in ["dividend_yield", "div_yield", "dividend"]:
            if env.div_yield and isinstance(env.div_yield, ContinuousDividendYield):
                current = env.div_yield.div_yield
                new_value = change.apply(current)
                if new_value < 0:
                    new_value = 0.0
                env.div_yield = ContinuousDividendYield(div_yield=new_value)
            elif env.div_yield is None:
                # Create dividend yield if it doesn't exist
                new_value = change.apply(0.0)
                if new_value < 0:
                    new_value = 0.0
                env.div_yield = ContinuousDividendYield(div_yield=new_value)
            time_to_maturity = self._infer_time_to_maturity(
                change, portfolio, underlying, env
            )
            self._apply_basis_dividend_relationship(
                env,
                change,
                source_param="dividend",
                time_to_maturity=time_to_maturity,
            )

        elif param == "basis":
            time_to_maturity = self._infer_time_to_maturity(
                change, portfolio, underlying, env
            )
            current = (
                env.basis_yield.get_basis_yield(time_to_maturity)
                if env.basis_yield
                else 0.0
            )
            new_value = change.apply(current)
            env.basis_yield = FlatBasisYield(basis_yield=new_value)
            self._apply_basis_dividend_relationship(
                env,
                change,
                source_param="basis",
                time_to_maturity=time_to_maturity,
            )
    
    def _get_market_data(self, portfolio: Portfolio) -> Dict[str, float]:
        """Get current market data from portfolio for strategy."""
        # Use first underlying's environment
        first_underlying = list(portfolio.pricing_environments.keys())[0]
        env = portfolio.pricing_environments[first_underlying]
        
        return {
            'spot': env.spot if env.spot_quote else 0.0,
            'volatility': env.vol_surface.volatility if isinstance(env.vol_surface, FlatVolSurface) else 0.0,
            'rate': env.rate_curve.get_rate(1.0) if env.rate_curve else 0.0,
            'div_yield': env.div_yield.div_yield if isinstance(env.div_yield, ContinuousDividendYield) else 0.0,
            'basis_yield': env.basis_yield.get_basis_yield(1.0) if env.basis_yield else 0.0,
        }
    
    def _execute_hedge(
        self,
        portfolio: Portfolio,
        underlying: str,
        hedge_size: float,
        transaction_cost_model: TransactionCostModel,
        hedge_positions: Dict[str, str],
        current_time: datetime
    ) -> tuple:
        """
        Execute a hedge trade.
        
        Returns:
            Tuple of (TradeSnapshot or None, transaction_cost)
        """
        env = portfolio.pricing_environments[underlying]
        hedge_price = env.spot
        notional = abs(hedge_size * hedge_price)
        
        # Calculate transaction cost
        transaction_cost = transaction_cost_model.calculate_cost(
            quantity=hedge_size,
            price=hedge_price,
            notional=notional,
            instrument_type='spot',
            trade_type='hedge'
        )
        
        # Check if hedge position exists
        existing_position_id = hedge_positions.get(underlying)
        
        if existing_position_id and existing_position_id in portfolio.positions:
            # Update existing position
            position = portfolio.positions[existing_position_id]
            old_quantity = position.quantity
            new_quantity = old_quantity + hedge_size
            
            if abs(new_quantity) < 1e-10:
                # Close position
                portfolio.remove_position(existing_position_id)
                del hedge_positions[underlying]
                trade_type = 'close'
            else:
                position.quantity = new_quantity
                trade_type = 'adjust'
            
            trade = TradeSnapshot(
                trade_type=trade_type,
                underlying=underlying,
                instrument_type='spot',
                quantity=hedge_size,
                price=hedge_price,
                notional=notional,
                transaction_cost=transaction_cost,
                reason='delta_hedge'
            )
        else:
            # Create new hedge position
            from quantark.util.enum.deltaone_enums import DeltaOneType
            hedge_product = SpotInstrument(
                underlying=underlying,
                deltaone_type=DeltaOneType.STOCK
            )
            
            position = portfolio.add_position(
                product=hedge_product,
                quantity=hedge_size,
                entry_price=hedge_price,
                underlying=underlying,
                engine=self._deltaone_engine,
                entry_timestamp=current_time
            )
            
            hedge_positions[underlying] = position.position_id
            
            trade = TradeSnapshot(
                trade_type='open',
                underlying=underlying,
                instrument_type='spot',
                quantity=hedge_size,
                price=hedge_price,
                notional=notional,
                transaction_cost=transaction_cost,
                reason='delta_hedge'
            )
        
        return trade, transaction_cost
    
    def _capture_position_snapshots(self, portfolio: Portfolio) -> List[PositionSnapshot]:
        """Capture snapshots of all positions."""
        snapshots = []
        
        for position_id, position in portfolio.positions.items():
            env = portfolio.pricing_environments[position.underlying]
            market_value = position.get_market_value(env)
            pnl = position.get_pnl(env)
            
            # Get position Greeks if calculating
            greeks = None
            if self.config.calculate_greeks and self.greeks_calculator:
                use_analytical = (self.config.greeks_method == 'analytical')
                greeks = position.get_greeks(
                    env,
                    self.greeks_calculator,
                    use_analytical=use_analytical
                )
            
            snapshot = PositionSnapshot(
                position_id=position_id,
                underlying=position.underlying,
                product_type=position.product.__class__.__name__,
                quantity=position.quantity,
                market_value=market_value,
                entry_price=position.entry_price,
                pnl=pnl,
                greeks=greeks,
            )
            snapshots.append(snapshot)
        
        return snapshots
    
    def _capture_market_state(self, portfolio: Portfolio) -> MarketState:
        """Capture current market state from portfolio."""
        spot = {}
        volatility = {}
        div_yield = {}
        basis_yield = {}
        rate = 0.0
        
        for underlying, env in portfolio.pricing_environments.items():
            if env.spot_quote:
                spot[underlying] = env.spot_quote.spot
            
            if isinstance(env.vol_surface, FlatVolSurface):
                volatility[underlying] = env.vol_surface.volatility
            
            if isinstance(env.div_yield, ContinuousDividendYield):
                div_yield[underlying] = env.div_yield.div_yield

            if env.basis_yield is not None:
                basis_yield[underlying] = env.basis_yield.get_basis_yield(1.0)
            
            # Use rate from first environment
            if env.rate_curve and rate == 0.0:
                rate = env.rate_curve.get_rate(1.0)
        
        return MarketState(
            spot=spot,
            volatility=volatility,
            rate=rate,
            div_yield=div_yield,
            basis_yield=basis_yield,
        )

    def _apply_basis_dividend_relationship(
        self,
        env: PricingEnvironment,
        change: ParameterChange,
        source_param: str,
        time_to_maturity: float,
    ) -> None:
        """Apply basis-dividend relationship adjustments based on config."""
        relationship_mode = self.config.basis_dividend_relationship_mode
        if time_to_maturity <= 0:
            raise ValidationError(
                f"time_to_maturity must be positive, got {time_to_maturity}"
            )

        rate = env.rate_curve.get_rate(time_to_maturity)

        if source_param == "basis":
            if relationship_mode == BasisDividendRelationshipMode.AUTO_ADJUST_DIVIDEND:
                basis_val = env.basis_yield.get_basis_yield(time_to_maturity)
                new_div = calculate_dividend_from_rate_basis(rate, basis_val)
                if isinstance(env.div_yield, TermStructureDividendYield):
                    current_div = env.div_yield.get_yield(time_to_maturity)
                    shift = new_div - current_div
                    new_yields = [max(0.0, float(y) + shift) for y in env.div_yield.yields]
                    env.div_yield = TermStructureDividendYield(
                        times=list(env.div_yield.times), yields=new_yields
                    )
                else:
                    if new_div < 0:
                        new_div = 0.0
                    env.div_yield = ContinuousDividendYield(div_yield=new_div)
            elif relationship_mode == BasisDividendRelationshipMode.SYNCHRONIZED:
                current_div = env.div_yield.get_yield(time_to_maturity) if env.div_yield else 0.0
                new_div = change.apply(current_div)
                if isinstance(env.div_yield, TermStructureDividendYield):
                    shift = new_div - current_div
                    new_yields = [max(0.0, float(y) + shift) for y in env.div_yield.yields]
                    env.div_yield = TermStructureDividendYield(
                        times=list(env.div_yield.times), yields=new_yields
                    )
                else:
                    if new_div < 0:
                        new_div = 0.0
                    env.div_yield = ContinuousDividendYield(div_yield=new_div)
        elif source_param == "dividend":
            if relationship_mode == BasisDividendRelationshipMode.AUTO_ADJUST_BASIS:
                div_val = env.div_yield.get_yield(time_to_maturity) if env.div_yield else 0.0
                new_basis = calculate_basis_from_rate_dividend(rate, div_val)
                if isinstance(env.basis_yield, TermStructureBasisYield):
                    current_basis = env.basis_yield.get_basis_yield(time_to_maturity)
                    shift = new_basis - current_basis
                    new_yields = [float(y) + shift for y in env.basis_yield.yields]
                    if any(abs(y) > 0.50 for y in new_yields):
                        raise ValidationError(
                            "Stressed term-structure basis yields must be within +/-50%."
                        )
                    env.basis_yield = TermStructureBasisYield(
                        times=list(env.basis_yield.times), yields=new_yields
                    )
                else:
                    env.basis_yield = FlatBasisYield(basis_yield=new_basis)
            elif relationship_mode == BasisDividendRelationshipMode.SYNCHRONIZED:
                current_basis = env.basis_yield.get_basis_yield(time_to_maturity) if env.basis_yield else 0.0
                new_basis = change.apply(current_basis)
                if isinstance(env.basis_yield, TermStructureBasisYield):
                    shift = new_basis - current_basis
                    new_yields = [float(y) + shift for y in env.basis_yield.yields]
                    if any(abs(y) > 0.50 for y in new_yields):
                        raise ValidationError(
                            "Stressed term-structure basis yields must be within +/-50%."
                        )
                    env.basis_yield = TermStructureBasisYield(
                        times=list(env.basis_yield.times), yields=new_yields
                    )
                else:
                    env.basis_yield = FlatBasisYield(basis_yield=new_basis)

    def _infer_time_to_maturity(
        self,
        change: ParameterChange,
        portfolio: Portfolio,
        underlying: str,
        env: PricingEnvironment,
    ) -> float:
        """Infer time to maturity from metadata or futures positions."""
        if change.metadata and "time_to_maturity" in change.metadata:
            time_to_maturity = float(change.metadata.get("time_to_maturity", 1.0))
            if time_to_maturity <= 0:
                raise ValidationError(
                    f"time_to_maturity must be positive, got {time_to_maturity}"
                )
            return time_to_maturity

        if hasattr(portfolio, "get_positions_by_underlying"):
            positions = portfolio.get_positions_by_underlying(underlying)
        else:
            positions = [
                pos for pos in portfolio.positions.values()
                if pos.underlying == underlying
            ]
        for position in positions:
            product = position.product
            if isinstance(product, Futures):
                try:
                    time_to_maturity = product.get_maturity(env)
                except Exception:
                    continue
                if time_to_maturity is not None and time_to_maturity > 0:
                    return float(time_to_maturity)

        return 1.0
    
    def __repr__(self) -> str:
        return f"DynamicScenarioEngine(config={self.config})"
