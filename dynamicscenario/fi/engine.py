"""
FI Dynamic Scenario Engine.

This module contains the main engine for running dynamic scenario simulations
on Fixed Income portfolios with rate curve evolution and DV01 hedging.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import time
from copy import deepcopy

from portfolio.fi.portfolio import FIPortfolio
from priceenv import PricingEnvironment
from param import FlatRateCurve

from backtest.strategy.base_strategy import BaseStrategy
from backtest.transaction_costs import TransactionCostModel, ZeroCostModel

from dynamicscenario.base import BaseDynamicScenarioEngine
from dynamicscenario.fi.config import FIDynamicScenarioConfig
from dynamicscenario.fi.results import (
    FIDynamicScenarioResults,
    FIDayResult,
    FIMarketState,
    FITradeSnapshot,
)
from dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from stresstest.stress.stress_types import StressType, StressLevel
from util.exceptions import ValidationError
from util.numerical import pnl_pct_of_abs_baseline


class FIDynamicScenarioEngine(BaseDynamicScenarioEngine):
    """
    Engine for executing FI dynamic scenario analysis.
    
    This is the main entry point for running multi-day scenario simulations
    on Fixed Income portfolios. It handles:
    - Applying day-by-day rate curve changes to pricing environments
    - Calculating DV01, convexity, and duration at each step
    - Optionally executing DV01-based hedging strategies
    - Recording day-by-day state evolution
    
    Example:
        >>> config = FIDynamicScenarioConfig(
        ...     calculate_dv01=True,
        ...     hedge_enabled=True,
        ...     hedge_dv01_threshold=50000,
        ... )
        >>> engine = FIDynamicScenarioEngine(config)
        >>> 
        >>> # Create a rate hike path
        >>> path = FIPathLibrary.rate_hike_cycle(days=10, total_bps=100)
        >>> 
        >>> # Run simulation
        >>> results = engine.run(fi_portfolio, path)
        >>> print(results.get_summary())
    """
    
    def __init__(self, config: Optional[FIDynamicScenarioConfig] = None):
        """
        Initialize FI dynamic scenario engine.
        
        Args:
            config: Configuration for execution
        """
        self.config = config or FIDynamicScenarioConfig()
    
    def supports_portfolio(self, portfolio: Any) -> bool:
        """Check if this engine supports the given portfolio type."""
        return isinstance(portfolio, FIPortfolio)
    
    def get_asset_class(self) -> str:
        """Get the asset class this engine handles."""
        return "fixed_income"
    
    def run(
        self,
        portfolio: FIPortfolio,
        day_path: DayPath,
        hedge_strategy: Optional[BaseStrategy] = None,
        transaction_cost_model: Optional[TransactionCostModel] = None
    ) -> FIDynamicScenarioResults:
        """
        Run FI dynamic scenario simulation.
        
        Simulates portfolio evolution through the day path, optionally
        applying DV01 hedging strategies at each step.
        
        Args:
            portfolio: FI Portfolio to simulate
            day_path: Day path defining rate curve evolution
            hedge_strategy: Optional hedging strategy (DV01NeutralStrategy)
            transaction_cost_model: Optional transaction cost model
            
        Returns:
            FIDynamicScenarioResults with day-by-day evolution
            
        Raises:
            ValidationError: If portfolio or path is invalid
        """
        # Validate inputs
        if not self.supports_portfolio(portfolio):
            raise ValidationError("FIDynamicScenarioEngine requires an FIPortfolio instance")
        
        if len(portfolio) == 0:
            raise ValidationError("Portfolio must contain at least one position")
        
        if not day_path or day_path.num_days == 0:
            raise ValidationError("Day path must have at least one day")
        
        start_time = time.time()
        print(f"Starting FI dynamic scenario: {day_path.name}")
        print(f"  Days: {day_path.num_days}")
        print(f"  Hedging: {'Yes' if (hedge_strategy or self.config.hedge_enabled) else 'No'}")
        
        # Use zero cost model if none provided
        if transaction_cost_model is None:
            transaction_cost_model = ZeroCostModel()
        
        # Create working copy of portfolio
        working_portfolio = self._clone_portfolio(portfolio)
        
        # Track state
        baseline_value = working_portfolio.get_portfolio_value()
        baseline_dv01 = working_portfolio.get_portfolio_dv01()
        baseline_duration = working_portfolio.get_portfolio_duration()
        
        cumulative_transaction_costs = 0.0
        total_hedges = 0
        day_results: List[FIDayResult] = []
        previous_value = baseline_value
        
        # Track hedge positions
        hedge_positions: Dict[str, float] = {}  # instrument -> contracts
        
        # Track cumulative rate changes for curve state
        cumulative_rate_change = 0.0
        cumulative_short_change = 0.0
        cumulative_long_change = 0.0
        
        # Reset strategy if provided
        if hedge_strategy:
            hedge_strategy.reset()
        
        # Run each day
        for day_step in day_path:
            print(f"  Processing Day {day_step.day_index}...")
            
            # Get date for this day
            day_date = day_path.get_date_for_day(day_step.day_index)
            
            # Apply day's rate changes
            rate_changes = self._apply_day_changes(working_portfolio, day_step)
            cumulative_rate_change += rate_changes.get('rate', 0)
            cumulative_short_change += rate_changes.get('short', 0)
            cumulative_long_change += rate_changes.get('long', 0)
            
            # Update valuation date if we have dates
            if day_date:
                for env in working_portfolio.pricing_environments.values():
                    env.valuation_date = day_date
            
            # Calculate portfolio value
            portfolio_value = working_portfolio.get_portfolio_value()
            daily_pnl = portfolio_value - previous_value
            cumulative_pnl = portfolio_value - baseline_value
            
            # Calculate FI risk measures (pre-hedge)
            dv01_pre_hedge = working_portfolio.get_portfolio_dv01()
            convexity = working_portfolio.get_portfolio_convexity()
            duration = working_portfolio.get_portfolio_duration()
            
            # Calculate key-rate DV01 if enabled
            key_rate_dv01 = {}
            if self.config.calculate_key_rate_dv01:
                key_rate_dv01 = self._calculate_key_rate_dv01(
                    working_portfolio,
                    self.config.key_rate_tenors
                )
            
            # Execute hedging if enabled
            trades_today: List[FITradeSnapshot] = []
            transaction_costs_today = 0.0
            dv01_post_hedge = dv01_pre_hedge
            
            if self.config.hedge_enabled or hedge_strategy:
                should_hedge = self._should_hedge(
                    dv01=dv01_pre_hedge,
                    threshold=self.config.hedge_dv01_threshold,
                    strategy=hedge_strategy,
                    day_date=day_date,
                )
                
                if should_hedge:
                    trade, cost, dv01_impact = self._execute_hedge(
                        dv01=dv01_pre_hedge,
                        hedge_positions=hedge_positions,
                        transaction_cost_model=transaction_cost_model,
                    )
                    
                    if trade:
                        trades_today.append(trade)
                        transaction_costs_today += cost
                        total_hedges += 1
                        dv01_post_hedge = dv01_pre_hedge + dv01_impact
            
            cumulative_transaction_costs += transaction_costs_today
            net_pnl = cumulative_pnl - cumulative_transaction_costs
            
            # Capture market state
            market_state = self._capture_market_state(
                working_portfolio,
                cumulative_rate_change,
                cumulative_short_change,
                cumulative_long_change,
            )
            
            # Create day result
            day_result = FIDayResult(
                day_index=day_step.day_index,
                date=day_date,
                label=day_step.label,
                portfolio_value=portfolio_value,
                daily_pnl=daily_pnl,
                cumulative_pnl=cumulative_pnl,
                transaction_costs_today=transaction_costs_today,
                cumulative_transaction_costs=cumulative_transaction_costs,
                net_pnl=net_pnl,
                risk_metrics={
                    'dv01': dv01_post_hedge,
                    'convexity': convexity,
                    'modified_duration': duration,
                },
                dv01_pre_hedge=dv01_pre_hedge,
                dv01_post_hedge=dv01_post_hedge,
                convexity=convexity,
                modified_duration=duration,
                key_rate_dv01=key_rate_dv01,
                market_state=market_state,
                hedge_positions=dict(hedge_positions),
                trades=trades_today,
            )
            day_results.append(day_result)
            
            # Update previous value for next iteration
            previous_value = portfolio_value
        
        total_time = time.time() - start_time
        
        # Build final results
        final_value = working_portfolio.get_portfolio_value()
        final_dv01 = working_portfolio.get_portfolio_dv01()
        final_duration = working_portfolio.get_portfolio_duration()
        
        results = FIDynamicScenarioResults(
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
            baseline_dv01=baseline_dv01,
            baseline_duration=baseline_duration,
            final_dv01=final_dv01,
            final_duration=final_duration,
            metadata={
                'path_description': day_path.description,
                'hedge_enabled': self.config.hedge_enabled,
            }
        )
        
        print(f"\nFI dynamic scenario completed in {total_time:.2f} seconds")
        print(f"  Final P&L: ${results.total_pnl:,.2f} ({results.total_pnl_pct:+.2f}%)")
        print(f"  Final DV01: ${final_dv01:,.2f}")
        
        return results
    
    def _clone_portfolio(self, portfolio: FIPortfolio) -> FIPortfolio:
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
                day_count_convention=env.day_count_convention,
                bus_days_in_year=env.bus_days_in_year,
            )
        
        # Create new portfolio with cloned environments
        cloned_portfolio = FIPortfolio(
            portfolio_name=portfolio.portfolio_name + "_simulation",
            pricing_environments=cloned_envs,
            creation_date=portfolio.creation_date,
        )
        
        # Deep copy positions
        cloned_portfolio.positions = deepcopy(portfolio.positions)
        
        return cloned_portfolio
    
    def _apply_day_changes(
        self,
        portfolio: FIPortfolio,
        day_step: DayStep
    ) -> Dict[str, float]:
        """
        Apply a day's rate changes to portfolio pricing environments.
        
        Returns:
            Dictionary of rate changes applied (rate, short, long)
        """
        rate_changes = {'rate': 0.0, 'short': 0.0, 'long': 0.0}
        
        for change in day_step.changes:
            if change.level == StressLevel.PORTFOLIO:
                # Apply to all underlyings
                for underlying in portfolio.pricing_environments.keys():
                    self._apply_parameter_change(
                        portfolio.pricing_environments[underlying],
                        change,
                        rate_changes,
                    )
            elif change.level == StressLevel.UNDERLYING:
                # Apply to specific underlying
                if change.target in portfolio.pricing_environments:
                    self._apply_parameter_change(
                        portfolio.pricing_environments[change.target],
                        change,
                        rate_changes,
                    )
        
        return rate_changes
    
    def _apply_parameter_change(
        self,
        env: PricingEnvironment,
        change: ParameterChange,
        rate_changes: Dict[str, float],
    ) -> None:
        """Apply a single parameter change to a pricing environment."""
        param = change.parameter.lower()
        
        if param in ['rate', 'rate_parallel']:
            if env.rate_curve and isinstance(env.rate_curve, FlatRateCurve):
                current = env.rate_curve.get_rate(1.0)
                new_value = change.apply(current)
                env.rate_curve = FlatRateCurve(rate=new_value)
                rate_changes['rate'] += change.stress_value
        
        elif param == 'rate_short':
            # For flat curve, we track short rate changes but apply uniformly
            if env.rate_curve and isinstance(env.rate_curve, FlatRateCurve):
                current = env.rate_curve.get_rate(1.0)
                # Apply short rate bump (simplified for flat curve)
                new_value = change.apply(current)
                env.rate_curve = FlatRateCurve(rate=new_value)
                rate_changes['short'] += change.stress_value
        
        elif param == 'rate_long':
            # For flat curve, track long rate changes
            if env.rate_curve and isinstance(env.rate_curve, FlatRateCurve):
                current = env.rate_curve.get_rate(1.0)
                new_value = change.apply(current)
                env.rate_curve = FlatRateCurve(rate=new_value)
                rate_changes['long'] += change.stress_value
    
    def _calculate_key_rate_dv01(
        self,
        portfolio: FIPortfolio,
        tenors: List[float]
    ) -> Dict[float, float]:
        """
        Calculate key-rate DV01 at specified tenors.
        
        This is a simplified implementation that distributes total DV01
        across tenors based on position maturities.
        """
        # For now, return empty dict (placeholder for future implementation)
        # A full implementation would require tenor-specific bumps
        return {}
    
    def _should_hedge(
        self,
        dv01: float,
        threshold: float,
        strategy: Optional[BaseStrategy],
        day_date: Optional[datetime],
    ) -> bool:
        """Determine if hedging should be executed."""
        if strategy:
            # Use strategy's logic if provided
            return abs(dv01) > threshold
        
        # Default: hedge if DV01 exceeds threshold
        return abs(dv01) > threshold
    
    def _execute_hedge(
        self,
        dv01: float,
        hedge_positions: Dict[str, float],
        transaction_cost_model: TransactionCostModel,
    ) -> tuple:
        """
        Execute a DV01 hedge trade.
        
        Returns:
            Tuple of (FITradeSnapshot or None, transaction_cost, dv01_impact)
        """
        # Calculate contracts needed to neutralize DV01
        dv01_per_contract = self.config.futures_dv01_per_contract
        contracts_needed = -dv01 / dv01_per_contract  # Opposite sign to hedge
        
        if abs(contracts_needed) < 0.5:
            return None, 0.0, 0.0
        
        # Round to whole contracts
        contracts = round(contracts_needed)
        dv01_impact = contracts * dv01_per_contract
        
        # Assume futures price ~100
        futures_price = 100.0
        notional = abs(contracts) * futures_price * 1000  # Per $1000 face
        
        # Calculate transaction cost
        transaction_cost = transaction_cost_model.calculate_cost(
            quantity=abs(contracts),
            price=futures_price,
            notional=notional,
            instrument_type='futures',
            trade_type='hedge'
        )
        
        # Update hedge positions
        instrument = "bond_futures"
        current_position = hedge_positions.get(instrument, 0)
        new_position = current_position + contracts
        hedge_positions[instrument] = new_position
        
        # Determine trade type
        if current_position == 0:
            trade_type = 'open'
        elif new_position == 0:
            trade_type = 'close'
        else:
            trade_type = 'adjust'
        
        trade = FITradeSnapshot(
            trade_type=trade_type,
            instrument=instrument,
            contracts=contracts,
            price=futures_price,
            notional=notional,
            dv01_impact=dv01_impact,
            transaction_cost=transaction_cost,
            reason='dv01_hedge',
        )
        
        return trade, transaction_cost, dv01_impact
    
    def _capture_market_state(
        self,
        portfolio: FIPortfolio,
        cumulative_rate: float,
        cumulative_short: float,
        cumulative_long: float,
    ) -> FIMarketState:
        """Capture current market state from portfolio."""
        # Get rate from first environment
        rate = 0.0
        for env in portfolio.pricing_environments.values():
            if env.rate_curve:
                rate = env.rate_curve.get_rate(1.0)
                break
        
        # For flat curve, short and long are same but we track changes
        short_rate = rate
        long_rate = rate
        spread = cumulative_long - cumulative_short
        
        return FIMarketState(
            rate=rate,
            rate_curve={},  # Could populate with term structure
            short_rate=short_rate,
            long_rate=long_rate,
            spread=spread,
        )
    
    def __repr__(self) -> str:
        return f"FIDynamicScenarioEngine(config={self.config})"
