"""
Results containers for FI dynamic scenario analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from quantark.dynamicscenario.base import BaseDayResult, BaseScenarioResults


@dataclass
class FIMarketState:
    """
    FI market state at a point in time.
    
    Attributes:
        rate: Current rate level (flat rate or short-term rate)
        rate_curve: Rate curve by tenor (if available)
        short_rate: Short-end rate
        long_rate: Long-end rate
        spread: Long - short rate spread
    """
    rate: float = 0.0
    rate_curve: Dict[float, float] = field(default_factory=dict)  # tenor -> rate
    short_rate: float = 0.0
    long_rate: float = 0.0
    spread: float = 0.0
    
    def __post_init__(self):
        """Calculate spread if not set."""
        if self.spread == 0.0 and self.long_rate != 0 and self.short_rate != 0:
            self.spread = self.long_rate - self.short_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rate': self.rate,
            'rate_curve': self.rate_curve,
            'short_rate': self.short_rate,
            'long_rate': self.long_rate,
            'spread': self.spread,
        }


@dataclass
class FITradeSnapshot:
    """
    Snapshot of an FI hedge trade executed during simulation.
    
    Attributes:
        trade_type: Type of trade ('open', 'adjust', 'close')
        instrument: Hedge instrument (e.g., 'TY_futures', 'ZN_futures')
        contracts: Number of contracts traded
        price: Execution price
        notional: Trade notional value
        dv01_impact: DV01 impact of this trade
        transaction_cost: Cost of transaction
        reason: Reason for trade (e.g., 'dv01_hedge')
    """
    trade_type: str
    instrument: str
    contracts: float
    price: float
    notional: float
    dv01_impact: float
    transaction_cost: float
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_type': self.trade_type,
            'instrument': self.instrument,
            'contracts': self.contracts,
            'price': self.price,
            'notional': self.notional,
            'dv01_impact': self.dv01_impact,
            'transaction_cost': self.transaction_cost,
            'reason': self.reason,
        }


@dataclass
class FIDayResult(BaseDayResult):
    """
    Results for a single day in an FI dynamic scenario.
    
    Extends BaseDayResult with FI-specific metrics like DV01, convexity,
    and duration.
    
    Attributes:
        dv01_pre_hedge: Portfolio DV01 before hedging
        dv01_post_hedge: Portfolio DV01 after hedging
        convexity: Portfolio convexity
        modified_duration: Portfolio modified duration
        key_rate_dv01: Key-rate DV01 by tenor (optional)
        market_state: FI market state (rate levels)
        hedge_positions: Current hedge positions (contracts by instrument)
        trades: List of hedge trades executed today
    """
    # FI-specific risk metrics
    dv01_pre_hedge: float = 0.0
    dv01_post_hedge: float = 0.0
    convexity: float = 0.0
    modified_duration: float = 0.0
    key_rate_dv01: Dict[float, float] = field(default_factory=dict)  # tenor -> DV01
    
    # Market state
    market_state: Optional[FIMarketState] = None
    
    # Hedge state
    hedge_positions: Dict[str, float] = field(default_factory=dict)  # instrument -> contracts
    trades: List[FITradeSnapshot] = field(default_factory=list)
    
    @property
    def dv01(self) -> float:
        """Get post-hedge DV01 (or pre-hedge if no hedging)."""
        return self.dv01_post_hedge if self.dv01_post_hedge != 0 else self.dv01_pre_hedge
    
    @property
    def num_trades(self) -> int:
        """Get number of trades executed today."""
        return len(self.trades)
    
    @property
    def total_hedge_dv01(self) -> float:
        """Get total DV01 from hedge positions."""
        return sum(t.dv01_impact for t in self.trades)
    
    def get_key_rate_dv01(self) -> Dict[float, float]:
        """
        Get key-rate DV01 breakdown.
        
        Returns:
            Dictionary mapping tenors to DV01 contributions
        """
        return self.key_rate_dv01
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'dv01_pre_hedge': self.dv01_pre_hedge,
            'dv01_post_hedge': self.dv01_post_hedge,
            'dv01': self.dv01,
            'convexity': self.convexity,
            'modified_duration': self.modified_duration,
            'key_rate_dv01': self.key_rate_dv01,
            'market_state': self.market_state.to_dict() if self.market_state else None,
            'hedge_positions': self.hedge_positions,
            'trades': [t.to_dict() for t in self.trades],
        })
        return base_dict
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        date_str = f" ({self.date.date()})" if self.date else ""
        label_str = f" - {self.label}" if self.label else ""
        
        lines = [
            f"Day {self.day_index}{date_str}{label_str}",
            f"  Portfolio Value: ${self.portfolio_value:,.2f}",
            f"  Daily P&L: ${self.daily_pnl:,.2f}",
            f"  Cumulative P&L: ${self.cumulative_pnl:,.2f}",
            f"  Net P&L (after costs): ${self.net_pnl:,.2f}",
            f"  DV01: ${self.dv01:,.2f}",
            f"  Convexity: {self.convexity:,.4f}",
            f"  Duration: {self.modified_duration:.2f}",
        ]
        
        if self.dv01_pre_hedge != self.dv01_post_hedge:
            lines.append(f"  DV01 Pre-Hedge: ${self.dv01_pre_hedge:,.2f}")
            lines.append(f"  DV01 Post-Hedge: ${self.dv01_post_hedge:,.2f}")
        
        if self.market_state:
            lines.append(f"  Rate: {self.market_state.rate:.4f}")
            if self.market_state.spread != 0:
                lines.append(f"  Curve Spread: {self.market_state.spread:.4f}")
        
        if self.trades:
            lines.append(f"  Trades: {len(self.trades)}")
            lines.append(f"  Transaction Costs: ${self.transaction_costs_today:,.2f}")
        
        return "\n".join(lines)


@dataclass
class FIDynamicScenarioResults(BaseScenarioResults):
    """
    Complete results from an FI dynamic scenario analysis run.
    
    Contains the full day-by-day evolution of the FI portfolio through
    the scenario, with FI-specific analysis methods.
    
    Attributes:
        day_results: List of FIDayResult for each day
        baseline_dv01: Initial portfolio DV01
        baseline_duration: Initial portfolio duration
        final_dv01: Final portfolio DV01
        final_duration: Final portfolio duration
    """
    # Override day_results type
    day_results: List[FIDayResult] = field(default_factory=list)
    
    # FI-specific baseline/final metrics
    baseline_dv01: float = 0.0
    baseline_duration: float = 0.0
    final_dv01: float = 0.0
    final_duration: float = 0.0
    
    asset_class: str = "fixed_income"
    
    def __post_init__(self):
        """Calculate derived fields."""
        super().__post_init__()
        
        # Calculate baseline/final from day results
        if self.day_results:
            if self.baseline_dv01 == 0.0:
                self.baseline_dv01 = self.day_results[0].dv01_pre_hedge
            if self.baseline_duration == 0.0:
                self.baseline_duration = self.day_results[0].modified_duration
            if self.final_dv01 == 0.0:
                self.final_dv01 = self.day_results[-1].dv01
            if self.final_duration == 0.0:
                self.final_duration = self.day_results[-1].modified_duration
    
    def get_dv01_evolution(self) -> pd.DataFrame:
        """
        Get DV01 evolution over time.
        
        Returns:
            DataFrame with DV01 values at each day (pre-hedge and post-hedge)
        """
        data = []
        for result in self.day_results:
            data.append({
                'day_index': result.day_index,
                'date': result.date,
                'dv01_pre_hedge': result.dv01_pre_hedge,
                'dv01_post_hedge': result.dv01_post_hedge,
                'dv01': result.dv01,
            })
        return pd.DataFrame(data)
    
    def get_duration_evolution(self) -> pd.DataFrame:
        """
        Get duration evolution over time.
        
        Returns:
            DataFrame with modified duration at each day
        """
        data = []
        for result in self.day_results:
            data.append({
                'day_index': result.day_index,
                'date': result.date,
                'modified_duration': result.modified_duration,
            })
        return pd.DataFrame(data)
    
    def get_convexity_evolution(self) -> pd.DataFrame:
        """
        Get convexity evolution over time.
        
        Returns:
            DataFrame with convexity at each day
        """
        data = []
        for result in self.day_results:
            data.append({
                'day_index': result.day_index,
                'date': result.date,
                'convexity': result.convexity,
            })
        return pd.DataFrame(data)
    
    def get_rate_evolution(self) -> pd.DataFrame:
        """
        Get rate curve evolution over time.
        
        Returns:
            DataFrame with rate levels at each day
        """
        data = []
        for result in self.day_results:
            row = {
                'day_index': result.day_index,
                'date': result.date,
            }
            if result.market_state:
                row['rate'] = result.market_state.rate
                row['short_rate'] = result.market_state.short_rate
                row['long_rate'] = result.market_state.long_rate
                row['spread'] = result.market_state.spread
            data.append(row)
        return pd.DataFrame(data)
    
    def get_key_rate_dv01_evolution(self) -> pd.DataFrame:
        """
        Get key-rate DV01 evolution over time.
        
        Returns:
            DataFrame with key-rate DV01 by tenor at each day
        """
        data = []
        for result in self.day_results:
            row = {
                'day_index': result.day_index,
                'date': result.date,
            }
            for tenor, dv01 in result.key_rate_dv01.items():
                row[f'kr_dv01_{tenor}y'] = dv01
            data.append(row)
        return pd.DataFrame(data)
    
    def get_hedge_trades(self) -> pd.DataFrame:
        """
        Get all hedge trades as DataFrame.
        
        Returns:
            DataFrame with all hedge trades
        """
        data = []
        for result in self.day_results:
            for trade in result.trades:
                data.append({
                    'day_index': result.day_index,
                    'date': result.date,
                    **trade.to_dict(),
                })
        return pd.DataFrame(data) if data else pd.DataFrame()
    
    def get_hedge_effectiveness(self) -> Dict[str, float]:
        """
        Calculate hedge effectiveness metrics.
        
        Returns:
            Dictionary with hedge effectiveness statistics
        """
        if not self.day_results:
            return {}
        
        # Calculate DV01 tracking error
        dv01_values = [r.dv01_post_hedge for r in self.day_results]
        dv01_mean = sum(dv01_values) / len(dv01_values)
        dv01_std = (sum((v - dv01_mean)**2 for v in dv01_values) / len(dv01_values)) ** 0.5
        
        # Count hedge trades
        total_trades = sum(r.num_trades for r in self.day_results)
        hedge_days = sum(1 for r in self.day_results if r.num_trades > 0)
        
        return {
            'dv01_mean': dv01_mean,
            'dv01_std': dv01_std,
            'dv01_tracking_error': dv01_std,
            'total_hedge_trades': total_trades,
            'hedge_days': hedge_days,
            'hedge_frequency': hedge_days / len(self.day_results) if self.day_results else 0,
        }
    
    def to_summary_dataframe(self) -> pd.DataFrame:
        """
        Create summary DataFrame with day-by-day results.
        
        Returns:
            DataFrame with one row per day
        """
        rows = []
        
        for result in self.day_results:
            row = {
                'day_index': result.day_index,
                'date': result.date,
                'label': result.label,
                'portfolio_value': result.portfolio_value,
                'daily_pnl': result.daily_pnl,
                'cumulative_pnl': result.cumulative_pnl,
                'net_pnl': result.net_pnl,
                'transaction_costs': result.cumulative_transaction_costs,
                'num_trades': result.num_trades,
                'dv01': result.dv01,
                'dv01_pre_hedge': result.dv01_pre_hedge,
                'dv01_post_hedge': result.dv01_post_hedge,
                'convexity': result.convexity,
                'modified_duration': result.modified_duration,
            }
            
            # Add market state if available
            if result.market_state:
                row['rate'] = result.market_state.rate
                row['short_rate'] = result.market_state.short_rate
                row['long_rate'] = result.market_state.long_rate
                row['spread'] = result.market_state.spread
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_summary(self) -> str:
        """Get human-readable summary of results."""
        lines = [
            "=" * 60,
            "FI DYNAMIC SCENARIO ANALYSIS RESULTS",
            "=" * 60,
            f"Path: {self.path_name}",
            f"Execution Time: {self.execution_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {self.total_execution_time:.2f} seconds",
            f"Number of Days: {self.num_days}",
            "",
            "-" * 60,
            "PORTFOLIO PERFORMANCE",
            "-" * 60,
            f"Initial Value: ${self.baseline_value:,.2f}",
            f"Final Value: ${self.final_value:,.2f}",
            f"Total P&L: ${self.total_pnl:,.2f} ({self.total_pnl_pct:+.2f}%)",
            f"Transaction Costs: ${self.total_transaction_costs:,.2f}",
            f"Net P&L: ${self.net_pnl:,.2f}",
            f"Total Hedge Trades: {self.total_hedges}",
            "",
            "-" * 60,
            "FI RISK MEASURES",
            "-" * 60,
            f"Initial DV01: ${self.baseline_dv01:,.2f}",
            f"Final DV01: ${self.final_dv01:,.2f}",
            f"Initial Duration: {self.baseline_duration:.2f}",
            f"Final Duration: {self.final_duration:.2f}",
        ]
        
        # Hedge effectiveness
        if self.total_hedges > 0:
            effectiveness = self.get_hedge_effectiveness()
            lines.extend([
                "",
                "-" * 60,
                "HEDGE EFFECTIVENESS",
                "-" * 60,
                f"DV01 Mean: ${effectiveness.get('dv01_mean', 0):,.2f}",
                f"DV01 Tracking Error: ${effectiveness.get('dv01_tracking_error', 0):,.2f}",
                f"Hedge Frequency: {effectiveness.get('hedge_frequency', 0):.1%}",
            ])
        
        # Drawdown
        dd_amt, dd_pct, peak_day, trough_day = self.get_max_drawdown()
        lines.extend([
            "",
            "-" * 60,
            "RISK METRICS",
            "-" * 60,
            f"Max Drawdown: ${dd_amt:,.2f} ({dd_pct:.2f}%)",
            f"  Peak Day: {peak_day}, Trough Day: {trough_day}",
        ])
        
        # Best/worst days
        best = self.get_best_day()
        worst = self.get_worst_day()
        if best and worst:
            lines.extend([
                f"Best Day: Day {best.day_index} (${best.daily_pnl:+,.2f})",
                f"Worst Day: Day {worst.day_index} (${worst.daily_pnl:+,.2f})",
            ])
        
        lines.append("")
        lines.append("-" * 60)
        lines.append("DAY-BY-DAY SUMMARY")
        lines.append("-" * 60)
        
        for result in self.day_results:
            date_str = f" ({result.date.date()})" if result.date else ""
            label_str = f" [{result.label}]" if result.label else ""
            trades_str = f", {result.num_trades} trades" if result.num_trades > 0 else ""
            lines.append(
                f"Day {result.day_index}{date_str}{label_str}: "
                f"Value=${result.portfolio_value:,.2f}, "
                f"P&L=${result.daily_pnl:+,.2f}, "
                f"DV01=${result.dv01:,.0f}{trades_str}"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'baseline_dv01': self.baseline_dv01,
            'baseline_duration': self.baseline_duration,
            'final_dv01': self.final_dv01,
            'final_duration': self.final_duration,
        })
        return base_dict
    
    def __repr__(self) -> str:
        return (
            f"FIDynamicScenarioResults("
            f"path='{self.path_name}', "
            f"days={self.num_days}, "
            f"pnl=${self.total_pnl:,.2f}, "
            f"dv01=${self.final_dv01:,.0f})"
        )

