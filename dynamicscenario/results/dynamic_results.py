"""
Results containers for dynamic scenario analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd


@dataclass
class PositionSnapshot:
    """
    Snapshot of a single position at a point in time.
    
    Attributes:
        position_id: Position identifier
        underlying: Underlying asset
        product_type: Type of product (e.g., 'EuropeanOption', 'SpotInstrument')
        quantity: Current quantity
        market_value: Current market value
        entry_price: Original entry price
        pnl: Unrealized P&L
        greeks: Optional Greeks for this position
    """
    position_id: str
    underlying: str
    product_type: str
    quantity: float
    market_value: float
    entry_price: float
    pnl: float
    greeks: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position_id': self.position_id,
            'underlying': self.underlying,
            'product_type': self.product_type,
            'quantity': self.quantity,
            'market_value': self.market_value,
            'entry_price': self.entry_price,
            'pnl': self.pnl,
            'greeks': self.greeks,
        }


@dataclass
class TradeSnapshot:
    """
    Snapshot of a trade executed during simulation.
    
    Attributes:
        trade_type: Type of trade ('open', 'adjust', 'close', 'hedge')
        underlying: Underlying asset
        instrument_type: Instrument type (e.g., 'spot', 'futures')
        quantity: Trade quantity
        price: Execution price
        notional: Trade notional value
        transaction_cost: Cost of transaction
        reason: Reason for trade
    """
    trade_type: str
    underlying: str
    instrument_type: str
    quantity: float
    price: float
    notional: float
    transaction_cost: float
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_type': self.trade_type,
            'underlying': self.underlying,
            'instrument_type': self.instrument_type,
            'quantity': self.quantity,
            'price': self.price,
            'notional': self.notional,
            'transaction_cost': self.transaction_cost,
            'reason': self.reason,
        }


@dataclass
class MarketState:
    """
    Market state at a point in time.
    
    Attributes:
        spot: Spot price (by underlying)
        volatility: Implied volatility (by underlying)
        rate: Risk-free rate
        div_yield: Dividend yield (by underlying)
        basis_yield: Basis yield (by underlying)
    """
    spot: Dict[str, float] = field(default_factory=dict)
    volatility: Dict[str, float] = field(default_factory=dict)
    rate: float = 0.0
    div_yield: Dict[str, float] = field(default_factory=dict)
    basis_yield: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'spot': self.spot,
            'volatility': self.volatility,
            'rate': self.rate,
            'div_yield': self.div_yield,
            'basis_yield': self.basis_yield,
        }


@dataclass
class DayResult:
    """
    Results for a single day in the dynamic scenario.
    
    Contains all state information for one day including portfolio value,
    P&L, Greeks, positions, trades, and market state.
    
    Attributes:
        day_index: 0-based day index
        date: Optional date (if start_date was provided)
        label: Optional day label
        portfolio_value: Total portfolio value
        daily_pnl: P&L change from previous day
        cumulative_pnl: Total P&L from start
        transaction_costs_today: Transaction costs incurred today
        cumulative_transaction_costs: Total transaction costs
        net_pnl: Cumulative P&L after transaction costs
        greeks: Portfolio-level Greeks
        positions: List of position snapshots
        trades: List of trades executed today
        market_state: Market state after day's changes
        metadata: Additional data
    """
    day_index: int
    date: Optional[datetime] = None
    label: Optional[str] = None
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    transaction_costs_today: float = 0.0
    cumulative_transaction_costs: float = 0.0
    net_pnl: float = 0.0
    greeks: Dict[str, float] = field(default_factory=dict)
    positions: List[PositionSnapshot] = field(default_factory=list)
    trades: List[TradeSnapshot] = field(default_factory=list)
    market_state: Optional[MarketState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_trades(self) -> int:
        """Get number of trades executed today."""
        return len(self.trades)
    
    @property
    def num_positions(self) -> int:
        """Get number of positions."""
        return len(self.positions)
    
    @property
    def delta(self) -> float:
        """Get portfolio delta."""
        return self.greeks.get('delta', 0.0)
    
    @property
    def gamma(self) -> float:
        """Get portfolio gamma."""
        return self.greeks.get('gamma', 0.0)
    
    @property
    def vega(self) -> float:
        """Get portfolio vega."""
        return self.greeks.get('vega', 0.0)
    
    @property
    def theta(self) -> float:
        """Get portfolio theta."""
        return self.greeks.get('theta', 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'day_index': self.day_index,
            'date': self.date.isoformat() if self.date else None,
            'label': self.label,
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'cumulative_pnl': self.cumulative_pnl,
            'transaction_costs_today': self.transaction_costs_today,
            'cumulative_transaction_costs': self.cumulative_transaction_costs,
            'net_pnl': self.net_pnl,
            'greeks': self.greeks,
            'positions': [p.to_dict() for p in self.positions],
            'trades': [t.to_dict() for t in self.trades],
            'market_state': self.market_state.to_dict() if self.market_state else None,
            'metadata': self.metadata,
        }
    
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
        ]
        
        if self.greeks:
            lines.append(f"  Delta: {self.greeks.get('delta', 0):.2f}")
            lines.append(f"  Gamma: {self.greeks.get('gamma', 0):.4f}")
            lines.append(f"  Vega: {self.greeks.get('vega', 0):.2f}")
        
        if self.trades:
            lines.append(f"  Trades: {len(self.trades)}")
            lines.append(f"  Transaction Costs: ${self.transaction_costs_today:,.2f}")
        
        return "\n".join(lines)


@dataclass
class DynamicScenarioResults:
    """
    Complete results from a dynamic scenario analysis run.
    
    Contains the full day-by-day evolution of the portfolio through
    the scenario, including all intermediate states.
    
    Attributes:
        path_name: Name of the day path used
        baseline_value: Initial portfolio value
        final_value: Final portfolio value
        day_results: List of results for each day
        total_pnl: Total P&L over scenario
        total_pnl_pct: Total P&L as percentage
        total_transaction_costs: Total transaction costs
        net_pnl: Net P&L after transaction costs
        total_hedges: Number of hedge trades executed
        execution_timestamp: When simulation was run
        total_execution_time: Time taken (seconds)
        config_summary: Configuration used
        metadata: Additional data
    """
    path_name: str
    baseline_value: float
    final_value: float
    day_results: List[DayResult]
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    total_transaction_costs: float = 0.0
    net_pnl: float = 0.0
    total_hedges: int = 0
    execution_timestamp: datetime = field(default_factory=datetime.now)
    total_execution_time: float = 0.0
    config_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields if not set."""
        if self.total_pnl == 0.0 and self.day_results:
            self.total_pnl = self.final_value - self.baseline_value
        
        if self.total_pnl_pct == 0.0 and self.baseline_value != 0:
            self.total_pnl_pct = (self.total_pnl / self.baseline_value) * 100
        
        if self.net_pnl == 0.0:
            self.net_pnl = self.total_pnl - self.total_transaction_costs
    
    @property
    def num_days(self) -> int:
        """Get number of days in scenario."""
        return len(self.day_results)
    
    def get_day_result(self, day_index: int) -> Optional[DayResult]:
        """Get result for specific day."""
        if 0 <= day_index < len(self.day_results):
            return self.day_results[day_index]
        return None
    
    def get_worst_day(self) -> Optional[DayResult]:
        """Get day with worst P&L."""
        if not self.day_results:
            return None
        return min(self.day_results, key=lambda d: d.daily_pnl)
    
    def get_best_day(self) -> Optional[DayResult]:
        """Get day with best P&L."""
        if not self.day_results:
            return None
        return max(self.day_results, key=lambda d: d.daily_pnl)
    
    def get_max_drawdown(self) -> tuple:
        """
        Calculate maximum drawdown.
        
        Returns:
            Tuple of (max_drawdown_amount, max_drawdown_pct, peak_day, trough_day)
        """
        if not self.day_results:
            return 0.0, 0.0, None, None
        
        peak_value = self.baseline_value
        peak_day = 0
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        trough_day = 0
        
        for result in self.day_results:
            if result.portfolio_value > peak_value:
                peak_value = result.portfolio_value
                peak_day = result.day_index
            
            drawdown = peak_value - result.portfolio_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = (drawdown / peak_value) * 100 if peak_value > 0 else 0
                trough_day = result.day_index
        
        return max_drawdown, max_drawdown_pct, peak_day, trough_day
    
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
                'delta': result.delta,
                'gamma': result.gamma,
                'vega': result.vega,
                'theta': result.theta,
            }
            
            # Add market state if available
            if result.market_state:
                for underlying, spot in result.market_state.spot.items():
                    row[f'spot_{underlying}'] = spot
                for underlying, vol in result.market_state.volatility.items():
                    row[f'vol_{underlying}'] = vol
                row['rate'] = result.market_state.rate
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_pnl_evolution(self) -> pd.DataFrame:
        """
        Get P&L evolution over time.
        
        Returns:
            DataFrame with columns: day_index, date, daily_pnl, cumulative_pnl, net_pnl
        """
        data = []
        for result in self.day_results:
            data.append({
                'day_index': result.day_index,
                'date': result.date,
                'portfolio_value': result.portfolio_value,
                'daily_pnl': result.daily_pnl,
                'cumulative_pnl': result.cumulative_pnl,
                'net_pnl': result.net_pnl,
            })
        return pd.DataFrame(data)
    
    def get_greeks_evolution(self) -> pd.DataFrame:
        """
        Get Greeks evolution over time.
        
        Returns:
            DataFrame with columns for each Greek over days
        """
        data = []
        for result in self.day_results:
            row = {
                'day_index': result.day_index,
                'date': result.date,
            }
            row.update(result.greeks)
            data.append(row)
        return pd.DataFrame(data)
    
    def get_position_evolution(self, position_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get position evolution over time.
        
        Args:
            position_id: Specific position to track (None = all positions)
            
        Returns:
            DataFrame with position data over days
        """
        data = []
        for result in self.day_results:
            for pos in result.positions:
                if position_id is None or pos.position_id == position_id:
                    data.append({
                        'day_index': result.day_index,
                        'date': result.date,
                        'position_id': pos.position_id,
                        'underlying': pos.underlying,
                        'product_type': pos.product_type,
                        'quantity': pos.quantity,
                        'market_value': pos.market_value,
                        'pnl': pos.pnl,
                    })
        return pd.DataFrame(data)
    
    def get_market_evolution(self, underlying: Optional[str] = None) -> pd.DataFrame:
        """
        Get market parameter evolution over time.
        
        Args:
            underlying: Specific underlying (None = first available)
            
        Returns:
            DataFrame with market parameters over days
        """
        data = []
        for result in self.day_results:
            if result.market_state:
                row = {
                    'day_index': result.day_index,
                    'date': result.date,
                }
                
                # Get underlying to use
                if underlying:
                    if underlying in result.market_state.spot:
                        row['spot'] = result.market_state.spot[underlying]
                    if underlying in result.market_state.volatility:
                        row['volatility'] = result.market_state.volatility[underlying]
                    if underlying in result.market_state.div_yield:
                        row['div_yield'] = result.market_state.div_yield[underlying]
                    if underlying in result.market_state.basis_yield:
                        row['basis_yield'] = result.market_state.basis_yield[underlying]
                else:
                    # Use first underlying
                    if result.market_state.spot:
                        first_und = list(result.market_state.spot.keys())[0]
                        row['spot'] = result.market_state.spot[first_und]
                        row['volatility'] = result.market_state.volatility.get(first_und, 0)
                        row['div_yield'] = result.market_state.div_yield.get(first_und, 0)
                        row['basis_yield'] = result.market_state.basis_yield.get(first_und, 0)
                
                row['rate'] = result.market_state.rate
                data.append(row)
        
        return pd.DataFrame(data)
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Get all trades as DataFrame.
        
        Returns:
            DataFrame with all trades
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
    
    def get_summary(self) -> str:
        """Get human-readable summary of results."""
        lines = [
            "=" * 60,
            "DYNAMIC SCENARIO ANALYSIS RESULTS",
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
        ]
        
        # Drawdown
        dd_amt, dd_pct, peak_day, trough_day = self.get_max_drawdown()
        lines.extend([
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
                f"Delta={result.delta:.1f}{trades_str}"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'path_name': self.path_name,
            'baseline_value': self.baseline_value,
            'final_value': self.final_value,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'total_transaction_costs': self.total_transaction_costs,
            'net_pnl': self.net_pnl,
            'total_hedges': self.total_hedges,
            'num_days': self.num_days,
            'execution_timestamp': self.execution_timestamp.isoformat(),
            'total_execution_time': self.total_execution_time,
            'config_summary': self.config_summary,
            'day_results': [d.to_dict() for d in self.day_results],
            'metadata': self.metadata,
        }
    
    def __repr__(self) -> str:
        return (
            f"DynamicScenarioResults("
            f"path='{self.path_name}', "
            f"days={self.num_days}, "
            f"pnl=${self.total_pnl:,.2f})"
        )
