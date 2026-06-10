"""
Fixed Income backtest results container and analysis.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
from .state import FIStateTracker
from .config import FIBacktestConfig


class FIBacktestResults:
    """
    Container for Fixed Income backtest results and analysis.
    
    Provides access to:
    - Time series of portfolio states
    - Trade history
    - FI-specific performance metrics (DV01 tracking, duration)
    - Configuration details
    
    Attributes:
        config: FI backtest configuration
        state_tracker: State history tracker
        initial_value: Initial portfolio value
        final_value: Final portfolio value
        num_hedges: Number of hedges executed
        total_transaction_costs: Total costs incurred
    """
    
    def __init__(
        self,
        config: FIBacktestConfig,
        state_tracker: FIStateTracker,
        initial_value: float,
        final_value: float,
        num_hedges: int,
        total_transaction_costs: float
    ):
        """Initialize FI backtest results."""
        self.config = config
        self.state_tracker = state_tracker
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_hedges = num_hedges
        self.total_transaction_costs = total_transaction_costs
        
        # Lazy-loaded
        self._states_df: Optional[pd.DataFrame] = None
        self._trades_df: Optional[pd.DataFrame] = None
        self._metrics: Optional['FIPerformanceMetrics'] = None
    
    @property
    def states_df(self) -> pd.DataFrame:
        """Get states as DataFrame (cached)."""
        if self._states_df is None:
            self._states_df = self.state_tracker.to_dataframe()
        return self._states_df
    
    @property
    def trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame (cached)."""
        if self._trades_df is None:
            self._trades_df = self.state_tracker.get_trades_dataframe()
        return self._trades_df
    
    @property
    def metrics(self) -> 'FIPerformanceMetrics':
        """Get performance metrics (cached)."""
        if self._metrics is None:
            from .metrics import FIPerformanceMetrics
            self._metrics = FIPerformanceMetrics(self)
        return self._metrics
    
    def get_total_pnl(self) -> float:
        """Get total P&L."""
        return self.final_value - self.initial_value
    
    def get_total_return(self) -> float:
        """Get total return as decimal."""
        if self.initial_value == 0:
            return 0.0
        return (self.final_value - self.initial_value) / self.initial_value
    
    def get_pnl_series(self) -> pd.Series:
        """Get P&L time series."""
        if 'pnl' in self.states_df.columns:
            return self.states_df['pnl']
        return pd.Series()
    
    def get_value_series(self) -> pd.Series:
        """Get portfolio value time series."""
        if 'portfolio_value' in self.states_df.columns:
            return self.states_df['portfolio_value']
        return pd.Series()
    
    def get_dv01_series(self) -> pd.Series:
        """Get DV01 time series."""
        if 'risk_dv01' in self.states_df.columns:
            return self.states_df['risk_dv01']
        return pd.Series()
    
    def get_duration_series(self) -> pd.Series:
        """Get modified duration time series."""
        if 'risk_modified_duration' in self.states_df.columns:
            return self.states_df['risk_modified_duration']
        return pd.Series()
    
    def get_convexity_series(self) -> pd.Series:
        """Get convexity time series."""
        if 'risk_convexity' in self.states_df.columns:
            return self.states_df['risk_convexity']
        return pd.Series()
    
    def get_risk_measures_series(self) -> pd.DataFrame:
        """Get all risk measures time series."""
        risk_cols = [col for col in self.states_df.columns if col.startswith('risk_')]
        if risk_cols:
            return self.states_df[risk_cols]
        return pd.DataFrame()
    
    def get_market_data_series(self) -> pd.DataFrame:
        """Get market data time series."""
        market_cols = [col for col in self.states_df.columns if col.startswith('market_')]
        if market_cols:
            return self.states_df[market_cols]
        return pd.DataFrame()
    
    def get_hedge_trades(self) -> pd.DataFrame:
        """Get only hedge trades."""
        if len(self.trades_df) == 0:
            return pd.DataFrame()
        return self.trades_df[self.trades_df['trade_type'].isin(['open', 'adjust', 'close'])]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of results."""
        total_pnl = self.get_total_pnl()
        total_return = self.get_total_return()
        
        summary = {
            'backtest_name': f"{self.config.underlying}_{self.config.strategy.name}",
            'asset_class': 'fixed_income',
            'start_date': self.config.start_date.isoformat(),
            'end_date': self.config.end_date.isoformat(),
            'num_timesteps': len(self.state_tracker),
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_hedges': self.num_hedges,
            'total_transaction_costs': self.total_transaction_costs,
            'avg_cost_per_hedge': self.total_transaction_costs / self.num_hedges if self.num_hedges > 0 else 0,
            'strategy': self.config.strategy.get_parameters(),
        }
        
        # Add final risk measures
        if len(self.state_tracker.states) > 0:
            final_state = self.state_tracker.states[-1]
            summary['final_dv01'] = final_state.risk_measures.get('dv01', 0)
            summary['final_duration'] = final_state.risk_measures.get('modified_duration', 0)
        
        return summary
    
    def export_to_excel(self, filepath: str):
        """Export results to Excel file."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            summary_df = pd.DataFrame([self.get_summary()]).T
            summary_df.columns = ['Value']
            summary_df.to_excel(writer, sheet_name='Summary')
            
            self.states_df.to_excel(writer, sheet_name='States')
            
            if len(self.trades_df) > 0:
                self.trades_df.to_excel(writer, sheet_name='Trades')
            
            risk_df = self.get_risk_measures_series()
            if len(risk_df) > 0:
                risk_df.to_excel(writer, sheet_name='RiskMeasures')
    
    def export_to_parquet(self, filepath: str):
        """Export results to Parquet file."""
        combined_df = self.states_df.copy()
        combined_df.attrs['metadata'] = self.get_summary()
        combined_df.to_parquet(filepath)
    
    def __repr__(self) -> str:
        return (
            f"FIBacktestResults("
            f"PnL=${self.get_total_pnl():,.2f}, "
            f"Return={self.get_total_return():.2%}, "
            f"Hedges={self.num_hedges})"
        )

