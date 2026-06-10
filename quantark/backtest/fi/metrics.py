"""
Fixed Income performance metrics calculation.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


class FIPerformanceMetrics:
    """
    Calculate performance metrics for FI backtest results.
    
    Provides comprehensive metrics including:
    - P&L metrics (total, Sharpe, max drawdown)
    - DV01 hedging metrics (tracking error, average exposure)
    - Duration and convexity evolution
    - Risk metrics (VaR, CVaR)
    """
    
    def __init__(self, results: 'FIBacktestResults'):
        """Initialize metrics calculator."""
        self.results = results
        self._returns_series: Optional[pd.Series] = None
    
    @property
    def returns_series(self) -> pd.Series:
        """Get returns time series (cached)."""
        if self._returns_series is None:
            value_series = self.results.get_value_series()
            if len(value_series) > 1:
                self._returns_series = value_series.pct_change().dropna()
            else:
                self._returns_series = pd.Series()
        return self._returns_series
    
    # =========================================================================
    # P&L Metrics
    # =========================================================================
    
    def total_pnl(self) -> float:
        """Total P&L."""
        return self.results.get_total_pnl()
    
    def total_return(self) -> float:
        """Total return as decimal."""
        return self.results.get_total_return()
    
    def sharpe_ratio(self, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
        """Calculate Sharpe ratio."""
        if len(self.returns_series) < 2:
            return 0.0
        
        excess_returns = self.returns_series - (risk_free_rate / annualization_factor)
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(annualization_factor)
        return sharpe
    
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        value_series = self.results.get_value_series()
        if len(value_series) < 2:
            return 0.0
        
        cumulative_max = value_series.expanding().max()
        drawdown = (value_series - cumulative_max) / cumulative_max
        max_dd = abs(drawdown.min())
        
        return max_dd
    
    def win_rate(self) -> float:
        """Calculate win rate."""
        if len(self.returns_series) == 0:
            return 0.0
        
        winning_periods = (self.returns_series > 0).sum()
        total_periods = len(self.returns_series)
        
        return winning_periods / total_periods
    
    # =========================================================================
    # DV01 Hedging Metrics
    # =========================================================================
    
    def dv01_tracking_error(self) -> float:
        """
        Calculate DV01 tracking error (RMSE from target).
        
        Returns:
            Root mean squared error of DV01 vs target
        """
        dv01_series = self.results.get_dv01_series()
        if len(dv01_series) == 0:
            return 0.0
        
        target_dv01 = self.results.config.strategy.target_dv01
        tracking_error = np.sqrt(((dv01_series - target_dv01) ** 2).mean())
        
        return tracking_error
    
    def average_absolute_dv01(self) -> float:
        """Calculate average absolute DV01 exposure."""
        dv01_series = self.results.get_dv01_series()
        if len(dv01_series) == 0:
            return 0.0
        
        return abs(dv01_series).mean()
    
    def max_dv01_exposure(self) -> float:
        """Calculate maximum absolute DV01 exposure."""
        dv01_series = self.results.get_dv01_series()
        if len(dv01_series) == 0:
            return 0.0
        
        return abs(dv01_series).max()
    
    def dv01_hedge_effectiveness(self) -> float:
        """
        Calculate DV01 hedge effectiveness ratio.
        
        Compares actual DV01 variance to unhedged (if available).
        Returns value between 0 and 1 (higher = more effective).
        """
        dv01_series = self.results.get_dv01_series()
        if len(dv01_series) < 2:
            return 0.0
        
        # Estimate unhedged DV01 variance from first observation
        initial_dv01 = abs(dv01_series.iloc[0]) if len(dv01_series) > 0 else 0
        if initial_dv01 == 0:
            return 0.0
        
        actual_variance = dv01_series.var()
        unhedged_estimate = initial_dv01 ** 2
        
        effectiveness = 1 - (actual_variance / unhedged_estimate)
        return max(0, min(1, effectiveness))
    
    # =========================================================================
    # Duration Metrics
    # =========================================================================
    
    def average_duration(self) -> float:
        """Calculate average portfolio modified duration."""
        duration_series = self.results.get_duration_series()
        if len(duration_series) == 0:
            return 0.0
        
        return duration_series.mean()
    
    def duration_contribution_to_pnl(self) -> float:
        """
        Estimate P&L contribution from duration exposure.
        
        Approximates how much of P&L came from rate movements.
        """
        dv01_series = self.results.get_dv01_series()
        pnl_series = self.results.get_pnl_series()
        
        if len(dv01_series) == 0 or len(pnl_series) == 0:
            return 0.0
        
        # Simplified: correlate DV01 changes with P&L changes
        if len(dv01_series) > 1:
            dv01_changes = dv01_series.diff().dropna()
            pnl_changes = pnl_series.diff().dropna()
            
            if len(dv01_changes) == len(pnl_changes) and len(dv01_changes) > 0:
                correlation = np.corrcoef(dv01_changes, pnl_changes)[0, 1]
                if not np.isnan(correlation):
                    return correlation
        
        return 0.0
    
    # =========================================================================
    # Hedging Metrics
    # =========================================================================
    
    def hedge_frequency(self) -> float:
        """Calculate average hedge frequency (hedges per day)."""
        if self.results.num_hedges == 0:
            return 0.0
        
        states_df = self.results.states_df
        if len(states_df) == 0:
            return 0.0
        
        time_span = (states_df.index[-1] - states_df.index[0]).days
        if time_span == 0:
            return 0.0
        
        return self.results.num_hedges / time_span
    
    def average_hedge_cost(self) -> float:
        """Calculate average transaction cost per hedge."""
        if self.results.num_hedges == 0:
            return 0.0
        
        return self.results.total_transaction_costs / self.results.num_hedges
    
    def average_hedge_size(self) -> float:
        """Calculate average hedge size (contracts)."""
        trades_df = self.results.get_hedge_trades()
        if len(trades_df) == 0:
            return 0.0
        
        return abs(trades_df['quantity']).mean()
    
    def total_hedge_cost_ratio(self) -> float:
        """Calculate transaction costs as % of initial value."""
        if self.results.initial_value == 0:
            return 0.0
        
        return self.results.total_transaction_costs / self.results.initial_value
    
    # =========================================================================
    # Risk Metrics
    # =========================================================================
    
    def value_at_risk(self, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if len(self.returns_series) < 2:
            return 0.0
        
        var = abs(self.returns_series.quantile(1 - confidence_level))
        return var
    
    def conditional_var(self, confidence_level: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(self.returns_series) < 2:
            return 0.0
        
        var = self.value_at_risk(confidence_level)
        cvar = abs(self.returns_series[self.returns_series <= -var].mean())
        
        return cvar if not np.isnan(cvar) else 0.0
    
    def volatility(self, annualization_factor: int = 252) -> float:
        """Calculate annualized volatility."""
        if len(self.returns_series) < 2:
            return 0.0
        
        return self.returns_series.std() * np.sqrt(annualization_factor)
    
    # =========================================================================
    # Comprehensive Report
    # =========================================================================
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all available metrics."""
        metrics = {
            # P&L Metrics
            'total_pnl': self.total_pnl(),
            'total_return': self.total_return(),
            'total_return_pct': self.total_return() * 100,
            'sharpe_ratio': self.sharpe_ratio(),
            'max_drawdown': self.max_drawdown(),
            'max_drawdown_pct': self.max_drawdown() * 100,
            'win_rate': self.win_rate(),
            
            # DV01 Hedging Metrics
            'dv01_tracking_error': self.dv01_tracking_error(),
            'avg_abs_dv01': self.average_absolute_dv01(),
            'max_dv01_exposure': self.max_dv01_exposure(),
            'dv01_hedge_effectiveness': self.dv01_hedge_effectiveness(),
            
            # Duration Metrics
            'avg_duration': self.average_duration(),
            'duration_pnl_contribution': self.duration_contribution_to_pnl(),
            
            # Hedging Metrics
            'num_hedges': self.results.num_hedges,
            'hedge_frequency': self.hedge_frequency(),
            'avg_hedge_cost': self.average_hedge_cost(),
            'avg_hedge_size': self.average_hedge_size(),
            'total_hedge_costs': self.results.total_transaction_costs,
            'hedge_cost_ratio': self.total_hedge_cost_ratio(),
            
            # Risk Metrics
            'volatility': self.volatility(),
            'var_95': self.value_at_risk(0.95),
            'cvar_95': self.conditional_var(0.95),
            
            # Basic Info
            'initial_value': self.results.initial_value,
            'final_value': self.results.final_value,
            'num_timesteps': len(self.results.state_tracker),
        }
        
        return metrics
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all metrics to DataFrame."""
        metrics = self.calculate_all_metrics()
        df = pd.DataFrame([metrics]).T
        df.columns = ['Value']
        return df
    
    def __repr__(self) -> str:
        return (
            f"FIPerformanceMetrics("
            f"Sharpe={self.sharpe_ratio():.2f}, "
            f"DV01 Error=${self.dv01_tracking_error():,.0f})"
        )

