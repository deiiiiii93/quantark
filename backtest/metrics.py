"""
Performance metrics calculation for backtest results.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


class PerformanceMetrics:
    """
    Calculate performance metrics from backtest results.
    
    Provides comprehensive metrics including:
    - P&L metrics (total, Sharpe, max drawdown)
    - Hedging metrics (frequency, costs, tracking error)
    - Greeks evolution metrics
    - Risk metrics (VaR, CVaR)
    
    Attributes:
        results: BacktestResults instance
    """
    
    def __init__(self, results: 'BacktestResults'):
        """
        Initialize metrics calculator.
        
        Args:
            results: BacktestResults instance
        """
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
        """
        Calculate Sharpe ratio.
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
            annualization_factor: Days per year for annualization
            
        Returns:
            Sharpe ratio
        """
        if len(self.returns_series) < 2:
            return 0.0
        
        excess_returns = self.returns_series - (risk_free_rate / annualization_factor)
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(annualization_factor)
        return sharpe
    
    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Returns:
            Maximum drawdown as positive decimal (e.g., 0.15 = 15% drawdown)
        """
        value_series = self.results.get_value_series()
        if len(value_series) < 2:
            return 0.0
        
        cumulative_max = value_series.expanding().max()
        drawdown = (value_series - cumulative_max) / cumulative_max
        max_dd = abs(drawdown.min())
        
        return max_dd
    
    def max_drawdown_duration(self) -> Optional[pd.Timedelta]:
        """
        Calculate maximum drawdown duration.
        
        Returns:
            Maximum time spent in drawdown
        """
        value_series = self.results.get_value_series()
        if len(value_series) < 2:
            return None
        
        cumulative_max = value_series.expanding().max()
        is_in_drawdown = value_series < cumulative_max
        
        # Find consecutive drawdown periods
        drawdown_periods = []
        current_start = None
        
        for timestamp, in_dd in is_in_drawdown.items():
            if in_dd and current_start is None:
                current_start = timestamp
            elif not in_dd and current_start is not None:
                drawdown_periods.append(timestamp - current_start)
                current_start = None
        
        if len(drawdown_periods) == 0:
            return pd.Timedelta(0)
        
        return max(drawdown_periods)
    
    def win_rate(self) -> float:
        """
        Calculate win rate (proportion of positive return periods).
        
        Returns:
            Win rate as decimal (0-1)
        """
        if len(self.returns_series) == 0:
            return 0.0
        
        winning_periods = (self.returns_series > 0).sum()
        total_periods = len(self.returns_series)
        
        return winning_periods / total_periods
    
    def profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Returns:
            Profit factor
        """
        if len(self.returns_series) == 0:
            return 0.0
        
        gross_profit = self.returns_series[self.returns_series > 0].sum()
        gross_loss = abs(self.returns_series[self.returns_series < 0].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    # =========================================================================
    # Hedging Metrics
    # =========================================================================
    
    def hedge_frequency(self) -> float:
        """
        Calculate average hedge frequency (hedges per day).
        
        Returns:
            Hedges per day
        """
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
        """
        Calculate average transaction cost per hedge.
        
        Returns:
            Average cost per hedge
        """
        if self.results.num_hedges == 0:
            return 0.0
        
        return self.results.total_transaction_costs / self.results.num_hedges
    
    def total_hedge_cost_ratio(self) -> float:
        """
        Calculate transaction costs as percentage of initial value.
        
        Returns:
            Cost ratio as decimal
        """
        if self.results.initial_value == 0:
            return 0.0
        
        return self.results.total_transaction_costs / self.results.initial_value
    
    def delta_tracking_error(self) -> float:
        """
        Calculate delta tracking error (RMSE of delta vs target).
        
        Returns:
            Root mean squared error of delta
        """
        delta_series = self.results.get_delta_series()
        if len(delta_series) == 0:
            return 0.0
        
        target_delta = self.results.config.strategy.target_delta
        tracking_error = np.sqrt(((delta_series - target_delta) ** 2).mean())
        
        return tracking_error
    
    def average_absolute_delta(self) -> float:
        """
        Calculate average absolute delta.
        
        Returns:
            Mean of absolute delta values
        """
        delta_series = self.results.get_delta_series()
        if len(delta_series) == 0:
            return 0.0
        
        return abs(delta_series).mean()
    
    def delta_rebalance_efficiency(self) -> float:
        """
        Calculate rebalancing efficiency (delta reduction per hedge).
        
        Returns:
            Average delta reduction per hedge
        """
        if self.results.num_hedges == 0:
            return 0.0
        
        delta_series = self.results.get_delta_series()
        if len(delta_series) == 0:
            return 0.0
        
        # Calculate average absolute delta
        avg_abs_delta = abs(delta_series).mean()
        
        # Efficiency = how much delta is managed per hedge
        return avg_abs_delta / self.results.num_hedges if self.results.num_hedges > 0 else 0.0
    
    # =========================================================================
    # Risk Metrics
    # =========================================================================
    
    def value_at_risk(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR as positive value
        """
        if len(self.returns_series) < 2:
            return 0.0
        
        var = abs(self.returns_series.quantile(1 - confidence_level))
        return var
    
    def conditional_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional VaR (CVaR or Expected Shortfall).
        
        Args:
            confidence_level: Confidence level
            
        Returns:
            CVaR as positive value
        """
        if len(self.returns_series) < 2:
            return 0.0
        
        var = self.value_at_risk(confidence_level)
        cvar = abs(self.returns_series[self.returns_series <= -var].mean())
        
        return cvar if not np.isnan(cvar) else 0.0
    
    def volatility(self, annualization_factor: int = 252) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            annualization_factor: Days per year
            
        Returns:
            Annualized volatility
        """
        if len(self.returns_series) < 2:
            return 0.0
        
        return self.returns_series.std() * np.sqrt(annualization_factor)
    
    def skewness(self) -> float:
        """Calculate returns skewness."""
        if len(self.returns_series) < 3:
            return 0.0
        
        return stats.skew(self.returns_series.dropna())
    
    def kurtosis(self) -> float:
        """Calculate returns kurtosis (excess kurtosis)."""
        if len(self.returns_series) < 4:
            return 0.0
        
        return stats.kurtosis(self.returns_series.dropna())
    
    # =========================================================================
    # Comprehensive Report
    # =========================================================================
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            # P&L Metrics
            'total_pnl': self.total_pnl(),
            'total_return': self.total_return(),
            'total_return_pct': self.total_return() * 100,
            'sharpe_ratio': self.sharpe_ratio(),
            'max_drawdown': self.max_drawdown(),
            'max_drawdown_pct': self.max_drawdown() * 100,
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            
            # Hedging Metrics
            'num_hedges': self.results.num_hedges,
            'hedge_frequency': self.hedge_frequency(),
            'avg_hedge_cost': self.average_hedge_cost(),
            'total_hedge_costs': self.results.total_transaction_costs,
            'hedge_cost_ratio': self.total_hedge_cost_ratio(),
            'delta_tracking_error': self.delta_tracking_error(),
            'avg_abs_delta': self.average_absolute_delta(),
            
            # Risk Metrics
            'volatility': self.volatility(),
            'var_95': self.value_at_risk(0.95),
            'cvar_95': self.conditional_var(0.95),
            'skewness': self.skewness(),
            'kurtosis': self.kurtosis(),
            
            # Basic Info
            'initial_value': self.results.initial_value,
            'final_value': self.results.final_value,
            'num_timesteps': len(self.results.state_tracker),
        }
        
        return metrics
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all metrics to DataFrame.
        
        Returns:
            DataFrame with metrics
        """
        metrics = self.calculate_all_metrics()
        df = pd.DataFrame([metrics]).T
        df.columns = ['Value']
        return df
    
    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics("
            f"Sharpe={self.sharpe_ratio():.2f}, "
            f"MaxDD={self.max_drawdown():.2%})"
        )

