"""
Performance metrics calculation for backtest results.

Core P&L/risk metrics live in :class:`quantark.backtest.metrics.CorePerformanceMetrics`
(protocol-only); this class adds the equity hedge/delta metrics that need
equity-results attributes. Public API unchanged.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

from quantark.backtest.metrics import CorePerformanceMetrics


class PerformanceMetrics(CorePerformanceMetrics):
    """
    Calculate performance metrics from backtest results.

    Provides comprehensive metrics including:
    - P&L metrics (total, Sharpe, max drawdown) — inherited core
    - Hedging metrics (frequency, costs, tracking error)
    - Risk metrics (VaR, CVaR) — inherited core

    Attributes:
        results: BacktestResults instance
    """

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
        return (
            avg_abs_delta / self.results.num_hedges
            if self.results.num_hedges > 0
            else 0.0
        )

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate all available metrics.

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            # P&L Metrics
            "total_pnl": self.total_pnl(),
            "total_return": self.total_return(),
            "total_return_pct": self.total_return() * 100,
            "sharpe_ratio": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_pct": self.max_drawdown() * 100,
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            # Hedging Metrics
            "num_hedges": self.results.num_hedges,
            "hedge_frequency": self.hedge_frequency(),
            "avg_hedge_cost": self.average_hedge_cost(),
            "total_hedge_costs": self.results.total_transaction_costs,
            "hedge_cost_ratio": self.total_hedge_cost_ratio(),
            "delta_tracking_error": self.delta_tracking_error(),
            "avg_abs_delta": self.average_absolute_delta(),
            # Risk Metrics
            "volatility": self.volatility(),
            "var_95": self.value_at_risk(0.95),
            "cvar_95": self.conditional_var(0.95),
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
            # Basic Info
            "initial_value": self.results.initial_value,
            "final_value": self.results.final_value,
            "num_timesteps": len(self.results.state_tracker),
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
        df.columns = ["Value"]
        return df

    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics("
            f"Sharpe={self.sharpe_ratio():.2f}, "
            f"MaxDD={self.max_drawdown():.2%})"
        )


# Alias for explicit naming
EquityPerformanceMetrics = PerformanceMetrics
