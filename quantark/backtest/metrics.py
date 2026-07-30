"""
Cross-asset performance metrics over the BaseBacktestResults protocol.

``CorePerformanceMetrics`` consumes only the protocol accessors
(``get_pnl_series`` / ``get_value_series`` / ``get_hedge_trades`` /
``get_total_pnl`` / ``get_total_return``), so any results object — equity,
replay, FI — can report Sharpe/drawdown/VaR without asset-class coupling.
Equity's ``PerformanceMetrics`` extends this with hedge/delta metrics that
need equity-specific attributes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


class CorePerformanceMetrics:
    """
    Performance metrics computable from any BaseBacktestResults.

    Attributes:
        results: A results object satisfying the BaseBacktestResults protocol.
    """

    def __init__(self, results: Any):
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
                self._returns_series = pd.Series(dtype=float)
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

    def sharpe_ratio(
        self, risk_free_rate: float = 0.0, annualization_factor: int = 252
    ) -> float:
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

        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(
            annualization_factor
        )

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
        return abs(drawdown.min())

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
        return winning_periods / len(self.returns_series)

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
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

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

        return abs(self.returns_series.quantile(1 - confidence_level))

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
        Calculate all protocol-computable metrics.

        Returns:
            Dictionary with core P&L and risk metrics
        """
        return {
            "total_pnl": self.total_pnl(),
            "total_return": self.total_return(),
            "total_return_pct": self.total_return() * 100,
            "sharpe_ratio": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_pct": self.max_drawdown() * 100,
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "volatility": self.volatility(),
            "var_95": self.value_at_risk(0.95),
            "cvar_95": self.conditional_var(0.95),
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
        }

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
            f"{type(self).__name__}("
            f"Sharpe={self.sharpe_ratio():.2f}, "
            f"MaxDD={self.max_drawdown():.2%})"
        )
