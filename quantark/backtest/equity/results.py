"""
Equity backtest results container and analysis.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
from .state import StateTracker
from .config import BacktestConfig


class BacktestResults:
    """
    Container for backtest results and analysis.

    Provides access to:
    - Time series of portfolio states
    - Trade history
    - Performance metrics
    - Configuration details

    Attributes:
        config: Backtest configuration
        state_tracker: State history tracker
        initial_value: Initial portfolio value
        final_value: Final portfolio value
        num_hedges: Number of hedges executed
        total_transaction_costs: Total costs incurred
    """

    def __init__(
        self,
        config: BacktestConfig,
        state_tracker: StateTracker,
        initial_value: float,
        final_value: float,
        num_hedges: int,
        total_transaction_costs: float,
    ):
        """
        Initialize backtest results.

        Args:
            config: Backtest configuration
            state_tracker: State history
            initial_value: Initial portfolio value
            final_value: Final portfolio value
            num_hedges: Number of hedges
            total_transaction_costs: Total transaction costs
        """
        self.config = config
        self.state_tracker = state_tracker
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_hedges = num_hedges
        self.total_transaction_costs = total_transaction_costs

        # Lazy-loaded dataframes
        self._states_df: Optional[pd.DataFrame] = None
        self._trades_df: Optional[pd.DataFrame] = None
        self._metrics: Optional["PerformanceMetrics"] = None

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
    def metrics(self) -> "PerformanceMetrics":
        """Get performance metrics (cached)."""
        if self._metrics is None:
            from .metrics import PerformanceMetrics

            self._metrics = PerformanceMetrics(self)
        return self._metrics

    def get_total_pnl(self) -> float:
        """Get total P&L."""
        return self.final_value - self.initial_value

    def get_total_return(self) -> float:
        """Get total return as percentage."""
        if self.initial_value == 0:
            return 0.0
        return (self.final_value - self.initial_value) / self.initial_value

    def get_pnl_series(self) -> pd.Series:
        """Get P&L time series."""
        if "pnl" in self.states_df.columns:
            return self.states_df["pnl"]
        return pd.Series()

    def get_value_series(self) -> pd.Series:
        """Get portfolio value time series."""
        if "portfolio_value" in self.states_df.columns:
            return self.states_df["portfolio_value"]
        return pd.Series()

    def get_delta_series(self) -> pd.Series:
        """Get delta time series."""
        if "greek_delta" in self.states_df.columns:
            return self.states_df["greek_delta"]
        return pd.Series()

    def get_greeks_series(self) -> pd.DataFrame:
        """Get all Greeks time series."""
        greek_cols = [col for col in self.states_df.columns if col.startswith("greek_")]
        if greek_cols:
            return self.states_df[greek_cols]
        return pd.DataFrame()

    def get_market_data_series(self) -> pd.DataFrame:
        """Get market data time series."""
        market_cols = [
            col for col in self.states_df.columns if col.startswith("market_")
        ]
        if market_cols:
            return self.states_df[market_cols]
        return pd.DataFrame()

    def get_hedge_trades(self) -> pd.DataFrame:
        """Get only hedge trades."""
        if len(self.trades_df) == 0:
            return pd.DataFrame()
        return self.trades_df[
            self.trades_df["trade_type"].isin(["open", "adjust", "close"])
        ]

    def get_lifecycle_events(self) -> pd.DataFrame:
        """Get realized product lifecycle events.

        One row per event (knock-out, knock-in, coupon, maturity, expiry)
        detected during the backtest, indexed by event date. Empty when
        ``handle_lifecycle_events`` was off or no events fired.
        """
        return self.state_tracker.get_lifecycle_events_dataframe()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of results.

        Returns:
            Dictionary with summary statistics
        """
        total_pnl = self.get_total_pnl()
        total_return = self.get_total_return()

        summary = {
            "backtest_name": f"{self.config.underlying}_{self.config.strategy.name}",
            "start_date": self.config.start_date.isoformat(),
            "end_date": self.config.end_date.isoformat(),
            "num_timesteps": len(self.state_tracker),
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "num_hedges": self.num_hedges,
            "total_transaction_costs": self.total_transaction_costs,
            "avg_cost_per_hedge": (
                self.total_transaction_costs / self.num_hedges
                if self.num_hedges > 0
                else 0
            ),
            "strategy": self.config.strategy.get_parameters(),
        }

        # Add metrics if calculated
        if self._metrics:
            summary["sharpe_ratio"] = self.metrics.sharpe_ratio()
            summary["max_drawdown"] = self.metrics.max_drawdown()
            summary["win_rate"] = self.metrics.win_rate()

        return summary

    def export_to_excel(self, filepath: str):
        """
        Export results to Excel file.

        Args:
            filepath: Path to save Excel file
        """
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Summary sheet
            summary_df = pd.DataFrame([self.get_summary()]).T
            summary_df.columns = ["Value"]
            summary_df.to_excel(writer, sheet_name="Summary")

            # States sheet
            self.states_df.to_excel(writer, sheet_name="States")

            # Trades sheet
            if len(self.trades_df) > 0:
                self.trades_df.to_excel(writer, sheet_name="Trades")

            # Greeks sheet
            greeks_df = self.get_greeks_series()
            if len(greeks_df) > 0:
                greeks_df.to_excel(writer, sheet_name="Greeks")

            # Market data sheet
            market_df = self.get_market_data_series()
            if len(market_df) > 0:
                market_df.to_excel(writer, sheet_name="MarketData")

    def export_to_parquet(self, filepath: str):
        """
        Export results to Parquet file.

        Args:
            filepath: Path to save Parquet file
        """
        # Combine all data
        combined_df = self.states_df.copy()

        # Add metadata as attributes
        combined_df.attrs["metadata"] = self.get_summary()

        # Save
        combined_df.to_parquet(filepath)

    def __repr__(self) -> str:
        return (
            f"BacktestResults("
            f"PnL=${self.get_total_pnl():,.2f}, "
            f"Return={self.get_total_return():.2%}, "
            f"Hedges={self.num_hedges})"
        )


# Alias for explicit naming
EquityBacktestResults = BacktestResults
