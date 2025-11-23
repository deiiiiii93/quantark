"""
Static visualization components using matplotlib and seaborn.
"""

from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


class StaticVisualizer:
    """
    Create static plots for backtest results.

    Provides methods for creating various plots:
    - P&L over time
    - Portfolio Greeks evolution
    - Delta tracking
    - Hedge frequency
    - Drawdown chart
    - Returns distribution

    Attributes:
        results: BacktestResults instance
        save_dir: Directory to save plots
    """

    def __init__(self, results: "BacktestResults", save_dir: Optional[str] = None):
        """
        Initialize static visualizer.

        Args:
            results: BacktestResults instance
            save_dir: Directory to save plots (created if doesn't exist)
        """
        self.results = results
        self.save_dir = Path(save_dir) if save_dir else Path("plots")

        if save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_pnl_over_time(
        self,
        figsize: Tuple[int, int] = (14, 6),
        save: bool = False,
        filename: str = "pnl_over_time.png",
    ) -> plt.Figure:
        """
        Plot P&L evolution over time.

        Args:
            figsize: Figure size
            save: Whether to save the plot
            filename: Filename if saving

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        pnl_series = self.results.get_pnl_series()

        ax.plot(pnl_series.index, pnl_series.values, linewidth=2, color="steelblue")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.fill_between(
            pnl_series.index,
            0,
            pnl_series.values,
            where=(pnl_series.values >= 0),
            alpha=0.3,
            color="green",
            label="Profit",
        )
        ax.fill_between(
            pnl_series.index,
            0,
            pnl_series.values,
            where=(pnl_series.values < 0),
            alpha=0.3,
            color="red",
            label="Loss",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("P&L ($)")
        ax.set_title(
            f"Portfolio P&L Over Time - {self.results.config.underlying}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add final P&L annotation
        final_pnl = pnl_series.iloc[-1]
        ax.annotate(
            f"Final P&L: ${final_pnl:,.2f}",
            xy=(pnl_series.index[-1], final_pnl),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            fontsize=10,
            fontweight="bold",
        )

        plt.tight_layout()

        if save and self.save_dir:
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        return fig

    def plot_portfolio_value(
        self,
        figsize: Tuple[int, int] = (14, 6),
        save: bool = False,
        filename: str = "portfolio_value.png",
    ) -> plt.Figure:
        """
        Plot portfolio value over time.

        Args:
            figsize: Figure size
            save: Whether to save
            filename: Filename if saving

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        value_series = self.results.get_value_series()

        ax.plot(value_series.index, value_series.values, linewidth=2, color="darkgreen")
        ax.axhline(
            y=self.results.initial_value,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Initial Value",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title(
            f"Portfolio Value Over Time - {self.results.config.underlying}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add return annotation
        total_return = self.results.get_total_return()
        ax.text(
            0.02,
            0.98,
            f"Total Return: {total_return:.2%}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save and self.save_dir:
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        return fig

    def plot_greeks_evolution(
        self,
        greeks: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10),
        save: bool = False,
        filename: str = "greeks_evolution.png",
    ) -> plt.Figure:
        """
        Plot Greeks evolution over time.

        Args:
            greeks: List of Greeks to plot (default: ['delta', 'gamma', 'vega', 'theta'])
            figsize: Figure size
            save: Whether to save
            filename: Filename if saving

        Returns:
            Matplotlib figure
        """
        if greeks is None:
            greeks = ["delta", "gamma", "vega", "theta"]

        greeks_df = self.results.get_greeks_series()

        # Filter available Greeks
        available_greeks = []
        for greek in greeks:
            col_name = f"greek_{greek}"
            if col_name in greeks_df.columns:
                available_greeks.append(greek)

        if not available_greeks:
            print("No Greeks data available")
            return None

        n_greeks = len(available_greeks)
        fig, axes = plt.subplots(n_greeks, 1, figsize=figsize, sharex=True)

        if n_greeks == 1:
            axes = [axes]

        colors = ["steelblue", "crimson", "purple", "orange", "green"]

        for i, greek in enumerate(available_greeks):
            col_name = f"greek_{greek}"
            data = greeks_df[col_name]

            axes[i].plot(
                data.index, data.values, linewidth=2, color=colors[i % len(colors)]
            )
            axes[i].axhline(y=0, color="black", linestyle="--", alpha=0.3)
            axes[i].set_ylabel(greek.capitalize())
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"{greek.capitalize()} Over Time", fontsize=12)

        axes[-1].set_xlabel("Date")
        fig.suptitle(
            f"Portfolio Greeks Evolution - {self.results.config.underlying}",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout()

        if save and self.save_dir:
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        return fig

    def plot_delta_tracking(
        self,
        figsize: Tuple[int, int] = (14, 6),
        save: bool = False,
        filename: str = "delta_tracking.png",
    ) -> plt.Figure:
        """
        Plot delta tracking (actual vs target).

        Args:
            figsize: Figure size
            save: Whether to save
            filename: Filename if saving

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        delta_series = self.results.get_delta_series()
        target_delta = self.results.config.strategy.target_delta
        threshold = self.results.config.strategy.delta_threshold

        # Plot delta
        ax.plot(
            delta_series.index,
            delta_series.values,
            linewidth=2,
            color="steelblue",
            label="Actual Delta",
        )

        # Plot target and thresholds
        ax.axhline(
            y=target_delta,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Target Delta",
        )
        ax.axhline(
            y=threshold, color="red", linestyle=":", alpha=0.5, label="Upper Threshold"
        )
        ax.axhline(
            y=-threshold, color="red", linestyle=":", alpha=0.5, label="Lower Threshold"
        )

        # Shade threshold zones
        ax.fill_between(
            delta_series.index, threshold, delta_series.max(), alpha=0.1, color="red"
        )
        ax.fill_between(
            delta_series.index, -threshold, delta_series.min(), alpha=0.1, color="red"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Delta")
        ax.set_title(
            f"Delta Tracking - {self.results.config.underlying}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add tracking error
        tracking_error = self.results.metrics.delta_tracking_error()
        ax.text(
            0.02,
            0.98,
            f"Tracking Error: {tracking_error:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save and self.save_dir:
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        return fig

    def plot_hedge_frequency(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save: bool = False,
        filename: str = "hedge_frequency.png",
    ) -> plt.Figure:
        """
        Plot hedge frequency histogram.

        Args:
            figsize: Figure size
            save: Whether to save
            filename: Filename if saving

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        trades_df = self.results.get_hedge_trades()

        if len(trades_df) == 0:
            print("No hedge trades to plot")
            return None

        # Histogram of hedge sizes
        hedge_sizes = trades_df["quantity"].values
        ax1.hist(hedge_sizes, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
        ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Hedge Size")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Hedge Sizes", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Trades over time
        trades_per_day = trades_df.resample("D").size()
        ax2.bar(trades_per_day.index, trades_per_day.values, color="orange", alpha=0.7)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Number of Hedges")
        ax2.set_title("Hedges Per Day", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f"Hedge Frequency Analysis - {self.results.config.underlying}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save and self.save_dir:
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        return fig

    def plot_drawdown(
        self,
        figsize: Tuple[int, int] = (14, 6),
        save: bool = False,
        filename: str = "drawdown.png",
    ) -> plt.Figure:
        """
        Plot drawdown over time.

        Args:
            figsize: Figure size
            save: Whether to save
            filename: Filename if saving

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        value_series = self.results.get_value_series()
        cumulative_max = value_series.expanding().max()
        drawdown = (value_series - cumulative_max) / cumulative_max

        ax.fill_between(drawdown.index, 0, drawdown.values, color="red", alpha=0.3)
        ax.plot(drawdown.index, drawdown.values, linewidth=2, color="darkred")

        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.set_title(
            f"Drawdown Over Time - {self.results.config.underlying}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Add max drawdown annotation
        max_dd = self.results.metrics.max_drawdown()
        ax.text(
            0.02,
            0.02,
            f"Max Drawdown: {max_dd:.2%}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
        )

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        plt.tight_layout()

        if save and self.save_dir:
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        return fig

    def plot_returns_distribution(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save: bool = False,
        filename: str = "returns_distribution.png",
    ) -> plt.Figure:
        """
        Plot returns distribution.

        Args:
            figsize: Figure size
            save: Whether to save
            filename: Filename if saving

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        returns = self.results.metrics.returns_series

        if len(returns) == 0:
            print("No returns data available")
            return None

        # Histogram
        ax1.hist(
            returns.values, bins=50, color="steelblue", alpha=0.7, edgecolor="black"
        )
        ax1.axvline(
            x=returns.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {returns.mean():.4f}",
        )
        ax1.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax1.set_xlabel("Returns")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Returns Distribution", fontsize=12, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats

        stats.probplot(returns.values, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal Distribution)", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Add statistics
        stats_text = (
            f"Mean: {returns.mean():.4f}\n"
            f"Std: {returns.std():.4f}\n"
            f"Skew: {self.results.metrics.skewness():.2f}\n"
            f"Kurt: {self.results.metrics.kurtosis():.2f}"
        )
        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.suptitle(
            f"Returns Analysis - {self.results.config.underlying}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save and self.save_dir:
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        return fig

    def create_summary_dashboard(
        self,
        figsize: Tuple[int, int] = (20, 12),
        save: bool = False,
        filename: str = "summary_dashboard.png",
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple plots.

        Args:
            figsize: Figure size
            save: Whether to save
            filename: Filename if saving

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # P&L
        ax1 = fig.add_subplot(gs[0, :2])
        pnl_series = self.results.get_pnl_series()
        ax1.plot(pnl_series.index, pnl_series.values, linewidth=2, color="steelblue")
        ax1.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax1.set_title("P&L Over Time", fontweight="bold")
        ax1.set_ylabel("P&L ($)")
        ax1.grid(True, alpha=0.3)

        # Delta tracking
        ax2 = fig.add_subplot(gs[1, :2])
        delta_series = self.results.get_delta_series()
        ax2.plot(delta_series.index, delta_series.values, linewidth=2, color="purple")
        ax2.axhline(
            y=self.results.config.strategy.target_delta, color="green", linestyle="--"
        )
        ax2.set_title("Delta Tracking", fontweight="bold")
        ax2.set_ylabel("Delta")
        ax2.grid(True, alpha=0.3)

        # Portfolio value
        ax3 = fig.add_subplot(gs[2, :2])
        value_series = self.results.get_value_series()
        ax3.plot(
            value_series.index, value_series.values, linewidth=2, color="darkgreen"
        )
        ax3.set_title("Portfolio Value", fontweight="bold")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Value ($)")
        ax3.grid(True, alpha=0.3)

        # Returns distribution
        ax4 = fig.add_subplot(gs[0, 2])
        returns = self.results.metrics.returns_series
        if len(returns) > 0:
            ax4.hist(
                returns.values, bins=30, color="steelblue", alpha=0.7, edgecolor="black"
            )
            ax4.axvline(x=0, color="red", linestyle="--", alpha=0.5)
            ax4.set_title("Returns Distribution", fontweight="bold")
            ax4.set_ylabel("Frequency")

        # Drawdown
        ax5 = fig.add_subplot(gs[1, 2])
        cumulative_max = value_series.expanding().max()
        drawdown = (value_series - cumulative_max) / cumulative_max
        ax5.fill_between(drawdown.index, 0, drawdown.values, color="red", alpha=0.3)
        ax5.plot(drawdown.index, drawdown.values, linewidth=1, color="darkred")
        ax5.set_title("Drawdown", fontweight="bold")
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # Metrics table
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis("off")
        metrics_data = [
            ["Total Return", f"{self.results.get_total_return():.2%}"],
            ["Sharpe Ratio", f"{self.results.metrics.sharpe_ratio():.2f}"],
            ["Max Drawdown", f"{self.results.metrics.max_drawdown():.2%}"],
            ["Win Rate", f"{self.results.metrics.win_rate():.2%}"],
            ["Num Hedges", f"{self.results.num_hedges}"],
            ["Hedge Costs", f"${self.results.total_transaction_costs:,.0f}"],
        ]
        table = ax6.table(
            cellText=metrics_data,
            cellLoc="left",
            bbox=[0, 0, 1, 1],
            colWidths=[0.6, 0.4],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax6.set_title("Key Metrics", fontweight="bold", pad=20)

        fig.suptitle(
            f"Backtest Summary Dashboard - {self.results.config.underlying}",
            fontsize=16,
            fontweight="bold",
        )

        if save and self.save_dir:
            fig.savefig(self.save_dir / filename, dpi=300, bbox_inches="tight")

        return fig

    def generate_all_plots(self, save: bool = True) -> Dict[str, plt.Figure]:
        """
        Generate all available plots.

        Args:
            save: Whether to save all plots

        Returns:
            Dictionary of plot name to figure
        """
        plots = {}

        plots["pnl"] = self.plot_pnl_over_time(save=save)
        plots["value"] = self.plot_portfolio_value(save=save)
        plots["greeks"] = self.plot_greeks_evolution(save=save)
        plots["delta_tracking"] = self.plot_delta_tracking(save=save)
        plots["hedge_frequency"] = self.plot_hedge_frequency(save=save)
        plots["drawdown"] = self.plot_drawdown(save=save)
        plots["returns"] = self.plot_returns_distribution(save=save)
        plots["dashboard"] = self.create_summary_dashboard(save=save)

        return plots

    def __repr__(self) -> str:
        return f"StaticVisualizer(underlying={self.results.config.underlying})"
