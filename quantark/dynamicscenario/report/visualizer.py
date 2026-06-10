"""
Visualization utilities for dynamic scenario results.

Provides both static (matplotlib) and interactive (plotly) visualizations
for day-by-day portfolio evolution.
"""

from pathlib import Path
from typing import Union, Optional, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from quantark.dynamicscenario.results.dynamic_results import DynamicScenarioResults


# Type hint for FI results (imported only when needed)
try:
    from quantark.dynamicscenario.fi.results import FIDynamicScenarioResults
except ImportError:
    FIDynamicScenarioResults = None


class DynamicScenarioVisualizer:
    """
    Creates visualizations for dynamic scenario results.

    Supports both static (matplotlib) and interactive (plotly) visualizations
    for day-by-day evolution of portfolio metrics.

    Example:
        >>> viz = DynamicScenarioVisualizer()
        >>> viz.create_all_plots(results, output_dir="./plots")
    """

    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style to use
        """
        self.style = style
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            pass

        # Set seaborn color palette
        sns.set_palette("husl")

        # Color scheme
        self.colors = {
            "primary": "#1a5f7a",
            "secondary": "#2d8ba4",
            "positive": "#27ae60",
            "negative": "#e74c3c",
            "neutral": "#7f8c8d",
            "warning": "#f39c12",
            "purple": "#9b59b6",
        }

    def plot_portfolio_value(
        self,
        results: DynamicScenarioResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 6),
    ) -> None:
        """
        Plot portfolio value evolution over days.

        Args:
            results: Dynamic scenario results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        days = [d.day_index for d in results.day_results]
        values = [d.portfolio_value for d in results.day_results]

        ax.plot(
            days, values, "o-", color=self.colors["primary"], linewidth=2, markersize=8
        )
        ax.axhline(
            y=results.baseline_value,
            color=self.colors["neutral"],
            linestyle="--",
            label=f"Initial: ${results.baseline_value:,.0f}",
        )

        ax.fill_between(
            days,
            results.baseline_value,
            values,
            where=[v >= results.baseline_value for v in values],
            color=self.colors["positive"],
            alpha=0.3,
        )
        ax.fill_between(
            days,
            results.baseline_value,
            values,
            where=[v < results.baseline_value for v in values],
            color=self.colors["negative"],
            alpha=0.3,
        )

        ax.set_xlabel("Day", fontsize=12)
        ax.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax.set_title(
            f"Portfolio Value Evolution - {results.path_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Add day labels if available
        for i, d in enumerate(results.day_results):
            if d.label:
                ax.annotate(
                    d.label,
                    (days[i], values[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    rotation=45,
                )

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved portfolio value plot to: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_pnl_evolution(
        self,
        results: DynamicScenarioResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 6),
    ) -> None:
        """
        Plot P&L evolution showing daily and cumulative P&L.

        Args:
            results: Dynamic scenario results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        days = [d.day_index for d in results.day_results]
        daily_pnl = [d.daily_pnl for d in results.day_results]
        cumulative_pnl = [d.cumulative_pnl for d in results.day_results]
        net_pnl = [d.net_pnl for d in results.day_results]

        # Daily P&L bar chart
        colors = [
            self.colors["positive"] if p >= 0 else self.colors["negative"]
            for p in daily_pnl
        ]
        ax1.bar(days, daily_pnl, color=colors, alpha=0.7, edgecolor="black")
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax1.set_ylabel("Daily P&L ($)", fontsize=11)
        ax1.set_title("Daily P&L", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        # Cumulative P&L line chart
        ax2.plot(
            days,
            cumulative_pnl,
            "o-",
            color=self.colors["positive"],
            linewidth=2,
            markersize=6,
            label="Cumulative P&L",
        )
        ax2.plot(
            days,
            net_pnl,
            "s--",
            color=self.colors["primary"],
            linewidth=1.5,
            markersize=5,
            label="Net P&L (after costs)",
        )
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.fill_between(
            days,
            0,
            cumulative_pnl,
            where=[p >= 0 for p in cumulative_pnl],
            color=self.colors["positive"],
            alpha=0.2,
        )
        ax2.fill_between(
            days,
            0,
            cumulative_pnl,
            where=[p < 0 for p in cumulative_pnl],
            color=self.colors["negative"],
            alpha=0.2,
        )
        ax2.set_xlabel("Day", fontsize=11)
        ax2.set_ylabel("Cumulative P&L ($)", fontsize=11)
        ax2.set_title("Cumulative P&L", fontsize=12, fontweight="bold")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            f"P&L Evolution - {results.path_name}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved P&L evolution plot to: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_greeks_evolution(
        self,
        results: DynamicScenarioResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (14, 10),
    ) -> None:
        """
        Plot Greeks evolution over days.

        Args:
            results: Dynamic scenario results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        days = [d.day_index for d in results.day_results]

        # Extract Greeks
        deltas = [d.greeks.get("delta", 0) for d in results.day_results]
        gammas = [d.greeks.get("gamma", 0) for d in results.day_results]
        vegas = [d.greeks.get("vega", 0) for d in results.day_results]
        thetas = [d.greeks.get("theta", 0) for d in results.day_results]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Delta
        ax = axes[0, 0]
        ax.plot(
            days, deltas, "o-", color=self.colors["negative"], linewidth=2, markersize=6
        )
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax.fill_between(days, 0, deltas, alpha=0.3, color=self.colors["negative"])
        ax.set_ylabel("Delta", fontsize=11)
        ax.set_title("Delta Evolution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Gamma
        ax = axes[0, 1]
        ax.plot(
            days, gammas, "o-", color=self.colors["purple"], linewidth=2, markersize=6
        )
        ax.fill_between(days, 0, gammas, alpha=0.3, color=self.colors["purple"])
        ax.set_ylabel("Gamma", fontsize=11)
        ax.set_title("Gamma Evolution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Vega
        ax = axes[1, 0]
        ax.plot(
            days, vegas, "o-", color=self.colors["warning"], linewidth=2, markersize=6
        )
        ax.fill_between(days, 0, vegas, alpha=0.3, color=self.colors["warning"])
        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Vega", fontsize=11)
        ax.set_title("Vega Evolution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Theta
        ax = axes[1, 1]
        ax.plot(
            days, thetas, "o-", color=self.colors["primary"], linewidth=2, markersize=6
        )
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax.fill_between(days, 0, thetas, alpha=0.3, color=self.colors["primary"])
        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Theta", fontsize=11)
        ax.set_title("Theta Evolution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Greeks Evolution - {results.path_name}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved Greeks evolution plot to: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_market_evolution(
        self,
        results: DynamicScenarioResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 8),
    ) -> None:
        """
        Plot market parameters (spot, vol) evolution over days.

        Args:
            results: Dynamic scenario results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        days = [d.day_index for d in results.day_results]

        # Extract market data
        spots = []
        vols = []
        rates = []

        for d in results.day_results:
            if d.market_state:
                spot_vals = list(d.market_state.spot.values())
                vol_vals = list(d.market_state.volatility.values())
                spots.append(spot_vals[0] if spot_vals else 0)
                vols.append(vol_vals[0] * 100 if vol_vals else 0)  # Convert to %
                rates.append(d.market_state.rate * 100)  # Convert to %
            else:
                spots.append(0)
                vols.append(0)
                rates.append(0)

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Spot price
        ax = axes[0]
        ax.plot(
            days, spots, "o-", color=self.colors["primary"], linewidth=2, markersize=6
        )
        ax.fill_between(
            days, min(spots) * 0.98, spots, alpha=0.3, color=self.colors["primary"]
        )
        ax.set_ylabel("Spot Price ($)", fontsize=11)
        ax.set_title("Spot Price Evolution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Volatility
        ax = axes[1]
        ax.plot(
            days, vols, "o-", color=self.colors["negative"], linewidth=2, markersize=6
        )
        ax.fill_between(days, 0, vols, alpha=0.3, color=self.colors["negative"])
        ax.set_ylabel("Volatility (%)", fontsize=11)
        ax.set_title("Implied Volatility Evolution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Rate
        ax = axes[2]
        ax.plot(
            days, rates, "o-", color=self.colors["positive"], linewidth=2, markersize=6
        )
        ax.fill_between(days, 0, rates, alpha=0.3, color=self.colors["positive"])
        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Risk-Free Rate (%)", fontsize=11)
        ax.set_title("Interest Rate Evolution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Market Parameters - {results.path_name}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved market evolution plot to: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_drawdown(
        self,
        results: DynamicScenarioResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 5),
    ) -> None:
        """
        Plot drawdown chart.

        Args:
            results: Dynamic scenario results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        days = [d.day_index for d in results.day_results]
        values = [d.portfolio_value for d in results.day_results]

        # Calculate running maximum and drawdown
        running_max = []
        drawdowns = []
        current_max = results.baseline_value

        for v in values:
            current_max = max(current_max, v)
            running_max.append(current_max)
            drawdowns.append((v - current_max) / current_max * 100)

        fig, ax = plt.subplots(figsize=figsize)

        ax.fill_between(days, 0, drawdowns, color=self.colors["negative"], alpha=0.5)
        ax.plot(
            days,
            drawdowns,
            "o-",
            color=self.colors["negative"],
            linewidth=2,
            markersize=6,
        )
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

        # Mark max drawdown point
        min_dd = min(drawdowns)
        min_dd_day = days[drawdowns.index(min_dd)]
        ax.scatter([min_dd_day], [min_dd], color="red", s=150, zorder=5, marker="v")
        ax.annotate(
            f"Max DD: {min_dd:.2f}%",
            (min_dd_day, min_dd),
            textcoords="offset points",
            xytext=(10, -15),
            fontsize=10,
            fontweight="bold",
            color="red",
        )

        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Drawdown (%)", fontsize=11)
        ax.set_title(
            f"Drawdown Chart - {results.path_name}", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved drawdown plot to: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_trades(
        self,
        results: DynamicScenarioResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 6),
    ) -> None:
        """
        Plot hedge trades over time.

        Args:
            results: Dynamic scenario results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        # Collect trade data
        trade_days = []
        trade_quantities = []
        trade_costs = []

        for d in results.day_results:
            for trade in d.trades:
                trade_days.append(d.day_index)
                trade_quantities.append(trade.quantity)
                trade_costs.append(trade.transaction_cost)

        if not trade_days:
            print("No trades to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Trade quantities
        colors = [
            self.colors["positive"] if q >= 0 else self.colors["negative"]
            for q in trade_quantities
        ]
        ax1.bar(
            trade_days, trade_quantities, color=colors, alpha=0.7, edgecolor="black"
        )
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax1.set_ylabel("Trade Quantity", fontsize=11)
        ax1.set_title("Hedge Trade Quantities", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        # Cumulative transaction costs
        cum_costs = np.cumsum(trade_costs)
        ax2.plot(
            trade_days,
            cum_costs,
            "o-",
            color=self.colors["warning"],
            linewidth=2,
            markersize=6,
        )
        ax2.fill_between(
            trade_days, 0, cum_costs, alpha=0.3, color=self.colors["warning"]
        )
        ax2.set_xlabel("Day", fontsize=11)
        ax2.set_ylabel("Cumulative Costs ($)", fontsize=11)
        ax2.set_title("Cumulative Transaction Costs", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            f"Hedge Trades - {results.path_name}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved trades plot to: {filepath}")
        else:
            plt.show()

        plt.close()

    def create_summary_dashboard(
        self,
        results: DynamicScenarioResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (16, 12),
    ) -> None:
        """
        Create a comprehensive summary dashboard with multiple plots.

        Args:
            results: Dynamic scenario results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        days = [d.day_index for d in results.day_results]
        values = [d.portfolio_value for d in results.day_results]
        daily_pnl = [d.daily_pnl for d in results.day_results]
        cumulative_pnl = [d.cumulative_pnl for d in results.day_results]
        deltas = [d.delta for d in results.day_results]

        # 1. Portfolio value (top left, 2 cols)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(
            days, values, "o-", color=self.colors["primary"], linewidth=2, markersize=6
        )
        ax1.axhline(
            y=results.baseline_value,
            color=self.colors["neutral"],
            linestyle="--",
            alpha=0.7,
        )
        ax1.fill_between(
            days,
            results.baseline_value,
            values,
            where=[v >= results.baseline_value for v in values],
            color=self.colors["positive"],
            alpha=0.3,
        )
        ax1.fill_between(
            days,
            results.baseline_value,
            values,
            where=[v < results.baseline_value for v in values],
            color=self.colors["negative"],
            alpha=0.3,
        )
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title("Portfolio Value", fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # 2. Summary stats (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis("off")

        dd_amt, dd_pct, _, _ = results.get_max_drawdown()
        stats_text = (
            f"Path: {results.path_name}\n"
            f"Days: {results.num_days}\n\n"
            f"Initial: ${results.baseline_value:,.0f}\n"
            f"Final: ${results.final_value:,.0f}\n\n"
            f"Total P&L: ${results.total_pnl:+,.0f}\n"
            f"P&L %: {results.total_pnl_pct:+.2f}%\n\n"
            f"Trans. Costs: ${results.total_transaction_costs:,.0f}\n"
            f"Net P&L: ${results.net_pnl:+,.0f}\n\n"
            f"Max Drawdown: {dd_pct:.2f}%\n"
            f"Hedge Trades: {results.total_hedges}"
        )
        ax2.text(
            0.1,
            0.9,
            stats_text,
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax2.set_title("Summary Statistics", fontweight="bold")

        # 3. Daily P&L (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        colors = [
            self.colors["positive"] if p >= 0 else self.colors["negative"]
            for p in daily_pnl
        ]
        ax3.bar(days, daily_pnl, color=colors, alpha=0.7, edgecolor="black")
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax3.set_xlabel("Day")
        ax3.set_ylabel("Daily P&L ($)")
        ax3.set_title("Daily P&L", fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")

        # 4. Cumulative P&L (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(
            days,
            cumulative_pnl,
            "o-",
            color=self.colors["positive"],
            linewidth=2,
            markersize=5,
        )
        ax4.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax4.fill_between(
            days,
            0,
            cumulative_pnl,
            where=[p >= 0 for p in cumulative_pnl],
            color=self.colors["positive"],
            alpha=0.3,
        )
        ax4.fill_between(
            days,
            0,
            cumulative_pnl,
            where=[p < 0 for p in cumulative_pnl],
            color=self.colors["negative"],
            alpha=0.3,
        )
        ax4.set_xlabel("Day")
        ax4.set_ylabel("Cumulative P&L ($)")
        ax4.set_title("Cumulative P&L", fontweight="bold")
        ax4.grid(True, alpha=0.3)

        # 5. Delta evolution (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(
            days, deltas, "o-", color=self.colors["negative"], linewidth=2, markersize=5
        )
        ax5.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax5.fill_between(days, 0, deltas, alpha=0.3, color=self.colors["negative"])
        ax5.set_xlabel("Day")
        ax5.set_ylabel("Delta")
        ax5.set_title("Delta Evolution", fontweight="bold")
        ax5.grid(True, alpha=0.3)

        # 6. Market spot (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        spots = []
        for d in results.day_results:
            if d.market_state and d.market_state.spot:
                spots.append(list(d.market_state.spot.values())[0])
            else:
                spots.append(0)
        ax6.plot(
            days, spots, "o-", color=self.colors["primary"], linewidth=2, markersize=5
        )
        ax6.set_xlabel("Day")
        ax6.set_ylabel("Spot ($)")
        ax6.set_title("Spot Price", fontweight="bold")
        ax6.grid(True, alpha=0.3)

        # 7. Market vol (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        vols = []
        for d in results.day_results:
            if d.market_state and d.market_state.volatility:
                vols.append(list(d.market_state.volatility.values())[0] * 100)
            else:
                vols.append(0)
        ax7.plot(
            days, vols, "o-", color=self.colors["warning"], linewidth=2, markersize=5
        )
        ax7.set_xlabel("Day")
        ax7.set_ylabel("Volatility (%)")
        ax7.set_title("Implied Volatility", fontweight="bold")
        ax7.grid(True, alpha=0.3)

        # 8. Drawdown (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        running_max = []
        drawdowns = []
        current_max = results.baseline_value
        for v in values:
            current_max = max(current_max, v)
            running_max.append(current_max)
            drawdowns.append((v - current_max) / current_max * 100)
        ax8.fill_between(days, 0, drawdowns, color=self.colors["negative"], alpha=0.5)
        ax8.plot(
            days,
            drawdowns,
            "o-",
            color=self.colors["negative"],
            linewidth=2,
            markersize=5,
        )
        ax8.set_xlabel("Day")
        ax8.set_ylabel("Drawdown (%)")
        ax8.set_title("Drawdown", fontweight="bold")
        ax8.grid(True, alpha=0.3)

        plt.suptitle(
            f"Dynamic Scenario Dashboard - {results.path_name}",
            fontsize=16,
            fontweight="bold",
            y=1.01,
        )

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved dashboard to: {filepath}")
        else:
            plt.show()

        plt.close()

    def create_interactive_dashboard(
        self,
        results: DynamicScenarioResults,
        filepath: Optional[Union[str, Path]] = None,
    ) -> go.Figure:
        """
        Create interactive Plotly dashboard.

        Args:
            results: Dynamic scenario results
            filepath: Optional path to save HTML

        Returns:
            Plotly figure object
        """
        # Extract data
        days = [d.day_index for d in results.day_results]
        labels = [d.label or f"Day {d.day_index}" for d in results.day_results]
        values = [d.portfolio_value for d in results.day_results]
        daily_pnl = [d.daily_pnl for d in results.day_results]
        cumulative_pnl = [d.cumulative_pnl for d in results.day_results]
        deltas = [d.delta for d in results.day_results]
        gammas = [d.gamma for d in results.day_results]
        vegas = [d.vega for d in results.day_results]

        spots = []
        vols = []
        for d in results.day_results:
            if d.market_state and d.market_state.spot:
                spots.append(list(d.market_state.spot.values())[0])
            else:
                spots.append(0)
            if d.market_state and d.market_state.volatility:
                vols.append(list(d.market_state.volatility.values())[0] * 100)
            else:
                vols.append(0)

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Portfolio Value Evolution",
                "Daily & Cumulative P&L",
                "Greeks Evolution (Delta, Gamma, Vega)",
                "Market Parameters",
                "Drawdown",
                "Summary Statistics",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # 1. Portfolio value
        fig.add_trace(
            go.Scatter(
                x=days,
                y=values,
                mode="lines+markers",
                name="Portfolio Value",
                line=dict(color="#1a5f7a", width=2),
                text=labels,
                hovertemplate="Day %{x}<br>Value: $%{y:,.0f}<br>%{text}",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=results.baseline_value,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Initial: ${results.baseline_value:,.0f}",
            row=1,
            col=1,
        )

        # 2. P&L
        colors = ["green" if p >= 0 else "red" for p in daily_pnl]
        fig.add_trace(
            go.Bar(
                x=days,
                y=daily_pnl,
                name="Daily P&L",
                marker_color=colors,
                hovertemplate="Day %{x}<br>Daily P&L: $%{y:,.0f}",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=days,
                y=cumulative_pnl,
                mode="lines+markers",
                name="Cumulative P&L",
                line=dict(color="#27ae60", width=2),
                hovertemplate="Day %{x}<br>Cum. P&L: $%{y:,.0f}",
            ),
            row=1,
            col=2,
        )

        # 3. Greeks
        fig.add_trace(
            go.Scatter(
                x=days,
                y=deltas,
                mode="lines+markers",
                name="Delta",
                line=dict(color="#e74c3c", width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=days,
                y=[g * 1000 for g in gammas],
                mode="lines+markers",
                name="Gamma (x1000)",
                line=dict(color="#9b59b6", width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=days,
                y=vegas,
                mode="lines+markers",
                name="Vega",
                line=dict(color="#f39c12", width=2),
            ),
            row=2,
            col=1,
        )

        # 4. Market
        fig.add_trace(
            go.Scatter(
                x=days,
                y=spots,
                mode="lines+markers",
                name="Spot Price",
                line=dict(color="#1a5f7a", width=2),
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=days,
                y=vols,
                mode="lines+markers",
                name="Volatility (%)",
                line=dict(color="#e74c3c", width=2),
                yaxis="y2",
            ),
            row=2,
            col=2,
        )

        # 5. Drawdown
        running_max = []
        drawdowns = []
        current_max = results.baseline_value
        for v in values:
            current_max = max(current_max, v)
            running_max.append(current_max)
            drawdowns.append((v - current_max) / current_max * 100)

        fig.add_trace(
            go.Scatter(
                x=days,
                y=drawdowns,
                mode="lines+markers",
                name="Drawdown",
                fill="tozeroy",
                fillcolor="rgba(231, 76, 60, 0.3)",
                line=dict(color="#e74c3c", width=2),
            ),
            row=3,
            col=1,
        )

        # 6. Summary table
        dd_amt, dd_pct, _, _ = results.get_max_drawdown()
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>"],
                    fill_color="#1a5f7a",
                    font=dict(color="white"),
                    align="left",
                ),
                cells=dict(
                    values=[
                        [
                            "Path Name",
                            "Days",
                            "Initial Value",
                            "Final Value",
                            "Total P&L",
                            "P&L %",
                            "Trans. Costs",
                            "Net P&L",
                            "Max Drawdown",
                            "Hedge Trades",
                        ],
                        [
                            results.path_name,
                            str(results.num_days),
                            f"${results.baseline_value:,.0f}",
                            f"${results.final_value:,.0f}",
                            f"${results.total_pnl:+,.0f}",
                            f"{results.total_pnl_pct:+.2f}%",
                            f"${results.total_transaction_costs:,.0f}",
                            f"${results.net_pnl:+,.0f}",
                            f"{dd_pct:.2f}%",
                            str(results.total_hedges),
                        ],
                    ],
                    fill_color="white",
                    align="left",
                ),
            ),
            row=3,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title_text=f"Dynamic Scenario Dashboard - {results.path_name}",
            title_font_size=18,
            showlegend=True,
            height=1000,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Update axes labels
        fig.update_xaxes(title_text="Day", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

        if filepath:
            fig.write_html(filepath)
            print(f"Saved interactive dashboard to: {filepath}")

        return fig

    def create_all_plots(
        self,
        results: DynamicScenarioResults,
        output_dir: Union[str, Path],
        prefix: Optional[str] = None,
    ) -> List[str]:
        """
        Create all standard plots.

        Args:
            results: Dynamic scenario results
            output_dir: Output directory for plots
            prefix: Prefix for filenames (default: path name)

        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = results.path_name.replace(" ", "_").replace("/", "_")

        created_files = []

        print(f"Generating plots in {output_dir}...")

        # Static plots
        self.plot_portfolio_value(results, output_dir / f"{prefix}_portfolio_value.png")
        created_files.append(str(output_dir / f"{prefix}_portfolio_value.png"))

        self.plot_pnl_evolution(results, output_dir / f"{prefix}_pnl_evolution.png")
        created_files.append(str(output_dir / f"{prefix}_pnl_evolution.png"))

        self.plot_greeks_evolution(
            results, output_dir / f"{prefix}_greeks_evolution.png"
        )
        created_files.append(str(output_dir / f"{prefix}_greeks_evolution.png"))

        self.plot_market_evolution(
            results, output_dir / f"{prefix}_market_evolution.png"
        )
        created_files.append(str(output_dir / f"{prefix}_market_evolution.png"))

        self.plot_drawdown(results, output_dir / f"{prefix}_drawdown.png")
        created_files.append(str(output_dir / f"{prefix}_drawdown.png"))

        # Trades plot (only if there are trades)
        if results.total_hedges > 0:
            self.plot_trades(results, output_dir / f"{prefix}_trades.png")
            created_files.append(str(output_dir / f"{prefix}_trades.png"))

        # Summary dashboard
        self.create_summary_dashboard(results, output_dir / f"{prefix}_dashboard.png")
        created_files.append(str(output_dir / f"{prefix}_dashboard.png"))

        # Interactive dashboard
        self.create_interactive_dashboard(
            results, output_dir / f"{prefix}_interactive.html"
        )
        created_files.append(str(output_dir / f"{prefix}_interactive.html"))

        print(f"All plots generated successfully! ({len(created_files)} files)")

        return created_files

    # =========================================================================
    # FI-SPECIFIC VISUALIZATION METHODS
    # =========================================================================

    def plot_dv01_evolution(
        self,
        results: "FIDynamicScenarioResults",
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 6),
    ) -> None:
        """
        Plot DV01 evolution over days.

        Shows pre-hedge and post-hedge DV01 if hedging was enabled.

        Args:
            results: FI Dynamic scenario results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        days = [d.day_index for d in results.day_results]
        dv01_pre = [d.dv01_pre_hedge for d in results.day_results]
        dv01_post = [d.dv01_post_hedge for d in results.day_results]

        ax.plot(
            days,
            dv01_pre,
            "o-",
            color=self.colors["primary"],
            linewidth=2,
            markersize=8,
            label="DV01 Pre-Hedge",
        )

        # Only show post-hedge if different from pre-hedge
        if any(pre != post for pre, post in zip(dv01_pre, dv01_post)):
            ax.plot(
                days,
                dv01_post,
                "s--",
                color=self.colors["positive"],
                linewidth=2,
                markersize=6,
                label="DV01 Post-Hedge",
            )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        ax.fill_between(
            days,
            0,
            dv01_pre,
            where=[d >= 0 for d in dv01_pre],
            color=self.colors["positive"],
            alpha=0.2,
        )
        ax.fill_between(
            days,
            0,
            dv01_pre,
            where=[d < 0 for d in dv01_pre],
            color=self.colors["negative"],
            alpha=0.2,
        )

        ax.set_xlabel("Day", fontsize=12)
        ax.set_ylabel("DV01 ($)", fontsize=12)
        ax.set_title(
            f"DV01 Evolution - {results.path_name}", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved DV01 evolution plot to: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_duration_evolution(
        self,
        results: "FIDynamicScenarioResults",
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 6),
    ) -> None:
        """
        Plot modified duration evolution over days.

        Args:
            results: FI Dynamic scenario results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        days = [d.day_index for d in results.day_results]
        durations = [d.modified_duration for d in results.day_results]

        ax.plot(
            days,
            durations,
            "o-",
            color=self.colors["purple"],
            linewidth=2,
            markersize=8,
        )
        ax.fill_between(days, 0, durations, alpha=0.3, color=self.colors["purple"])

        ax.set_xlabel("Day", fontsize=12)
        ax.set_ylabel("Modified Duration (years)", fontsize=12)
        ax.set_title(
            f"Duration Evolution - {results.path_name}", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Add reference lines
        if durations:
            avg_duration = sum(durations) / len(durations)
            ax.axhline(
                y=avg_duration,
                color=self.colors["neutral"],
                linestyle="--",
                label=f"Average: {avg_duration:.2f}",
            )
            ax.legend(loc="best")

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved duration evolution plot to: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_rate_evolution(
        self,
        results: "FIDynamicScenarioResults",
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 6),
    ) -> None:
        """
        Plot rate curve evolution over days.

        Args:
            results: FI Dynamic scenario results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        days = [d.day_index for d in results.day_results]
        rates = []
        short_rates = []
        long_rates = []

        for d in results.day_results:
            if d.market_state:
                rates.append(d.market_state.rate * 100)  # Convert to %
                short_rates.append(d.market_state.short_rate * 100)
                long_rates.append(d.market_state.long_rate * 100)
            else:
                rates.append(0)
                short_rates.append(0)
                long_rates.append(0)

        ax.plot(
            days,
            rates,
            "o-",
            color=self.colors["primary"],
            linewidth=2,
            markersize=8,
            label="Rate",
        )

        # Plot short/long if they differ
        if any(s != r for s, r in zip(short_rates, rates)):
            ax.plot(
                days,
                short_rates,
                "s--",
                color=self.colors["positive"],
                linewidth=1.5,
                markersize=5,
                label="Short Rate",
            )
            ax.plot(
                days,
                long_rates,
                "^--",
                color=self.colors["negative"],
                linewidth=1.5,
                markersize=5,
                label="Long Rate",
            )

        ax.fill_between(days, 0, rates, alpha=0.2, color=self.colors["primary"])

        ax.set_xlabel("Day", fontsize=12)
        ax.set_ylabel("Rate (%)", fontsize=12)
        ax.set_title(
            f"Rate Evolution - {results.path_name}", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved rate evolution plot to: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_fi_risk_dashboard(
        self,
        results: "FIDynamicScenarioResults",
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (16, 12),
    ) -> None:
        """
        Create FI-specific risk dashboard with DV01, duration, and rate plots.

        Args:
            results: FI Dynamic scenario results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        days = [d.day_index for d in results.day_results]
        values = [d.portfolio_value for d in results.day_results]
        daily_pnl = [d.daily_pnl for d in results.day_results]
        dv01 = [d.dv01 for d in results.day_results]
        dv01_pre = [d.dv01_pre_hedge for d in results.day_results]
        dv01_post = [d.dv01_post_hedge for d in results.day_results]
        durations = [d.modified_duration for d in results.day_results]
        convexities = [d.convexity for d in results.day_results]

        rates = []
        for d in results.day_results:
            if d.market_state:
                rates.append(d.market_state.rate * 100)
            else:
                rates.append(0)

        # 1. Portfolio value (top left, 2 cols)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(
            days, values, "o-", color=self.colors["primary"], linewidth=2, markersize=6
        )
        ax1.axhline(
            y=results.baseline_value,
            color=self.colors["neutral"],
            linestyle="--",
            alpha=0.7,
        )
        ax1.fill_between(
            days,
            results.baseline_value,
            values,
            where=[v >= results.baseline_value for v in values],
            color=self.colors["positive"],
            alpha=0.3,
        )
        ax1.fill_between(
            days,
            results.baseline_value,
            values,
            where=[v < results.baseline_value for v in values],
            color=self.colors["negative"],
            alpha=0.3,
        )
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title("Portfolio Value", fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # 2. Summary stats (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis("off")

        dd_amt, dd_pct, _, _ = results.get_max_drawdown()
        stats_text = (
            f"Path: {results.path_name}\n"
            f"Days: {results.num_days}\n\n"
            f"Initial: ${results.baseline_value:,.0f}\n"
            f"Final: ${results.final_value:,.0f}\n\n"
            f"Total P&L: ${results.total_pnl:+,.0f}\n"
            f"P&L %: {results.total_pnl_pct:+.2f}%\n\n"
            f"Initial DV01: ${results.baseline_dv01:,.0f}\n"
            f"Final DV01: ${results.final_dv01:,.0f}\n\n"
            f"Initial Duration: {results.baseline_duration:.2f}\n"
            f"Final Duration: {results.final_duration:.2f}\n\n"
            f"Hedge Trades: {results.total_hedges}"
        )
        ax2.text(
            0.1,
            0.9,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax2.set_title("Summary Statistics", fontweight="bold")

        # 3. DV01 evolution (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(
            days,
            dv01_pre,
            "o-",
            color=self.colors["primary"],
            linewidth=2,
            markersize=5,
            label="Pre-Hedge",
        )
        if any(pre != post for pre, post in zip(dv01_pre, dv01_post)):
            ax3.plot(
                days,
                dv01_post,
                "s--",
                color=self.colors["positive"],
                linewidth=1.5,
                markersize=4,
                label="Post-Hedge",
            )
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax3.set_xlabel("Day")
        ax3.set_ylabel("DV01 ($)")
        ax3.set_title("DV01 Evolution", fontweight="bold")
        ax3.legend(loc="best", fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Duration evolution (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(
            days,
            durations,
            "o-",
            color=self.colors["purple"],
            linewidth=2,
            markersize=5,
        )
        ax4.fill_between(days, 0, durations, alpha=0.3, color=self.colors["purple"])
        ax4.set_xlabel("Day")
        ax4.set_ylabel("Duration (years)")
        ax4.set_title("Duration Evolution", fontweight="bold")
        ax4.grid(True, alpha=0.3)

        # 5. Rate evolution (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(
            days, rates, "o-", color=self.colors["warning"], linewidth=2, markersize=5
        )
        ax5.fill_between(days, 0, rates, alpha=0.3, color=self.colors["warning"])
        ax5.set_xlabel("Day")
        ax5.set_ylabel("Rate (%)")
        ax5.set_title("Rate Evolution", fontweight="bold")
        ax5.grid(True, alpha=0.3)

        # 6. Daily P&L (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        colors = [
            self.colors["positive"] if p >= 0 else self.colors["negative"]
            for p in daily_pnl
        ]
        ax6.bar(days, daily_pnl, color=colors, alpha=0.7, edgecolor="black")
        ax6.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax6.set_xlabel("Day")
        ax6.set_ylabel("Daily P&L ($)")
        ax6.set_title("Daily P&L", fontweight="bold")
        ax6.grid(True, alpha=0.3, axis="y")

        # 7. Convexity (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.plot(
            days,
            convexities,
            "o-",
            color=self.colors["secondary"],
            linewidth=2,
            markersize=5,
        )
        ax7.fill_between(
            days, 0, convexities, alpha=0.3, color=self.colors["secondary"]
        )
        ax7.set_xlabel("Day")
        ax7.set_ylabel("Convexity")
        ax7.set_title("Convexity Evolution", fontweight="bold")
        ax7.grid(True, alpha=0.3)

        # 8. Drawdown (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        running_max = []
        drawdowns = []
        current_max = results.baseline_value
        for v in values:
            current_max = max(current_max, v)
            running_max.append(current_max)
            drawdowns.append((v - current_max) / current_max * 100)
        ax8.fill_between(days, 0, drawdowns, color=self.colors["negative"], alpha=0.5)
        ax8.plot(
            days,
            drawdowns,
            "o-",
            color=self.colors["negative"],
            linewidth=2,
            markersize=5,
        )
        ax8.set_xlabel("Day")
        ax8.set_ylabel("Drawdown (%)")
        ax8.set_title("Drawdown", fontweight="bold")
        ax8.grid(True, alpha=0.3)

        plt.suptitle(
            f"FI Dynamic Scenario Dashboard - {results.path_name}",
            fontsize=16,
            fontweight="bold",
            y=1.01,
        )

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved FI dashboard to: {filepath}")
        else:
            plt.show()

        plt.close()

    def create_fi_all_plots(
        self,
        results: "FIDynamicScenarioResults",
        output_dir: Union[str, Path],
        prefix: Optional[str] = None,
    ) -> List[str]:
        """
        Create all FI-specific plots.

        Args:
            results: FI Dynamic scenario results
            output_dir: Output directory for plots
            prefix: Prefix for filenames (default: path name)

        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            prefix = results.path_name.replace(" ", "_").replace("/", "_")

        created_files = []

        print(f"Generating FI plots in {output_dir}...")

        # Portfolio value and P&L (reuse standard methods)
        self.plot_portfolio_value(results, output_dir / f"{prefix}_portfolio_value.png")
        created_files.append(str(output_dir / f"{prefix}_portfolio_value.png"))

        self.plot_pnl_evolution(results, output_dir / f"{prefix}_pnl_evolution.png")
        created_files.append(str(output_dir / f"{prefix}_pnl_evolution.png"))

        # FI-specific plots
        self.plot_dv01_evolution(results, output_dir / f"{prefix}_dv01_evolution.png")
        created_files.append(str(output_dir / f"{prefix}_dv01_evolution.png"))

        self.plot_duration_evolution(
            results, output_dir / f"{prefix}_duration_evolution.png"
        )
        created_files.append(str(output_dir / f"{prefix}_duration_evolution.png"))

        self.plot_rate_evolution(results, output_dir / f"{prefix}_rate_evolution.png")
        created_files.append(str(output_dir / f"{prefix}_rate_evolution.png"))

        self.plot_drawdown(results, output_dir / f"{prefix}_drawdown.png")
        created_files.append(str(output_dir / f"{prefix}_drawdown.png"))

        # FI Dashboard
        self.plot_fi_risk_dashboard(results, output_dir / f"{prefix}_fi_dashboard.png")
        created_files.append(str(output_dir / f"{prefix}_fi_dashboard.png"))

        print(f"All FI plots generated successfully! ({len(created_files)} files)")

        return created_files
