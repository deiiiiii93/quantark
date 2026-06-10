"""Visualization utilities for stress test results."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from quantark.stresstest.equity.results import StressTestResults
from quantark.stresstest.results.result_aggregator import ResultAggregator


class StressTestVisualizer:
    """
    Creates visualizations for stress test results.

    Supports both static (matplotlib) and interactive (plotly) visualizations.
    """

    def __init__(self, style: str = "seaborn-v0_8"):
        self.style = style
        try:
            plt.style.use(style)
        except Exception:
            pass
        sns.set_palette("husl")

    def plot_pnl_waterfall(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 6),
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)

        scenarios = ["Baseline"] + [r.scenario.name for r in results.scenario_results]
        values = [0] + [r.portfolio_pnl for r in results.scenario_results]
        colors = ["blue"] + ["green" if v >= 0 else "red" for v in values[1:]]

        ax.bar(scenarios, values, color=colors, alpha=0.7, edgecolor="black")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_ylabel("P&L ($)")
        ax.set_title("Scenario P&L Waterfall")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved waterfall chart to: {filepath}")
        else:
            plt.show()
        plt.close()

    def plot_pnl_distribution(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (10, 6),
    ) -> None:
        pnl_pcts = [r.portfolio_pnl_pct for r in results.scenario_results]
        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(pnl_pcts, bins=20, alpha=0.7, edgecolor="black", color="steelblue")
        ax.axvline(x=0, color="red", linestyle="--", label="Baseline")
        ax.set_xlabel("P&L (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("P&L Distribution Across Scenarios")
        ax.legend()

        plt.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved distribution plot to: {filepath}")
        else:
            plt.show()
        plt.close()

    def plot_scenario_comparison(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 8),
    ) -> None:
        df = ResultAggregator.compare_scenarios(results, metric="portfolio_pnl")
        fig, ax = plt.subplots(figsize=figsize)

        colors = ["red" if pnl < 0 else "green" for pnl in df["portfolio_pnl"]]
        ax.barh(df["scenario"], df["portfolio_pnl"], color=colors, alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("P&L ($)")
        ax.set_title("Scenario Comparison")

        plt.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved scenario comparison to: {filepath}")
        else:
            plt.show()
        plt.close()

    def plot_greeks_comparison(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (14, 8),
    ) -> None:
        greeks_df = ResultAggregator.get_greeks_comparison(results)

        if greeks_df is None:
            print("No Greeks data available to plot")
            return

        scenarios = greeks_df["scenario"].tolist()
        greeks_df = greeks_df.drop("scenario", axis=1)

        n_greeks = len(greeks_df.columns)
        n_cols = 2
        n_rows = (n_greeks + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, greek in enumerate(greeks_df.columns):
            ax = axes[i]
            values = greeks_df[greek].tolist()
            colors = ["blue"] + ["red" if v < 0 else "green" for v in values[1:]]

            ax.barh(scenarios, values, color=colors, alpha=0.7, edgecolor="black")
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
            ax.set_xlabel(greek.capitalize())
            ax.set_title(f"{greek.capitalize()} by Scenario")

        for i in range(n_greeks, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved Greeks comparison to: {filepath}")
        else:
            plt.show()
        plt.close()

    def create_interactive_dashboard(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None,
    ) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Scenario P&L Comparison",
                "P&L Distribution",
                "Portfolio Value by Scenario",
                "Risk Metrics",
            ),
            specs=[[{"type": "bar"}, {"type": "histogram"}], [{"type": "bar"}, {"type": "table"}]],
        )

        scenarios = [r.scenario.name for r in results.scenario_results]
        pnls = [r.portfolio_pnl for r in results.scenario_results]
        pnl_pcts = [r.portfolio_pnl_pct for r in results.scenario_results]
        values = [r.portfolio_value for r in results.scenario_results]

        colors = ["red" if p < 0 else "green" for p in pnls]
        fig.add_trace(go.Bar(x=scenarios, y=pnls, marker_color=colors, name="P&L"), row=1, col=1)

        fig.add_trace(
            go.Histogram(x=pnl_pcts, nbinsx=20, marker_color="steelblue", name="Distribution"),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Bar(x=scenarios, y=values, marker_color="lightblue", name="Portfolio Value"),
            row=2,
            col=1,
        )

        risk_summary = ResultAggregator.get_risk_summary(results)
        table_data = [
            ["Metric", "Value"],
            ["Baseline Value", f"${risk_summary['baseline_value']:,.2f}"],
            ["Worst Scenario", risk_summary["worst_scenario"]],
            [
                "Worst P&L",
                f"${risk_summary['worst_pnl']:,.2f} ({risk_summary['worst_pnl_pct']:.2f}%)",
            ],
            ["Best Scenario", risk_summary["best_scenario"]],
            ["Best P&L", f"${risk_summary['best_pnl']:,.2f} ({risk_summary['best_pnl_pct']:.2f}%)"],
            ["Average P&L", f"${risk_summary['avg_pnl']:,.2f}"],
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=["<b>Metric</b>", "<b>Value</b>"], fill_color="lightgray", align="left"),
                cells=dict(
                    values=[[row[0] for row in table_data[1:]], [row[1] for row in table_data[1:]]],
                    fill_color="white",
                    align="left",
                ),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text="Stress Test Results Dashboard",
            showlegend=False,
            height=900,
        )

        if filepath:
            fig.write_html(filepath)
            print(f"Saved interactive dashboard to: {filepath}")

        return fig

    def create_all_plots(
        self,
        results: StressTestResults,
        output_dir: Union[str, Path],
        prefix: str = "stress",
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating plots in {output_dir}...")

        self.plot_pnl_waterfall(results, output_dir / f"{prefix}_pnl_waterfall.png")
        self.plot_pnl_distribution(results, output_dir / f"{prefix}_pnl_distribution.png")
        self.plot_scenario_comparison(results, output_dir / f"{prefix}_scenario_comparison.png")

        if results.baseline_greeks:
            self.plot_greeks_comparison(results, output_dir / f"{prefix}_greeks_comparison.png")

        self.create_interactive_dashboard(results, output_dir / f"{prefix}_interactive_dashboard.html")

        if self._has_fi_metrics(results):
            self.plot_fi_dv01_waterfall(
                results, output_dir / f"{prefix}_fi_dv01_waterfall.png"
            )
            self.plot_curve_shift_heatmap(
                results, output_dir / f"{prefix}_curve_shift.png"
            )

        print("All plots generated successfully!")

    def _has_fi_metrics(self, results: StressTestResults) -> bool:
        get_series = getattr(results, "get_dv01_series", None)
        return callable(get_series)

    def plot_fi_dv01_waterfall(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 6),
    ) -> None:
        get_series = getattr(results, "get_dv01_series", None)
        if not callable(get_series):
            return
        dv01_df = get_series()
        if dv01_df is None or dv01_df.empty:
            return

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=dv01_df, x="scenario", y="dv01", ax=ax, palette="coolwarm")
        ax.set_ylabel("DV01 ($)")
        ax.set_title("DV01 by Scenario")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved FI DV01 waterfall to: {filepath}")
        else:
            plt.show()
        plt.close()

    def plot_curve_shift_heatmap(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (10, 6),
    ) -> None:
        getter = getattr(results, "get_curve_shift_summary", None)
        if not callable(getter):
            return

        rows: List[Dict[str, Any]] = []
        for scenario in results.scenario_results:
            summary = getter(scenario.scenario.name) or {}
            for underlying, values in summary.items():
                rate_info = values.get("rate")
                if rate_info:
                    rows.append(
                        {
                            "scenario": scenario.scenario.name,
                            "underlying": underlying,
                            "change_bps": rate_info.get("change_bps", 0.0),
                        }
                    )

        if not rows:
            return

        df = pd.DataFrame(rows)
        pivot = df.pivot_table(
            index="scenario", columns="underlying", values="change_bps", fill_value=0.0
        )

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Curve Shift Heatmap (bps)")
        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved curve shift heatmap to: {filepath}")
        else:
            plt.show()
        plt.close()

