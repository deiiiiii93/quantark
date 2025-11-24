"""
Visualization utilities for stress test results.
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

from stresstest.results.stress_results import StressTestResults
from stresstest.results.result_aggregator import ResultAggregator


class StressTestVisualizer:
    """
    Creates visualizations for stress test results.
    
    Supports both static (matplotlib) and interactive (plotly) visualizations.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
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
    
    def plot_pnl_waterfall(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 6)
    ) -> None:
        """
        Create waterfall chart of P&L across scenarios.
        
        Args:
            results: Stress test results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        scenarios = ['Baseline'] + [r.scenario.name for r in results.scenario_results]
        values = [0] + [r.portfolio_pnl for r in results.scenario_results]
        
        # Create waterfall
        colors = ['green' if v >= 0 else 'red' for v in values]
        colors[0] = 'blue'  # Baseline
        
        ax.bar(scenarios, values, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('P&L ($)')
        ax.set_title('Scenario P&L Waterfall')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved waterfall chart to: {filepath}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_pnl_distribution(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Create histogram of P&L distribution.
        
        Args:
            results: Stress test results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        pnl_pcts = [r.portfolio_pnl_pct for r in results.scenario_results]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(pnl_pcts, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(x=0, color='red', linestyle='--', label='Baseline')
        ax.set_xlabel('P&L (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('P&L Distribution Across Scenarios')
        ax.legend()
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved distribution plot to: {filepath}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_scenario_comparison(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 8)
    ) -> None:
        """
        Create horizontal bar chart comparing scenarios.
        
        Args:
            results: Stress test results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        df = ResultAggregator.compare_scenarios(results, metric='portfolio_pnl')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['red' if pnl < 0 else 'green' for pnl in df['portfolio_pnl']]
        ax.barh(df['scenario'], df['portfolio_pnl'], color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('P&L ($)')
        ax.set_title('Scenario Comparison')
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved scenario comparison to: {filepath}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_greeks_comparison(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None,
        figsize: tuple = (14, 8)
    ) -> None:
        """
        Create comparison chart of Greeks across scenarios.
        
        Args:
            results: Stress test results
            filepath: Optional path to save figure
            figsize: Figure size
        """
        greeks_df = ResultAggregator.get_greeks_comparison(results)
        
        if greeks_df is None:
            print("No Greeks data available to plot")
            return
        
        # Remove scenario column for plotting
        scenarios = greeks_df['scenario'].tolist()
        greeks_df = greeks_df.drop('scenario', axis=1)
        
        # Create subplots for each Greek
        n_greeks = len(greeks_df.columns)
        n_cols = 2
        n_rows = (n_greeks + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, greek in enumerate(greeks_df.columns):
            ax = axes[i]
            values = greeks_df[greek].tolist()
            colors = ['blue'] + ['red' if v < 0 else 'green' for v in values[1:]]
            
            ax.barh(scenarios, values, color=colors, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel(greek.capitalize())
            ax.set_title(f'{greek.capitalize()} by Scenario')
        
        # Hide unused subplots
        for i in range(n_greeks, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved Greeks comparison to: {filepath}")
        else:
            plt.show()
        
        plt.close()
    
    def create_interactive_dashboard(
        self,
        results: StressTestResults,
        filepath: Optional[Union[str, Path]] = None
    ) -> go.Figure:
        """
        Create interactive Plotly dashboard.
        
        Args:
            results: Stress test results
            filepath: Optional path to save HTML
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Scenario P&L Comparison',
                'P&L Distribution',
                'Portfolio Value by Scenario',
                'Risk Metrics'
            ),
            specs=[
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # Data preparation
        scenarios = [r.scenario.name for r in results.scenario_results]
        pnls = [r.portfolio_pnl for r in results.scenario_results]
        pnl_pcts = [r.portfolio_pnl_pct for r in results.scenario_results]
        values = [r.portfolio_value for r in results.scenario_results]
        
        # 1. P&L comparison
        colors = ['red' if p < 0 else 'green' for p in pnls]
        fig.add_trace(
            go.Bar(x=scenarios, y=pnls, marker_color=colors, name='P&L'),
            row=1, col=1
        )
        
        # 2. P&L distribution
        fig.add_trace(
            go.Histogram(x=pnl_pcts, nbinsx=20, marker_color='steelblue', name='Distribution'),
            row=1, col=2
        )
        
        # 3. Portfolio values
        fig.add_trace(
            go.Bar(x=scenarios, y=values, marker_color='lightblue', name='Portfolio Value'),
            row=2, col=1
        )
        
        # 4. Risk metrics table
        risk_summary = ResultAggregator.get_risk_summary(results)
        table_data = [
            ['Metric', 'Value'],
            ['Baseline Value', f"${risk_summary['baseline_value']:,.2f}"],
            ['Worst Scenario', risk_summary['worst_scenario']],
            ['Worst P&L', f"${risk_summary['worst_pnl']:,.2f} ({risk_summary['worst_pnl_pct']:.2f}%)"],
            ['Best Scenario', risk_summary['best_scenario']],
            ['Best P&L', f"${risk_summary['best_pnl']:,.2f} ({risk_summary['best_pnl_pct']:.2f}%)"],
            ['Average P&L', f"${risk_summary['avg_pnl']:,.2f}"],
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['<b>Metric</b>', '<b>Value</b>'],
                           fill_color='lightgray',
                           align='left'),
                cells=dict(values=[[row[0] for row in table_data[1:]], 
                                  [row[1] for row in table_data[1:]]],
                          fill_color='white',
                          align='left')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Stress Test Results Dashboard",
            showlegend=False,
            height=900
        )
        
        if filepath:
            fig.write_html(filepath)
            print(f"Saved interactive dashboard to: {filepath}")
        
        return fig
    
    def create_all_plots(
        self,
        results: StressTestResults,
        output_dir: Union[str, Path],
        prefix: str = "stress"
    ) -> None:
        """
        Create all standard plots.
        
        Args:
            results: Stress test results
            output_dir: Output directory for plots
            prefix: Prefix for filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating plots in {output_dir}...")
        
        # Static plots
        self.plot_pnl_waterfall(results, output_dir / f"{prefix}_pnl_waterfall.png")
        self.plot_pnl_distribution(results, output_dir / f"{prefix}_pnl_distribution.png")
        self.plot_scenario_comparison(results, output_dir / f"{prefix}_scenario_comparison.png")
        
        if results.baseline_greeks:
            self.plot_greeks_comparison(results, output_dir / f"{prefix}_greeks_comparison.png")
        
        # Interactive dashboard
        self.create_interactive_dashboard(results, output_dir / f"{prefix}_interactive_dashboard.html")
        
        print(f"All plots generated successfully!")

