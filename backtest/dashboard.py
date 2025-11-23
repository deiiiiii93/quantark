"""
Interactive dashboard using Plotly for backtest results.
"""
from typing import Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path


class InteractiveDashboard:
    """
    Create interactive visualizations using Plotly.
    
    Provides interactive plots with:
    - Hover details
    - Zoom/pan capabilities
    - Toggleable series
    - Export to HTML
    
    Attributes:
        results: BacktestResults instance
        save_dir: Directory to save HTML files
    """
    
    def __init__(self, results: 'BacktestResults', save_dir: Optional[str] = None):
        """
        Initialize interactive dashboard.
        
        Args:
            results: BacktestResults instance
            save_dir: Directory to save HTML files
        """
        self.results = results
        self.save_dir = Path(save_dir) if save_dir else Path("plots/interactive")
        
        if save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_pnl_interactive(
        self,
        save: bool = False,
        filename: str = "pnl_interactive.html"
    ) -> go.Figure:
        """
        Create interactive P&L plot.
        
        Args:
            save: Whether to save to HTML
            filename: Filename if saving
            
        Returns:
            Plotly figure
        """
        pnl_series = self.results.get_pnl_series()
        
        fig = go.Figure()
        
        # Add P&L line
        fig.add_trace(go.Scatter(
            x=pnl_series.index,
            y=pnl_series.values,
            mode='lines',
            name='P&L',
            line=dict(color='steelblue', width=2),
            hovertemplate='<b>Date</b>: %{x}<br><b>P&L</b>: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add trades markers
        trades_df = self.results.get_hedge_trades()
        if len(trades_df) > 0:
            # Get P&L at trade times
            trade_pnls = []
            for trade_time in trades_df.index:
                closest_idx = pnl_series.index.get_indexer([trade_time], method='nearest')[0]
                if closest_idx >= 0:
                    trade_pnls.append(pnl_series.iloc[closest_idx])
                else:
                    trade_pnls.append(0)
            
            fig.add_trace(go.Scatter(
                x=trades_df.index,
                y=trade_pnls,
                mode='markers',
                name='Hedge Trades',
                marker=dict(
                    size=8,
                    color=trades_df['quantity'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Quantity"),
                    line=dict(width=1, color='black')
                ),
                hovertemplate='<b>Trade</b><br>Time: %{x}<br>Quantity: %{marker.color:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Interactive P&L - {self.results.config.underlying}',
            xaxis_title='Date',
            yaxis_title='P&L ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        if save and self.save_dir:
            fig.write_html(self.save_dir / filename)
        
        return fig
    
    def plot_greeks_interactive(
        self,
        greeks: Optional[List[str]] = None,
        save: bool = False,
        filename: str = "greeks_interactive.html"
    ) -> go.Figure:
        """
        Create interactive Greeks plots.
        
        Args:
            greeks: List of Greeks to plot
            save: Whether to save
            filename: Filename if saving
            
        Returns:
            Plotly figure
        """
        if greeks is None:
            greeks = ['delta', 'gamma', 'vega', 'theta']
        
        greeks_df = self.results.get_greeks_series()
        
        # Filter available Greeks
        available_greeks = []
        for greek in greeks:
            col_name = f'greek_{greek}'
            if col_name in greeks_df.columns:
                available_greeks.append(greek)
        
        if not available_greeks:
            print("No Greeks data available")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=len(available_greeks),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[g.capitalize() for g in available_greeks]
        )
        
        colors = ['steelblue', 'crimson', 'purple', 'orange', 'green']
        
        for i, greek in enumerate(available_greeks):
            col_name = f'greek_{greek}'
            data = greeks_df[col_name]
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines',
                    name=greek.capitalize(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{greek.capitalize()}</b>: %{{y:.4f}}<extra></extra>'
                ),
                row=i+1,
                col=1
            )
            
            # Add zero line
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                opacity=0.3,
                row=i+1,
                col=1
            )
        
        fig.update_layout(
            title=f'Portfolio Greeks Evolution - {self.results.config.underlying}',
            hovermode='x unified',
            template='plotly_white',
            height=250 * len(available_greeks),
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Date", row=len(available_greeks), col=1)
        
        if save and self.save_dir:
            fig.write_html(self.save_dir / filename)
        
        return fig
    
    def plot_delta_tracking_interactive(
        self,
        save: bool = False,
        filename: str = "delta_tracking_interactive.html"
    ) -> go.Figure:
        """
        Create interactive delta tracking plot.
        
        Args:
            save: Whether to save
            filename: Filename if saving
            
        Returns:
            Plotly figure
        """
        delta_series = self.results.get_delta_series()
        target_delta = self.results.config.strategy.target_delta
        threshold = self.results.config.strategy.delta_threshold
        
        fig = go.Figure()
        
        # Actual delta
        fig.add_trace(go.Scatter(
            x=delta_series.index,
            y=delta_series.values,
            mode='lines',
            name='Actual Delta',
            line=dict(color='steelblue', width=2),
            hovertemplate='<b>Delta</b>: %{y:.2f}<extra></extra>'
        ))
        
        # Target delta
        fig.add_hline(
            y=target_delta,
            line_dash="dash",
            line_color="green",
            annotation_text="Target",
            line_width=2
        )
        
        # Thresholds
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="red",
            annotation_text="Upper Threshold",
            opacity=0.7
        )
        fig.add_hline(
            y=-threshold,
            line_dash="dot",
            line_color="red",
            annotation_text="Lower Threshold",
            opacity=0.7
        )
        
        # Add threshold zones
        fig.add_hrect(
            y0=threshold,
            y1=delta_series.max() * 1.1,
            fillcolor="red",
            opacity=0.1,
            layer="below",
            line_width=0
        )
        fig.add_hrect(
            y0=-threshold,
            y1=delta_series.min() * 1.1,
            fillcolor="red",
            opacity=0.1,
            layer="below",
            line_width=0
        )
        
        # Add hedge trade markers
        trades_df = self.results.get_hedge_trades()
        if len(trades_df) > 0:
            trade_deltas = []
            for trade_time in trades_df.index:
                closest_idx = delta_series.index.get_indexer([trade_time], method='nearest')[0]
                if closest_idx >= 0:
                    trade_deltas.append(delta_series.iloc[closest_idx])
                else:
                    trade_deltas.append(0)
            
            fig.add_trace(go.Scatter(
                x=trades_df.index,
                y=trade_deltas,
                mode='markers',
                name='Hedge Trades',
                marker=dict(size=10, color='orange', symbol='diamond', line=dict(width=1, color='black')),
                hovertemplate='<b>Hedge Trade</b><br>Delta before: %{y:.2f}<extra></extra>'
            ))
        
        tracking_error = self.results.metrics.delta_tracking_error()
        
        fig.update_layout(
            title=f'Delta Tracking (Tracking Error: {tracking_error:.2f}) - {self.results.config.underlying}',
            xaxis_title='Date',
            yaxis_title='Delta',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        if save and self.save_dir:
            fig.write_html(self.save_dir / filename)
        
        return fig
    
    def create_comprehensive_dashboard(
        self,
        save: bool = False,
        filename: str = "comprehensive_dashboard.html"
    ) -> go.Figure:
        """
        Create comprehensive multi-panel dashboard.
        
        Args:
            save: Whether to save
            filename: Filename if saving
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                'P&L Over Time',
                'Portfolio Value',
                'Delta Tracking',
                'Returns Distribution',
                'Drawdown',
                'Hedge Trade Sizes'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # P&L
        pnl_series = self.results.get_pnl_series()
        fig.add_trace(
            go.Scatter(x=pnl_series.index, y=pnl_series.values,
                      mode='lines', name='P&L', line=dict(color='steelblue', width=2)),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        
        # Portfolio Value
        value_series = self.results.get_value_series()
        fig.add_trace(
            go.Scatter(x=value_series.index, y=value_series.values,
                      mode='lines', name='Value', line=dict(color='darkgreen', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=self.results.initial_value, line_dash="dash",
                     line_color="gray", opacity=0.5, row=1, col=2)
        
        # Delta
        delta_series = self.results.get_delta_series()
        fig.add_trace(
            go.Scatter(x=delta_series.index, y=delta_series.values,
                      mode='lines', name='Delta', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=self.results.config.strategy.target_delta,
                     line_dash="dash", line_color="green", row=2, col=1)
        
        # Returns distribution
        returns = self.results.metrics.returns_series
        if len(returns) > 0:
            fig.add_trace(
                go.Histogram(x=returns.values, name='Returns',
                           marker=dict(color='steelblue', line=dict(color='black', width=1)),
                           nbinsx=50),
                row=2, col=2
            )
        
        # Drawdown
        cumulative_max = value_series.expanding().max()
        drawdown = (value_series - cumulative_max) / cumulative_max
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      mode='lines', name='Drawdown',
                      fill='tozeroy', line=dict(color='darkred', width=2)),
            row=3, col=1
        )
        
        # Hedge sizes
        trades_df = self.results.get_hedge_trades()
        if len(trades_df) > 0:
            fig.add_trace(
                go.Bar(x=trades_df.index, y=trades_df['quantity'],
                      name='Hedge Size', marker=dict(color='orange')),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'Comprehensive Dashboard - {self.results.config.underlying}',
            showlegend=False,
            template='plotly_white',
            height=1200,
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="Value ($)", row=1, col=2)
        fig.update_yaxes(title_text="Delta", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Drawdown", row=3, col=1)
        fig.update_yaxes(title_text="Quantity", row=3, col=2)
        
        if save and self.save_dir:
            fig.write_html(self.save_dir / filename)
        
        return fig
    
    def plot_transaction_costs_breakdown(
        self,
        save: bool = False,
        filename: str = "transaction_costs.html"
    ) -> go.Figure:
        """
        Plot transaction costs over time.
        
        Args:
            save: Whether to save
            filename: Filename if saving
            
        Returns:
            Plotly figure
        """
        trades_df = self.results.get_hedge_trades()
        
        if len(trades_df) == 0:
            print("No trades to plot")
            return None
        
        # Calculate cumulative costs
        cumulative_costs = trades_df['transaction_cost'].cumsum()
        
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=('Cost Per Trade', 'Cumulative Transaction Costs'),
            vertical_spacing=0.15
        )
        
        # Cost per trade
        fig.add_trace(
            go.Bar(
                x=trades_df.index,
                y=trades_df['transaction_cost'],
                name='Cost per Trade',
                marker=dict(color='crimson'),
                hovertemplate='<b>Cost</b>: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Cumulative costs
        fig.add_trace(
            go.Scatter(
                x=cumulative_costs.index,
                y=cumulative_costs.values,
                mode='lines',
                name='Cumulative Costs',
                line=dict(color='darkred', width=2),
                fill='tozeroy',
                hovertemplate='<b>Total Cost</b>: $%{y:,.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Transaction Costs Analysis - {self.results.config.underlying}',
            showlegend=False,
            template='plotly_white',
            height=800
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Cost ($)", row=2, col=1)
        
        if save and self.save_dir:
            fig.write_html(self.save_dir / filename)
        
        return fig
    
    def generate_all_interactive_plots(self, save: bool = True):
        """
        Generate all interactive plots.
        
        Args:
            save: Whether to save all plots to HTML
            
        Returns:
            Dictionary of plot name to figure
        """
        plots = {}
        
        plots['pnl'] = self.plot_pnl_interactive(save=save)
        plots['greeks'] = self.plot_greeks_interactive(save=save)
        plots['delta_tracking'] = self.plot_delta_tracking_interactive(save=save)
        plots['dashboard'] = self.create_comprehensive_dashboard(save=save)
        plots['transaction_costs'] = self.plot_transaction_costs_breakdown(save=save)
        
        return plots
    
    def __repr__(self) -> str:
        return f"InteractiveDashboard(underlying={self.results.config.underlying})"

