"""
Report generation for backtest results.

Creates comprehensive HTML and text reports with metrics and visualizations.
"""
from typing import Optional
from pathlib import Path
from datetime import datetime
import pandas as pd


class ReportGenerator:
    """
    Generate comprehensive reports for backtest results.
    
    Creates HTML reports with:
    - Executive summary
    - Strategy parameters
    - Performance metrics table
    - Key visualizations (embedded as images or HTML)
    - Trade log summary
    
    Attributes:
        results: BacktestResults instance
        output_dir: Directory for report output
    """
    
    def __init__(self, results: 'BacktestResults', output_dir: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            results: BacktestResults instance
            output_dir: Output directory for reports
        """
        self.results = results
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(
        self,
        filename: Optional[str] = None,
        include_plots: bool = True
    ) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            filename: Output filename (auto-generated if None)
            include_plots: Whether to include embedded plots
            
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{self.results.config.underlying}_{timestamp}.html"
        
        report_path = self.output_dir / filename
        
        # Generate HTML content
        html_content = self._generate_html_content(include_plots)
        
        # Write to file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_html_content(self, include_plots: bool) -> str:
        """Generate HTML report content."""
        # Get summary and metrics
        summary = self.results.get_summary()
        metrics = self.results.metrics.calculate_all_metrics()
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {self.results.config.underlying}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_executive_summary(summary)}
        {self._generate_strategy_section(summary)}
        {self._generate_metrics_section(metrics)}
        {self._generate_trade_summary()}
        {self._generate_footer()}
    </div>
</body>
</html>
"""
        return html
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            margin: -20px -20px 30px -20px;
            border-radius: 5px 5px 0 0;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .section h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .summary-card {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 5px;
        }
        
        .summary-card h3 {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        
        .summary-card.positive .value {
            color: #28a745;
        }
        
        .summary-card.negative .value {
            color: #dc3545;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        .metric-row td:last-child {
            font-weight: bold;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #ddd;
            margin-top: 40px;
        }
        
        .positive {
            color: #28a745;
        }
        
        .negative {
            color: #dc3545;
        }
        """
    
    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""
        <div class="header">
            <h1>Backtest Report</h1>
            <p>Strategy: {self.results.config.strategy.name} | Underlying: {self.results.config.underlying}</p>
            <p>Period: {self.results.config.start_date.strftime('%Y-%m-%d')} to {self.results.config.end_date.strftime('%Y-%m-%d')}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
    
    def _generate_executive_summary(self, summary: dict) -> str:
        """Generate executive summary section."""
        total_return = summary.get('total_return', 0)
        total_pnl = summary.get('total_pnl', 0)
        
        return_class = "positive" if total_return >= 0 else "negative"
        pnl_class = "positive" if total_pnl >= 0 else "negative"
        
        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card {return_class}">
                    <h3>Total Return</h3>
                    <div class="value">{total_return:.2%}</div>
                </div>
                <div class="summary-card {pnl_class}">
                    <h3>Total P&L</h3>
                    <div class="value">${total_pnl:,.2f}</div>
                </div>
                <div class="summary-card">
                    <h3>Initial Value</h3>
                    <div class="value">${summary.get('initial_value', 0):,.2f}</div>
                </div>
                <div class="summary-card">
                    <h3>Final Value</h3>
                    <div class="value">${summary.get('final_value', 0):,.2f}</div>
                </div>
                <div class="summary-card">
                    <h3>Number of Hedges</h3>
                    <div class="value">{summary.get('num_hedges', 0)}</div>
                </div>
                <div class="summary-card">
                    <h3>Transaction Costs</h3>
                    <div class="value">${summary.get('total_transaction_costs', 0):,.2f}</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_strategy_section(self, summary: dict) -> str:
        """Generate strategy parameters section."""
        strategy_params = summary.get('strategy', {})
        
        rows = ""
        for key, value in strategy_params.items():
            rows += f"""
            <tr>
                <td>{key.replace('_', ' ').title()}</td>
                <td>{value}</td>
            </tr>
            """
        
        return f"""
        <div class="section">
            <h2>Strategy Configuration</h2>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_metrics_section(self, metrics: dict) -> str:
        """Generate performance metrics section."""
        # Group metrics
        pnl_metrics = {
            'Total P&L': f"${metrics.get('total_pnl', 0):,.2f}",
            'Total Return': f"{metrics.get('total_return_pct', 0):.2f}%",
            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
            'Max Drawdown': f"{metrics.get('max_drawdown_pct', 0):.2f}%",
            'Win Rate': f"{metrics.get('win_rate', 0):.2%}",
            'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}",
        }
        
        hedge_metrics = {
            'Number of Hedges': f"{metrics.get('num_hedges', 0)}",
            'Hedge Frequency (per day)': f"{metrics.get('hedge_frequency', 0):.3f}",
            'Avg Hedge Cost': f"${metrics.get('avg_hedge_cost', 0):.2f}",
            'Total Hedge Costs': f"${metrics.get('total_hedge_costs', 0):,.2f}",
            'Delta Tracking Error': f"{metrics.get('delta_tracking_error', 0):.4f}",
            'Avg Absolute Delta': f"{metrics.get('avg_abs_delta', 0):.2f}",
        }
        
        risk_metrics = {
            'Volatility (annualized)': f"{metrics.get('volatility', 0):.2%}",
            'VaR (95%)': f"{metrics.get('var_95', 0):.2%}",
            'CVaR (95%)': f"{metrics.get('cvar_95', 0):.2%}",
            'Skewness': f"{metrics.get('skewness', 0):.3f}",
            'Kurtosis': f"{metrics.get('kurtosis', 0):.3f}",
        }
        
        def create_metrics_table(title: str, metrics_dict: dict) -> str:
            rows = ""
            for key, value in metrics_dict.items():
                rows += f'<tr class="metric-row"><td>{key}</td><td>{value}</td></tr>'
            return f"""
            <h3>{title}</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            """
        
        return f"""
        <div class="section">
            <h2>Performance Metrics</h2>
            {create_metrics_table("P&L Metrics", pnl_metrics)}
            {create_metrics_table("Hedging Metrics", hedge_metrics)}
            {create_metrics_table("Risk Metrics", risk_metrics)}
        </div>
        """
    
    def _generate_trade_summary(self) -> str:
        """Generate trade summary section."""
        trades_df = self.results.get_hedge_trades()
        
        if len(trades_df) == 0:
            return """
            <div class="section">
                <h2>Trade Summary</h2>
                <p>No trades executed during backtest.</p>
            </div>
            """
        
        # Summary statistics
        total_trades = len(trades_df)
        total_notional = trades_df['notional'].sum()
        avg_notional = trades_df['notional'].mean()
        
        # Recent trades (last 10)
        recent_trades = trades_df.tail(10)
        
        rows = ""
        for idx, row in recent_trades.iterrows():
            direction = "BUY" if row['quantity'] > 0 else "SELL"
            dir_class = "positive" if row['quantity'] > 0 else "negative"
            rows += f"""
            <tr>
                <td>{idx.strftime('%Y-%m-%d %H:%M:%S')}</td>
                <td class="{dir_class}">{direction}</td>
                <td>{abs(row['quantity']):.2f}</td>
                <td>${row['price']:.2f}</td>
                <td>${row['notional']:,.2f}</td>
                <td>${row['transaction_cost']:.2f}</td>
            </tr>
            """
        
        return f"""
        <div class="section">
            <h2>Trade Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Trades</h3>
                    <div class="value">{total_trades}</div>
                </div>
                <div class="summary-card">
                    <h3>Total Notional</h3>
                    <div class="value">${total_notional:,.0f}</div>
                </div>
                <div class="summary-card">
                    <h3>Avg Notional</h3>
                    <div class="value">${avg_notional:,.0f}</div>
                </div>
            </div>
            
            <h3>Recent Trades (Last 10)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Direction</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>Notional</th>
                        <th>Cost</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""
        <div class="footer">
            <p>Report generated by QuantArk Backtest Module</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
    
    def generate_text_report(self, filename: Optional[str] = None) -> str:
        """
        Generate simple text report.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{self.results.config.underlying}_{timestamp}.txt"
        
        report_path = self.output_dir / filename
        
        summary = self.results.get_summary()
        metrics = self.results.metrics.calculate_all_metrics()
        
        # Generate text content
        lines = []
        lines.append("=" * 80)
        lines.append("BACKTEST REPORT".center(80))
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Strategy: {self.results.config.strategy.name}")
        lines.append(f"Underlying: {self.results.config.underlying}")
        lines.append(f"Period: {self.results.config.start_date.date()} to {self.results.config.end_date.date()}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        lines.append("-" * 80)
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Return:        {summary.get('total_return', 0):>12.2%}")
        lines.append(f"Total P&L:           ${summary.get('total_pnl', 0):>12,.2f}")
        lines.append(f"Initial Value:       ${summary.get('initial_value', 0):>12,.2f}")
        lines.append(f"Final Value:         ${summary.get('final_value', 0):>12,.2f}")
        lines.append(f"Number of Hedges:    {summary.get('num_hedges', 0):>12}")
        lines.append(f"Transaction Costs:   ${summary.get('total_transaction_costs', 0):>12,.2f}")
        lines.append("")
        
        lines.append("-" * 80)
        lines.append("PERFORMANCE METRICS")
        lines.append("-" * 80)
        lines.append(f"Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):>12.3f}")
        lines.append(f"Max Drawdown:        {metrics.get('max_drawdown_pct', 0):>12.2f}%")
        lines.append(f"Win Rate:            {metrics.get('win_rate', 0):>12.2%}")
        lines.append(f"Profit Factor:       {metrics.get('profit_factor', 0):>12.2f}")
        lines.append(f"Volatility:          {metrics.get('volatility', 0):>12.2%}")
        lines.append(f"VaR (95%):           {metrics.get('var_95', 0):>12.2%}")
        lines.append("")
        
        lines.append("-" * 80)
        lines.append("HEDGING METRICS")
        lines.append("-" * 80)
        lines.append(f"Hedge Frequency:     {metrics.get('hedge_frequency', 0):>12.3f} per day")
        lines.append(f"Avg Hedge Cost:      ${metrics.get('avg_hedge_cost', 0):>12.2f}")
        lines.append(f"Delta Tracking Err:  {metrics.get('delta_tracking_error', 0):>12.4f}")
        lines.append(f"Avg Abs Delta:       {metrics.get('avg_abs_delta', 0):>12.2f}")
        lines.append("")
        
        lines.append("=" * 80)
        
        # Write to file
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        return str(report_path)
    
    def __repr__(self) -> str:
        return f"ReportGenerator(underlying={self.results.config.underlying})"

