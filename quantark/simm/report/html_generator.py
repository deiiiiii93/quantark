"""
SIMM HTML Report Generator.

This module generates HTML reports for SIMM results with interactive charts
and professional styling.
"""
from typing import Dict, List, Any, Optional
from datetime import date
import os

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..results.simm_result import SIMMResult
from ..results.attribution import SIMMAttribution
from ..taxonomy import ProductClass, RiskClass, MarginType


class SIMMHTMLReportGenerator:
    """Generate HTML reports for SIMM results."""

    def __init__(self, include_charts: bool = True):
        """Initialize report generator.

        Args:
            include_charts: Whether to include Plotly charts.
        """
        self.include_charts = include_charts and PLOTLY_AVAILABLE

        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available. Charts will be disabled.")

    def generate(
        self,
        result: SIMMResult,
        output_path: str,
        include_charts: bool = True,
        custom_css: Optional[str] = None,
        logo_path: Optional[str] = None,
    ) -> str:
        """Generate HTML report.

        Args:
            result: SIMMResult to report on.
            output_path: Output HTML file path.
            include_charts: Whether to include interactive charts.
            custom_css: Optional custom CSS to inject.
            logo_path: Optional path to logo image.

        Returns:
            Path to generated HTML file.
        """
        include_charts = include_charts and self.include_charts

        # Generate HTML sections
        html_parts = []

        # Header
        html_parts.append(self._render_header(result, logo_path))

        # Executive summary
        html_parts.append(self._render_summary_section(result))

        # Product class breakdown
        html_parts.append(self._render_product_class_breakdown(result, include_charts))

        # Risk class breakdown
        html_parts.append(self._render_risk_class_breakdown(result, include_charts))

        # Margin type breakdown
        html_parts.append(self._render_margin_type_breakdown(result, include_charts))

        # Bucket detail
        html_parts.append(self._render_bucket_detail(result))

        # Top contributors
        if result.attribution:
            html_parts.append(self._render_top_contributors(result))

        # Diversification analysis
        if result.attribution:
            html_parts.append(self._render_diversification_analysis(result))

        # Configuration and warnings
        html_parts.append(self._render_config_and_warnings(result))

        # Footer
        html_parts.append(self._render_footer())

        # Combine all sections
        html_content = "\n".join(html_parts)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _render_header(
        self,
        result: SIMMResult,
        logo_path: Optional[str] = None
    ) -> str:
        """Render report header.

        Args:
            result: SIMMResult to report on.
            logo_path: Optional logo path.

        Returns:
            HTML header string.
        """
        logo_html = f'<img src="{logo_path}" alt="Logo" class="logo"/>' if logo_path else ""

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIMM Report - {result.calculation_date}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <div class="header-content">
                {logo_html}
                <div class="header-text">
                    <h1>SIMM Calculation Report</h1>
                    <p class="subtitle">ISDA Standard Initial Margin Model v{result.simm_version}</p>
                </div>
            </div>
        </header>
"""

    def _render_summary_section(self, result: SIMMResult) -> str:
        """Render executive summary section.

        Args:
            result: SIMMResult to report on.

        Returns:
            HTML summary section.
        """
        currency_symbol = self._get_currency_symbol(result.calculation_currency)

        return f"""
        <section class="section">
            <h2>Executive Summary</h2>
            <div class="summary-cards">
                <div class="card primary">
                    <h3>Total SIMM</h3>
                    <p class="metric">{currency_symbol}{result.total_simm:,.2f}</p>
                    <p class="label">{result.calculation_currency}</p>
                </div>
                <div class="card">
                    <h3>Calculation Date</h3>
                    <p class="metric">{result.calculation_date}</p>
                </div>
                <div class="card">
                    <h3>Add-ons</h3>
                    <p class="metric">{currency_symbol}{result.addon_amount:,.2f}</p>
                </div>
                <div class="card">
                    <h3>Execution Time</h3>
                    <p class="metric">{result.execution_time_seconds:.3f}s</p>
                </div>
            </div>
        </section>
"""

    def _render_product_class_breakdown(
        self,
        result: SIMMResult,
        include_charts: bool
    ) -> str:
        """Render product class breakdown section.

        Args:
            result: SIMMResult to report on.
            include_charts: Whether to include charts.

        Returns:
            HTML product class breakdown section.
        """
        html = ['<section class="section">']
        html.append('<h2>Product Class Breakdown</h2>')

        # Table
        html.append('<table class="data-table">')
        html.append('<thead><tr><th>Product Class</th><th>SIMM Amount</th><th>% of Total</th></tr></thead>')
        html.append('<tbody>')

        for pc, amount in sorted(result.product_class_simm.items(), key=lambda x: x[1], reverse=True):
            pct = (amount / result.total_simm * 100) if result.total_simm != 0 else 0
            html.append(f"""
                <tr>
                    <td>{pc.value}</td>
                    <td class="number">{amount:,.2f}</td>
                    <td class="number">{pct:.2f}%</td>
                </tr>
            """)

        html.append('</tbody></table>')

        # Waterfall chart
        if include_charts:
            chart_div = self._create_product_class_waterfall(result)
            html.append(f'<div class="chart-container">{chart_div}</div>')

        html.append('</section>')
        return "\n".join(html)

    def _render_risk_class_breakdown(
        self,
        result: SIMMResult,
        include_charts: bool
    ) -> str:
        """Render risk class breakdown section.

        Args:
            result: SIMMResult to report on.
            include_charts: Whether to include charts.

        Returns:
            HTML risk class breakdown section.
        """
        html = ['<section class="section">']
        html.append('<h2>Risk Class Analysis</h2>')

        margin_by_risk = result.get_margin_by_risk_class()

        # Summary table
        html.append('<table class="data-table">')
        html.append('<thead><tr><th>Risk Class</th><th>Margin</th><th>% of Total</th></tr></thead>')
        html.append('<tbody>')

        for rc, amount in sorted(margin_by_risk.items(), key=lambda x: x[1], reverse=True):
            pct = (amount / result.total_simm * 100) if result.total_simm != 0 else 0
            html.append(f"""
                <tr>
                    <td>{rc.value}</td>
                    <td class="number">{amount:,.2f}</td>
                    <td class="number">{pct:.2f}%</td>
                </tr>
            """)

        html.append('</tbody></table>')

        # Detailed breakdown by product class
        html.append('<h3>Breakdown by Product Class</h3>')

        for pc in ProductClass:
            if pc not in result.risk_class_margin:
                continue

            html.append(f'<h4>{pc.value}</h4>')
            html.append('<table class="data-table">')
            html.append('<thead><tr><th>Risk Class</th><th>Delta</th><th>Vega</th><th>Curvature</th><th>Total</th></tr></thead>')
            html.append('<tbody>')

            for rc, rc_margin in result.risk_class_margin[pc].items():
                html.append(f"""
                    <tr>
                        <td>{rc.value}</td>
                        <td class="number">{rc_margin.delta_margin:,.2f}</td>
                        <td class="number">{rc_margin.vega_margin:,.2f}</td>
                        <td class="number">{rc_margin.curvature_margin:,.2f}</td>
                        <td class="number"><strong>{rc_margin.total_margin:,.2f}</strong></td>
                    </tr>
                """)

            html.append('</tbody></table>')

        # Risk class pie chart
        if include_charts:
            chart_div = self._create_risk_class_pie(margin_by_risk)
            html.append(f'<div class="chart-container">{chart_div}</div>')

        html.append('</section>')
        return "\n".join(html)

    def _render_margin_type_breakdown(
        self,
        result: SIMMResult,
        include_charts: bool
    ) -> str:
        """Render margin type breakdown section.

        Args:
            result: SIMMResult to report on.
            include_charts: Whether to include charts.

        Returns:
            HTML margin type breakdown section.
        """
        html = ['<section class="section">']
        html.append('<h2>Margin Type Distribution</h2>')

        margin_by_type = result.get_margin_by_margin_type()

        # Table
        html.append('<table class="data-table">')
        html.append('<thead><tr><th>Margin Type</th><th>Amount</th><th>% of Total</th></tr></thead>')
        html.append('<tbody>')

        for mt, amount in margin_by_type.items():
            pct = (amount / result.total_simm * 100) if result.total_simm != 0 else 0
            html.append(f"""
                <tr>
                    <td>{mt.value}</td>
                    <td class="number">{amount:,.2f}</td>
                    <td class="number">{pct:.2f}%</td>
                </tr>
            """)

        html.append('</tbody></table>')

        # Pie chart
        if include_charts:
            chart_div = self._create_margin_type_pie(margin_by_type)
            html.append(f'<div class="chart-container">{chart_div}</div>')

        html.append('</section>')
        return "\n".join(html)

    def _render_bucket_detail(self, result: SIMMResult) -> str:
        """Render bucket-level detail section.

        Args:
            result: SIMMResult to report on.

        Returns:
            HTML bucket detail section.
        """
        html = ['<section class="section">']
        html.append('<h2>Bucket Detail</h2>')

        for pc, rc_dict in result.risk_class_margin.items():
            html.append(f'<h3>{pc.value}</h3>')

            for rc, rc_margin in rc_dict.items():
                if not rc_margin.bucket_detail:
                    continue

                html.append(f'<h4>{rc.value}</h4>')
                html.append('<div class="expandable">')
                html.append('<table class="data-table">')
                html.append('<thead><tr><th>Bucket</th><th>K Value</th><th>WS Sum</th><th>Concentration</th></tr></thead>')
                html.append('<tbody>')

                for bucket, bucket_detail in sorted(
                    rc_margin.bucket_detail.items(),
                    key=lambda x: x[1].k_value,
                    reverse=True
                ):
                    html.append(f"""
                        <tr>
                            <td>{bucket}</td>
                            <td class="number">{bucket_detail.k_value:,.2f}</td>
                            <td class="number">{bucket_detail.ws_sum:,.2f}</td>
                            <td class="number">{bucket_detail.concentration_factor:.3f}</td>
                        </tr>
                    """)

                html.append('</tbody></table>')
                html.append('</div>')

        html.append('</section>')
        return "\n".join(html)

    def _render_top_contributors(self, result: SIMMResult) -> str:
        """Render top contributors section.

        Args:
            result: SIMMResult to report on.

        Returns:
            HTML top contributors section.
        """
        if not isinstance(result.attribution, SIMMAttribution):
            return ""

        html = ['<section class="section">']
        html.append('<h2>Top Contributors</h2>')

        html.append('<table class="data-table">')
        html.append('<thead><tr><th>Contributor</th><th>Type</th><th>Amount</th><th>% of Total</th></tr></thead>')
        html.append('<tbody>')

        for contributor in result.attribution.top_contributors[:20]:  # Top 20
            html.append(f"""
                <tr>
                    <td>{contributor.identifier}</td>
                    <td>{contributor.contribution_type}</td>
                    <td class="number">{contributor.amount:,.2f}</td>
                    <td class="number">{contributor.pct_of_total:.2f}%</td>
                </tr>
            """)

        html.append('</tbody></table>')
        html.append('</section>')

        return "\n".join(html)

    def _render_diversification_analysis(self, result: SIMMResult) -> str:
        """Render diversification analysis section.

        Args:
            result: SIMMResult to report on.

        Returns:
            HTML diversification analysis section.
        """
        if not isinstance(result.attribution, SIMMAttribution):
            return ""

        # Calculate standalone margins (sum of risk class margins)
        standalone_margins = result.get_margin_by_risk_class()
        div_benefit = result.attribution.get_diversification_benefit(standalone_margins)

        html = ['<section class="section">']
        html.append('<h2>Diversification Analysis</h2>')

        html.append(f"""
        <div class="summary-cards">
            <div class="card">
                <h3>Standalone Margins</h3>
                <p class="metric">{div_benefit['total_standalone']:,.2f}</p>
            </div>
            <div class="card">
                <h3>SIMM (Aggregated)</h3>
                <p class="metric">{div_benefit['total_simm']:,.2f}</p>
            </div>
            <div class="card primary">
                <h3>Diversification Benefit</h3>
                <p class="metric">{div_benefit['diversification_benefit']:,.2f}</p>
                <p class="label">{div_benefit['diversification_pct']:.2f}%</p>
            </div>
        </div>
        """)

        html.append('</section>')
        return "\n".join(html)

    def _render_config_and_warnings(self, result: SIMMResult) -> str:
        """Render configuration and warnings section.

        Args:
            result: SIMMResult to report on.

        Returns:
            HTML config and warnings section.
        """
        html = ['<section class="section">']
        html.append('<h2>Configuration & Metadata</h2>')

        # Configuration summary
        if result.config_summary:
            html.append('<h3>Configuration</h3>')
            html.append('<table class="data-table">')
            html.append('<tbody>')

            for key, value in result.config_summary.items():
                html.append(f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value}</td>
                    </tr>
                """)

            html.append('</tbody></table>')

        # Warnings
        if result.warnings:
            html.append('<h3>Warnings</h3>')
            html.append('<ul class="warnings">')

            for warning in result.warnings:
                html.append(f'<li class="warning">{warning}</li>')

            html.append('</ul>')

        html.append('</section>')
        return "\n".join(html)

    def _render_footer(self) -> str:
        """Render report footer.

        Returns:
            HTML footer string.
        """
        return """
        <footer class="report-footer">
            <p>Generated by QuantArk SIMM Module</p>
            <p class="timestamp">Generated on: {timestamp}</p>
        </footer>
    </div>
</body>
</html>
""".format(timestamp=date.today())

    def _create_product_class_waterfall(self, result: SIMMResult) -> str:
        """Create product class waterfall chart.

        Args:
            result: SIMMResult to chart.

        Returns:
            Plotly chart HTML div.
        """
        if not PLOTLY_AVAILABLE:
            return "<p>Charts not available (Plotly not installed)</p>"

        # Waterfall chart
        categories = [pc.value for pc in ProductClass]
        values = [result.product_class_simm.get(pc, 0) for pc in ProductClass]

        fig = go.Figure(go.Bar(
            x=categories,
            y=values,
            text=[f'{v:,.0f}' for v in values],
            textposition='auto',
            marker_color='rgba(55, 128, 191, 0.7)',
        ))

        fig.update_layout(
            title='SIMM by Product Class',
            xaxis_title='Product Class',
            yaxis_title='SIMM Amount',
            template='plotly_white',
            height=400,
        )

        return fig.to_html(include_plotlyjs='cdn', div_id="waterfall-chart")

    def _create_risk_class_pie(self, margin_by_risk: Dict[RiskClass, float]) -> str:
        """Create risk class pie chart.

        Args:
            margin_by_risk: Margin by risk class.

        Returns:
            Plotly chart HTML div.
        """
        if not PLOTLY_AVAILABLE:
            return "<p>Charts not available (Plotly not installed)</p>"

        labels = [rc.value for rc in margin_by_risk.keys()]
        values = list(margin_by_risk.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
        )])

        fig.update_layout(
            title='SIMM Distribution by Risk Class',
            template='plotly_white',
            height=400,
        )

        return fig.to_html(include_plotlyjs='cdn', div_id="pie-chart")

    def _create_margin_type_pie(self, margin_by_type: Dict[MarginType, float]) -> str:
        """Create margin type pie chart.

        Args:
            margin_by_type: Margin by margin type.

        Returns:
            Plotly chart HTML div.
        """
        if not PLOTLY_AVAILABLE:
            return "<p>Charts not available (Plotly not installed)</p>"

        labels = [mt.value for mt in margin_by_type.keys()]
        values = list(margin_by_type.values())

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=colors,
        )])

        fig.update_layout(
            title='SIMM Distribution by Margin Type',
            template='plotly_white',
            height=400,
        )

        return fig.to_html(include_plotlyjs='cdn', div_id="margintype-chart")

    def _get_currency_symbol(self, currency: str) -> str:
        """Get currency symbol for display.

        Args:
            currency: Three-letter currency code.

        Returns:
            Currency symbol or code.
        """
        symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'CHF': 'Fr ',
            'AUD': 'A$',
            'CAD': 'C$',
        }
        return symbols.get(currency, f'{currency} ')

    def _get_css_styles(self) -> str:
        """Get CSS styles for the report.

        Returns:
            CSS styles string.
        """
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
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

        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            margin: -20px -20px 30px -20px;
            border-radius: 0;
        }

        .header-content {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .logo {
            height: 60px;
        }

        .header-text h1 {
            font-size: 2em;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 1em;
            opacity: 0.9;
        }

        .section {
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e0e0e0;
        }

        .section:last-child {
            border-bottom: none;
        }

        .section h2 {
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #667eea;
        }

        .section h3 {
            font-size: 1.4em;
            margin: 20px 0 10px 0;
            color: #555;
        }

        .section h4 {
            font-size: 1.1em;
            margin: 15px 0 10px 0;
            color: #666;
        }

        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }

        .card.primary {
            border-left-color: #764ba2;
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        }

        .card h3 {
            font-size: 0.9em;
            color: #666;
            margin: 0 0 10px 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .card .metric {
            font-size: 2em;
            font-weight: bold;
            color: #333;
            margin: 0;
        }

        .card .label {
            font-size: 0.85em;
            color: #999;
            margin: 5px 0 0 0;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .data-table th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }

        .data-table td {
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }

        .data-table tr:hover {
            background-color: #f5f5f5;
        }

        .data-table .number {
            text-align: right;
            font-family: 'Courier New', monospace;
        }

        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
        }

        .warnings {
            list-style: none;
            padding: 0;
        }

        .warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 4px;
        }

        .report-footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.9em;
        }

        .timestamp {
            margin-top: 5px;
        }

        .expandable {
            margin: 10px 0;
        }

        @media print {
            body {
                background: white;
            }
            .container {
                box-shadow: none;
                max-width: 100%;
            }
            .report-header {
                background: #667eea;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
        }

        @media (max-width: 768px) {
            .summary-cards {
                grid-template-columns: 1fr;
            }
            .header-content {
                flex-direction: column;
                text-align: center;
            }
        }
        """
