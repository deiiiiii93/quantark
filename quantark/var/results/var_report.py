"""
VaR report generation module.

This module contains the VaRReportGenerator class for creating
human-readable reports from VaR calculations.
"""

from typing import Any, Dict, List, Optional, TextIO
from datetime import datetime

from quantark.var.results.var_result import VaRResult
from quantark.var.results.incremental_var_result import IncrementalVaRResult


class VaRReportGenerator:
    """
    Generates comprehensive reports from VaR calculation results.

    This class provides methods to generate summary reports, position reports,
    factor reports, and backtesting reports from VaR calculations.
    """

    def __init__(self, output_format: str = "text"):
        """
        Initialize VaR report generator.

        Args:
            output_format: Output format ('text', 'json', 'html')
        """
        self.output_format = output_format

    def generate_summary(
        self,
        var_result: VaRResult,
        output: Optional[TextIO] = None,
    ) -> str:
        """
        Generate a summary report of VaR calculation.

        Args:
            var_result: VaR calculation result
            output: Optional file-like object to write to

        Returns:
            Formatted summary report as string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("VaR CALCULATION SUMMARY REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Basic VaR metrics
        lines.append("CORE METRICS")
        lines.append("-" * 70)
        lines.append(f"Portfolio Value:          ${var_result.portfolio_value:,.2f}")
        lines.append(f"Confidence Level:         {var_result.confidence_level * 100:.1f}%")
        lines.append(f"Holding Period:           {var_result.holding_period} day(s)")
        lines.append(f"VaR Method:               {var_result.method}")
        lines.append("")

        lines.append("VaR RESULTS")
        lines.append("-" * 70)
        lines.append(f"Value-at-Risk (VaR):      ${var_result.var:,.2f}")
        lines.append(f"  As % of Portfolio:      {var_result.var_as_pct * 100:.2f}%")
        lines.append(f"Conditional VaR (CVaR):   ${var_result.cvar:,.2f}")
        lines.append("")

        # Stressed VaR if available
        if var_result.stressed_var is not None:
            lines.append("STRESSED VaR (SVaR)")
            lines.append("-" * 70)
            lines.append(f"Stressed VaR:             ${var_result.stressed_var:,.2f}")
            lines.append(f"  As % of Portfolio:      {(var_result.stressed_var / var_result.portfolio_value) * 100:.2f}%")
            lines.append(f"Stressed CVaR:            ${var_result.stressed_cvar:,.2f}")

            if var_result.stressed_period:
                lines.append(f"Stressed Period:          {var_result.stressed_period.get('start')} to {var_result.stressed_period.get('end')}")
            lines.append("")

        # Attribution breakdown if available
        if var_result.component_var:
            lines.append("COMPONENT VaR (Top 10)")
            lines.append("-" * 70)
            sorted_components = sorted(
                var_result.component_var.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            for pos_id, comp_var in sorted_components:
                pct = (comp_var / var_result.portfolio_value) * 100
                lines.append(f"{pos_id:20s} ${comp_var:>10,.2f} ({pct:>6.2f}%)")
            lines.append("")

        # Factor attribution if available
        if var_result.factor_var:
            lines.append("FACTOR VaR ATTRIBUTION")
            lines.append("-" * 70)
            for factor, factor_var in var_result.factor_var.items():
                pct = (factor_var / var_result.portfolio_value) * 100
                lines.append(f"{factor:20s} ${factor_var:>10,.2f} ({pct:>6.2f}%)")
            lines.append("")

        # Metadata
        lines.append("CALCULATION DETAILS")
        lines.append("-" * 70)
        lines.append(f"Calculation Time:         {var_result.calculation_timestamp}")
        lines.append(f"Execution Time:           {var_result.execution_time_seconds:.3f} seconds")
        lines.append("")
        lines.append("=" * 70)

        report = "\n".join(lines)

        if output:
            output.write(report)

        return report

    def generate_position_report(
        self,
        var_result: VaRResult,
        incremental_result: Optional[IncrementalVaRResult] = None,
        output: Optional[TextIO] = None,
    ) -> str:
        """
        Generate a detailed position-level report.

        Args:
            var_result: VaR calculation result
            incremental_result: Optional incremental VaR result
            output: Optional file-like object to write to

        Returns:
            Formatted position report as string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("POSITION-LEVEL VaR REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Component VaR table
        if var_result.component_var:
            lines.append("COMPONENT VaR BREAKDOWN")
            lines.append("-" * 70)
            lines.append(f"{'Position ID':<20} {'Component VaR':>15} {'% of Portfolio':>15}")
            lines.append("-" * 70)

            sorted_components = sorted(
                var_result.component_var.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for pos_id, comp_var in sorted_components:
                pct = (comp_var / var_result.portfolio_value) * 100
                lines.append(
                    f"{pos_id:<20} ${comp_var:>13,.2f} {pct:>13.2f}%"
                )
            lines.append("")

        # Incremental VaR table
        if incremental_result:
            lines.append("INCREMENTAL VaR BREAKDOWN")
            lines.append("-" * 70)
            lines.append(f"{'Position ID':<20} {'Incremental VaR':>15} {'% of Total':>15}")
            lines.append("-" * 70)

            # Sort by incremental VaR
            sorted_ivari = sorted(
                incremental_result.position_ivari.items(),
                key=lambda x: x[1],
                reverse=True
            )

            total_ivari = sum(incremental_result.position_ivari.values())
            for pos_id, ivari in sorted_ivari:
                pct = (ivari / total_ivari) * 100 if total_ivari > 0 else 0
                lines.append(
                    f"{pos_id:<20} ${ivari:>13,.2f} {pct:>13.2f}%"
                )
            lines.append("")

            # Diversification summary
            lines.append("DIVERSIFICATION ANALYSIS")
            lines.append("-" * 70)
            lines.append(f"Diversification Benefit:  ${incremental_result.diversification_benefit:,.2f}")
            lines.append(f"Diversification Ratio:    {incremental_result.get_diversification_ratio():.3f}")
            lines.append("")

        # Marginal VaR table
        if var_result.marginal_var:
            lines.append("MARGINAL VaR BREAKDOWN")
            lines.append("-" * 70)
            lines.append(f"{'Position ID':<20} {'Marginal VaR':>15} {'% of Total':>15}")
            lines.append("-" * 70)

            sorted_marginal = sorted(
                var_result.marginal_var.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for pos_id, margin_var in sorted_marginal:
                pct = (margin_var / var_result.var) * 100
                lines.append(
                    f"{pos_id:<20} ${margin_var:>13,.2f} {pct:>13.2f}%"
                )
            lines.append("")

        lines.append("=" * 70)

        report = "\n".join(lines)

        if output:
            output.write(report)

        return report

    def generate_factor_report(
        self,
        var_result: VaRResult,
        output: Optional[TextIO] = None,
    ) -> str:
        """
        Generate a risk factor attribution report.

        Args:
            var_result: VaR calculation result
            output: Optional file-like object to write to

        Returns:
            Formatted factor report as string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("RISK FACTOR ATTRIBUTION REPORT")
        lines.append("=" * 70)
        lines.append("")

        if not var_result.factor_var:
            lines.append("No factor attribution data available.")
            lines.append("")
        else:
            lines.append("FACTOR VaR CONTRIBUTION")
            lines.append("-" * 70)
            lines.append(f"{'Risk Factor':<25} {'VaR':>15} {'% of Total':>15}")
            lines.append("-" * 70)

            sorted_factors = sorted(
                var_result.factor_var.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for factor, factor_var in sorted_factors:
                pct = (factor_var / var_result.var) * 100
                lines.append(
                    f"{factor:<25} ${factor_var:>13,.2f} {pct:>13.2f}%"
                )
            lines.append("")

            # Calculate factor correlations if available
            lines.append("FACTOR CONCENTRATION")
            lines.append("-" * 70)
            total_factor_var = sum(var_result.factor_var.values())
            for factor, factor_var in sorted_factors:
                pct_of_total = (factor_var / total_factor_var) * 100
                lines.append(f"{factor:<25} {pct_of_total:>28.2f}%")
            lines.append("")

        lines.append("=" * 70)

        report = "\n".join(lines)

        if output:
            output.write(report)

        return report

    def generate_backtest_report(
        self,
        backtest_result: Any,
        output: Optional[TextIO] = None,
    ) -> str:
        """
        Generate a VaR backtesting report.

        Args:
            backtest_result: VaR backtest result object
            output: Optional file-like object to write to

        Returns:
            Formatted backtest report as string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("VaR BACKTESTING REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Extract backtest metrics
        if hasattr(backtest_result, 'total_observations'):
            lines.append("BACKTEST SUMMARY")
            lines.append("-" * 70)
            lines.append(f"Total Observations:       {backtest_result.total_observations}")
            lines.append(f"Violations:               {backtest_result.num_violations}")
            lines.append(f"Violation Rate:           {(backtest_result.num_violations / backtest_result.total_observations * 100):.2f}%")
            lines.append(f"Expected Rate:            {((1 - backtest_result.confidence_level) * 100):.2f}%")
            lines.append("")

        # Kupiec test
        if hasattr(backtest_result, 'kupiec_p_value'):
            lines.append("KUPIEC PROPORTION OF FAILURES TEST")
            lines.append("-" * 70)
            lines.append(f"P-Value:                  {backtest_result.kupiec_p_value:.6f}")
            lines.append(f"Test Statistic:           {backtest_result.kupiec_statistic:.4f}")
            lines.append(f"Result:                   {'PASS' if backtest_result.kupiec_pass else 'FAIL'}")
            lines.append("")

        # Christoffersen test
        if hasattr(backtest_result, 'christoffersen_p_value'):
            lines.append("CHRISTOFFERSEN CONDITIONAL COVERAGE TEST")
            lines.append("-" * 70)
            lines.append(f"P-Value:                  {backtest_result.christoffersen_p_value:.6f}")
            lines.append(f"Test Statistic:           {backtest_result.christoffersen_statistic:.4f}")
            lines.append(f"Result:                   {'PASS' if backtest_result.christoffersen_pass else 'FAIL'}")
            lines.append("")

        # Basel traffic light
        if hasattr(backtest_result, 'traffic_light_zone'):
            lines.append("BASEL TRAFFIC LIGHT CLASSIFICATION")
            lines.append("-" * 70)
            lines.append(f"Zone:                     {backtest_result.traffic_light_zone}")
            lines.append(f"Status:                   {backtest_result.traffic_light_status}")
            lines.append("")

        lines.append("=" * 70)

        report = "\n".join(lines)

        if output:
            output.write(report)

        return report
