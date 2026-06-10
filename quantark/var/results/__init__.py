"""
VaR results and reporting module.

This module contains classes for storing VaR calculation results,
generating reports, and managing incremental VaR calculations.
"""

from var.results.var_result import VaRResult
from var.results.incremental_var_result import IncrementalVaRResult
from var.results.var_report import VaRReportGenerator

__all__ = [
    "VaRResult",
    "IncrementalVaRResult",
    "VaRReportGenerator",
]
