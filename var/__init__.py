"""
VaR module for portfolio Value-at-Risk calculations.
"""

from var.backtest import VaRBacktestResult, VaRBacktester
from var.base import VaREngine
from var.config import (
    EquityRiskFactorConfig,
    FIRiskFactorConfig,
    VaRConfig,
    VaRMethod,
)
from var.engines import (
    HistoricalVaREngine,
    MonteCarloVaREngine,
    ParametricVaREngine,
)
from var.results import IncrementalVaRResult, VaRResult
from var.results.var_report import VaRReportGenerator
from var.attribution import ComponentVaRCalculator, MarginalVaRCalculator, VaRAttributor

__all__ = [
    "VaREngine",
    "VaRConfig",
    "VaRMethod",
    "EquityRiskFactorConfig",
    "FIRiskFactorConfig",
    "VaRResult",
    "IncrementalVaRResult",
    "ParametricVaREngine",
    "HistoricalVaREngine",
    "MonteCarloVaREngine",
    "VaRBacktester",
    "VaRBacktestResult",
    "VaRReportGenerator",
    "ComponentVaRCalculator",
    "MarginalVaRCalculator",
    "VaRAttributor",
]