"""
VaR module for portfolio Value-at-Risk calculations.
"""

from quantark.var.backtest import VaRBacktestResult, VaRBacktester
from quantark.var.base import VaREngine
from quantark.var.config import (
    EquityRiskFactorConfig,
    FIRiskFactorConfig,
    VaRConfig,
    VaRMethod,
)
from quantark.var.engines import (
    HistoricalVaREngine,
    MonteCarloVaREngine,
    ParametricVaREngine,
)
from quantark.var.results import IncrementalVaRResult, VaRResult
from quantark.var.results.var_report import VaRReportGenerator
from quantark.var.attribution import ComponentVaRCalculator, MarginalVaRCalculator, VaRAttributor

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