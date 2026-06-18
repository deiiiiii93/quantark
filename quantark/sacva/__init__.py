"""SA-CVA — Basel standardised approach for CVA risk (MAR50.27-50.77).

Aggregates supplied CVA and hedge sensitivities into delta + vega capital. The
module consumes sensitivities (MAR50.29); it does not compute CVA itself.
See ``quantark/sacva/doc/sacva_basel.md`` and the design spec.
"""

from quantark.sacva.calculator import SACVACalculator
from quantark.sacva.results.result import SACVAResult
from quantark.sacva.models.enums import RiskClass, RiskType, CreditQuality
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.models.portfolio import CVAPortfolio
from quantark.sacva.parameters.supervisory import SACVA_VERSION
from quantark.sacva.dashboard import SACVADashboard

# portfolio/engine integration: compute CVA + sensitivities from a real portfolio
from quantark.sacva.sacva_engine import SACVAEngine
from quantark.sacva.exposure.simulator import (
    MonteCarloExposureConfig,
    MonteCarloExposureEngine,
)
from quantark.sacva.cva.engine import RegulatoryCVAEngine
from quantark.sacva.exposure.historical.engine import (
    HistoricalExposureConfig,
    HistoricalExposureEngine,
)
from quantark.sacva.sensitivities.engine import CVASensitivityEngine
from quantark.sacva.portfolio.trade import CVAHedge, CVATrade
from quantark.sacva.portfolio.netting import NettingSet
from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.portfolio.trade_portfolio import CVATradePortfolio
from quantark.sacva.portfolio.credit_curve import PillarHazardCurve

__version__ = SACVA_VERSION

__all__ = [
    "SACVACalculator", "SACVAResult", "CVASensitivity", "CVAPortfolio",
    "RiskClass", "RiskType", "CreditQuality", "SACVADashboard", "SACVA_VERSION",
    # portfolio/engine integration
    "SACVAEngine", "MonteCarloExposureEngine", "MonteCarloExposureConfig",
    "RegulatoryCVAEngine", "CVASensitivityEngine", "CVATrade", "CVAHedge",
    "NettingSet", "Counterparty", "CVATradePortfolio", "PillarHazardCurve",
    # non-regulatory real-world PFE/EE (never SA-CVA eligible)
    "HistoricalExposureEngine", "HistoricalExposureConfig",
]
