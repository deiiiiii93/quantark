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

__version__ = SACVA_VERSION

__all__ = [
    "SACVACalculator", "SACVAResult", "CVASensitivity", "CVAPortfolio",
    "RiskClass", "RiskType", "CreditQuality", "SACVADashboard", "SACVA_VERSION",
]
