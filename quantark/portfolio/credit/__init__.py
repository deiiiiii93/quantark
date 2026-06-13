"""
Credit portfolio subpackage.

CreditPosition / CreditPortfolio conform to the asset-agnostic BasePosition /
BasePortfolio protocols and to the SIMM SIMMSensitivityProvider protocol
(CreditPosition.get_simm_sensitivities -> CreditQ / CreditNQ margin).
"""
from .position import CreditPosition
from .portfolio import CreditPortfolio

__all__ = ["CreditPosition", "CreditPortfolio"]
