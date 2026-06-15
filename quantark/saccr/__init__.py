"""SA-CCR (Standardised Approach for Counterparty Credit Risk) calculator.

Implements the Basel Committee SA-CCR methodology for Exposure at Default (EAD)
on derivative netting sets: ``EAD = alpha x (RC + PFE)``. See
``quantark/saccr/doc/saccr_basel.md`` for the methodology reference and
``quantark/saccr/CLAUDE.md`` for the developer guide.

Reference: Basel Committee on Banking Supervision, "The standardised approach for
measuring counterparty credit risk exposures" (BCBS d291, Mar 2014, rev Apr 2014).
"""

from quantark.saccr.calculator import SACCRCalculator
from quantark.saccr.results.result import SACCRResult
from quantark.saccr.models.trade import SACCRTrade
from quantark.saccr.models.netting_set import SACCRNettingSet
from quantark.saccr.models.enums import (
    AssetClass,
    Position,
    CreditRating,
    IndexGrade,
    CommodityHedgingSet,
    CommodityType,
    TransactionType,
)
from quantark.saccr.parameters.supervisory import SACCR_VERSION
from quantark.util.enum.option_enums import OptionType

__version__ = SACCR_VERSION

__all__ = [
    "SACCRCalculator",
    "SACCRResult",
    "SACCRTrade",
    "SACCRNettingSet",
    "AssetClass",
    "Position",
    "CreditRating",
    "IndexGrade",
    "CommodityHedgingSet",
    "CommodityType",
    "TransactionType",
    "OptionType",
    "SACCR_VERSION",
]
