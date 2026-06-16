"""Enumerations for SA-CVA calculations.

Reference: Basel Committee, MAR50.43-50.77. See ``quantark/sacva/doc/sacva_basel.md``.
"""

from enum import Enum, auto


class RiskClass(Enum):
    """SA-CVA risk classes (MAR50.43, 50.45)."""

    INTEREST_RATE = auto()
    FX = auto()
    COUNTERPARTY_CREDIT = auto()
    REFERENCE_CREDIT = auto()
    EQUITY = auto()
    COMMODITY = auto()


class RiskType(Enum):
    """Sensitivity type. Counterparty credit has no vega (MAR50.45, 50.63)."""

    DELTA = auto()
    VEGA = auto()


class CreditQuality(Enum):
    """Credit quality for credit risk classes (MAR50.16, 50.66)."""

    IG = auto()      # investment grade
    HY_NR = auto()   # high yield / not rated
