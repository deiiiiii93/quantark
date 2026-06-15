"""Enumerations for SA-CCR calculations.

Reference: Basel Committee SA-CCR document (BCBS d291, Mar 2014, rev Apr 2014),
paragraphs 151, 161-163, 183 (Table 2). See ``quantark/saccr/doc/saccr_basel.md``.

Note: the option call/put taxonomy is intentionally *not* redefined here. SA-CCR
reuses ``quantark.util.enum.option_enums.OptionType`` (a pure CALL/PUT payoff enum),
whose semantics are identical to what the supervisory delta needs.
"""

from enum import Enum, auto
from types import MappingProxyType


class AssetClass(Enum):
    """Asset classes for SA-CCR (paragraph 151)."""

    INTEREST_RATE = auto()
    FX = auto()
    CREDIT = auto()
    EQUITY = auto()
    COMMODITY = auto()


class Position(Enum):
    """Position direction in the primary risk factor (paragraph 159)."""

    LONG = auto()
    SHORT = auto()


class CreditRating(Enum):
    """Credit ratings for single-name CDS (Table 2).

    Supervisory factors: AAA/AA 0.38%, A 0.42%, BBB 0.54%, BB 1.06%,
    B 1.6%, CCC 6.0%.
    """

    AAA = auto()
    AA = auto()
    A = auto()
    BBB = auto()
    BB = auto()
    B = auto()
    CCC = auto()


class IndexGrade(Enum):
    """Index grade for credit indices (Table 2): IG 0.38%, SG 1.06%."""

    IG = auto()  # Investment Grade
    SG = auto()  # Speculative Grade


class CommodityHedgingSet(Enum):
    """Commodity hedging sets (paragraphs 161, 178-182)."""

    ENERGY = auto()
    METALS = auto()
    AGRICULTURAL = auto()
    OTHER = auto()


class CommodityType(Enum):
    """Commodity types within hedging sets (paragraphs 181-182)."""

    # Energy
    CRUDE_OIL = auto()
    NATURAL_GAS = auto()
    COAL = auto()
    ELECTRICITY = auto()

    # Metals
    GOLD = auto()
    SILVER = auto()
    COPPER = auto()
    ALUMINUM = auto()

    # Agricultural
    CORN = auto()
    WHEAT = auto()
    SOYBEANS = auto()
    COFFEE = auto()

    # Other
    OTHER = auto()


# Mapping from CommodityType to CommodityHedgingSet (paragraph 161).
# Wrapped in MappingProxyType so the regulatory classification is immutable.
COMMODITY_TYPE_TO_HEDGING_SET = MappingProxyType({
    CommodityType.CRUDE_OIL: CommodityHedgingSet.ENERGY,
    CommodityType.NATURAL_GAS: CommodityHedgingSet.ENERGY,
    CommodityType.COAL: CommodityHedgingSet.ENERGY,
    CommodityType.ELECTRICITY: CommodityHedgingSet.ENERGY,
    CommodityType.GOLD: CommodityHedgingSet.METALS,
    CommodityType.SILVER: CommodityHedgingSet.METALS,
    CommodityType.COPPER: CommodityHedgingSet.METALS,
    CommodityType.ALUMINUM: CommodityHedgingSet.METALS,
    CommodityType.CORN: CommodityHedgingSet.AGRICULTURAL,
    CommodityType.WHEAT: CommodityHedgingSet.AGRICULTURAL,
    CommodityType.SOYBEANS: CommodityHedgingSet.AGRICULTURAL,
    CommodityType.COFFEE: CommodityHedgingSet.AGRICULTURAL,
    CommodityType.OTHER: CommodityHedgingSet.OTHER,
})


class TransactionType(Enum):
    """Transaction types for special supervisory-factor treatment (paragraphs 162-163)."""

    REGULAR = auto()
    BASIS = auto()  # SF multiplied by 0.5
    VOLATILITY = auto()  # SF multiplied by 5.0
