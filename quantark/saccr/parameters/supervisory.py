"""Supervisory parameters for SA-CCR calculations.

Reference: Basel Committee SA-CCR document, Table 2 (paragraph 183).
See ``quantark/saccr/doc/saccr_basel.md``.

All tabular data is frozen at class level (dicts wrapped in
``types.MappingProxyType``) so the regulatory calibration cannot be mutated at
runtime. Lookup getters raise :class:`ValidationError` for an unknown asset
class / commodity type / credit grade / index grade rather than silently
returning a default factor.
"""

from types import MappingProxyType

from quantark.saccr.models.enums import (
    AssetClass,
    CreditRating,
    IndexGrade,
    CommodityHedgingSet,
    CommodityType,
    TransactionType,
    COMMODITY_TYPE_TO_HEDGING_SET,
)
from quantark.util.exceptions import ValidationError


#: Implemented rule set / calibration version.
SACCR_VERSION = "Basel SA-CCR (BCBS d291, Mar 2014, rev Apr 2014)"


class SupervisoryParameters:
    """Supervisory factors, correlations, and option volatilities from Table 2.

    Reference: Basel Committee SA-CCR document, paragraph 183.
    """

    # Alpha multiplier (paragraph 128)
    ALPHA = 1.4

    # PFE multiplier floor (paragraph 149)
    FLOOR = 0.05

    # Interest Rate
    SF_INTEREST_RATE = 0.005  # 0.50%
    VOLATILITY_INTEREST_RATE = 0.50  # 50%

    # Foreign Exchange
    SF_FX = 0.04  # 4.0%
    VOLATILITY_FX = 0.15  # 15%

    # Credit - Single Name by rating
    SF_CREDIT_SINGLE = MappingProxyType({
        CreditRating.AAA: 0.0038,  # 0.38%
        CreditRating.AA: 0.0038,   # 0.38%
        CreditRating.A: 0.0042,    # 0.42%
        CreditRating.BBB: 0.0054,  # 0.54%
        CreditRating.BB: 0.0106,   # 1.06%
        CreditRating.B: 0.016,     # 1.6%
        CreditRating.CCC: 0.06,    # 6.0%
    })

    # Credit - Index by grade
    SF_CREDIT_INDEX = MappingProxyType({
        IndexGrade.IG: 0.0038,  # 0.38%
        IndexGrade.SG: 0.0106,  # 1.06%
    })

    # Credit correlations
    CORRELATION_CREDIT_SINGLE = 0.50  # 50%
    CORRELATION_CREDIT_INDEX = 0.80   # 80%

    # Credit volatility
    VOLATILITY_CREDIT_SINGLE = 1.00  # 100%
    VOLATILITY_CREDIT_INDEX = 0.80   # 80%

    # Equity - Single Name
    SF_EQUITY_SINGLE = 0.32  # 32%
    CORRELATION_EQUITY_SINGLE = 0.50  # 50%
    VOLATILITY_EQUITY_SINGLE = 1.20  # 120%

    # Equity - Index
    SF_EQUITY_INDEX = 0.20  # 20%
    CORRELATION_EQUITY_INDEX = 0.80  # 80%
    VOLATILITY_EQUITY_INDEX = 0.75  # 75%

    # Commodity - by hedging set
    SF_COMMODITY = MappingProxyType({
        CommodityHedgingSet.ENERGY: 0.18,        # 18%
        CommodityHedgingSet.METALS: 0.18,        # 18%
        CommodityHedgingSet.AGRICULTURAL: 0.18,  # 18%
        CommodityHedgingSet.OTHER: 0.18,         # 18%
    })

    # Commodity - special case for electricity
    SF_COMMODITY_ELECTRICITY = 0.40  # 40%

    # Commodity correlation (same for all hedging sets)
    CORRELATION_COMMODITY = 0.40  # 40%

    # Commodity volatility
    VOLATILITY_COMMODITY = MappingProxyType({
        CommodityHedgingSet.ENERGY: 0.70,        # 70%
        CommodityHedgingSet.METALS: 0.70,        # 70%
        CommodityHedgingSet.AGRICULTURAL: 0.70,  # 70%
        CommodityHedgingSet.OTHER: 0.70,         # 70%
    })
    VOLATILITY_COMMODITY_ELECTRICITY = 1.50  # 150%

    # Transaction type multipliers (paragraphs 162-163)
    BASIS_SF_MULTIPLIER = 0.5
    VOLATILITY_SF_MULTIPLIER = 5.0

    @classmethod
    def get_supervisory_factor(
        cls,
        asset_class: AssetClass,
        credit_rating: CreditRating = None,
        index_grade: IndexGrade = None,
        is_index: bool = False,
        commodity_type: CommodityType = None,
        transaction_type: TransactionType = TransactionType.REGULAR,
    ) -> float:
        """Return the supervisory factor for a trade (Table 2).

        Raises:
            ValidationError: if a required discriminator (credit rating, index
                grade, commodity type) is missing or unknown — no silent default.
        """
        if asset_class == AssetClass.INTEREST_RATE:
            sf = cls.SF_INTEREST_RATE

        elif asset_class == AssetClass.FX:
            sf = cls.SF_FX

        elif asset_class == AssetClass.CREDIT:
            if is_index:
                if index_grade not in cls.SF_CREDIT_INDEX:
                    raise ValidationError(
                        f"Unknown/missing credit index grade: {index_grade!r}")
                sf = cls.SF_CREDIT_INDEX[index_grade]
            else:
                if credit_rating not in cls.SF_CREDIT_SINGLE:
                    raise ValidationError(
                        f"Unknown/missing credit rating: {credit_rating!r}")
                sf = cls.SF_CREDIT_SINGLE[credit_rating]

        elif asset_class == AssetClass.EQUITY:
            sf = cls.SF_EQUITY_INDEX if is_index else cls.SF_EQUITY_SINGLE

        elif asset_class == AssetClass.COMMODITY:
            if commodity_type == CommodityType.ELECTRICITY:
                sf = cls.SF_COMMODITY_ELECTRICITY
            else:
                hedging_set = cls._commodity_hedging_set(commodity_type)
                sf = cls.SF_COMMODITY[hedging_set]
        else:
            raise ValidationError(f"Unknown asset class: {asset_class!r}")

        if transaction_type == TransactionType.BASIS:
            sf *= cls.BASIS_SF_MULTIPLIER
        elif transaction_type == TransactionType.VOLATILITY:
            sf *= cls.VOLATILITY_SF_MULTIPLIER

        return sf

    @classmethod
    def get_correlation(
        cls,
        asset_class: AssetClass,
        is_index: bool = False,
    ) -> float:
        """Return the single-factor-model correlation parameter (Table 2).

        IR and FX do not use a correlation (N/A in Table 2) and return 0.0.
        """
        if asset_class == AssetClass.CREDIT:
            return cls.CORRELATION_CREDIT_INDEX if is_index else cls.CORRELATION_CREDIT_SINGLE
        if asset_class == AssetClass.EQUITY:
            return cls.CORRELATION_EQUITY_INDEX if is_index else cls.CORRELATION_EQUITY_SINGLE
        if asset_class == AssetClass.COMMODITY:
            return cls.CORRELATION_COMMODITY
        if asset_class in (AssetClass.INTEREST_RATE, AssetClass.FX):
            return 0.0
        raise ValidationError(f"Unknown asset class: {asset_class!r}")

    @classmethod
    def get_option_volatility(
        cls,
        asset_class: AssetClass,
        is_index: bool = False,
        commodity_type: CommodityType = None,
    ) -> float:
        """Return the supervisory option volatility for the delta calculation (Table 2)."""
        if asset_class == AssetClass.INTEREST_RATE:
            return cls.VOLATILITY_INTEREST_RATE
        if asset_class == AssetClass.FX:
            return cls.VOLATILITY_FX
        if asset_class == AssetClass.CREDIT:
            return cls.VOLATILITY_CREDIT_INDEX if is_index else cls.VOLATILITY_CREDIT_SINGLE
        if asset_class == AssetClass.EQUITY:
            return cls.VOLATILITY_EQUITY_INDEX if is_index else cls.VOLATILITY_EQUITY_SINGLE
        if asset_class == AssetClass.COMMODITY:
            if commodity_type == CommodityType.ELECTRICITY:
                return cls.VOLATILITY_COMMODITY_ELECTRICITY
            hedging_set = cls._commodity_hedging_set(commodity_type)
            return cls.VOLATILITY_COMMODITY[hedging_set]
        raise ValidationError(f"Unknown asset class: {asset_class!r}")

    @staticmethod
    def _commodity_hedging_set(commodity_type: CommodityType) -> CommodityHedgingSet:
        """Resolve a commodity type to its hedging set, or raise if unmapped."""
        if commodity_type not in COMMODITY_TYPE_TO_HEDGING_SET:
            raise ValidationError(
                f"Unknown/missing commodity type: {commodity_type!r}")
        return COMMODITY_TYPE_TO_HEDGING_SET[commodity_type]
