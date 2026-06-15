"""Trade data model for SA-CCR calculations.

Reference: Basel Committee SA-CCR document, paragraphs 155-159.

Conventions:
- Times (``start_date`` S, ``end_date`` E, ``maturity`` M, ``exercise_date`` T)
  are year-fractions (floats). ``maturity`` is floored at 10 business days
  (``10/250``).
- All monetary values are in a single reporting currency (no FX conversion).
- ``currency`` (IR) / ``currency_pair`` (FX) are hedging-set keys only;
  ``currency_pair`` must use the canonical ``"CCY1/CCY2"`` format.
"""

import re
from dataclasses import dataclass
from typing import Optional

from quantark.saccr.models.enums import (
    AssetClass,
    Position,
    CreditRating,
    IndexGrade,
    CommodityHedgingSet,
    CommodityType,
    TransactionType,
)
from quantark.util.enum.option_enums import OptionType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import validate_positive, is_valid_number

#: Minimum maturity: 10 business days, assuming 250 business days per year.
MIN_MATURITY = 10 / 250

_CCY_PAIR_RE = re.compile(r"^[A-Z]{3}/[A-Z]{3}$")


@dataclass
class SACCRTrade:
    """A derivative trade for SA-CCR calculation (paragraphs 155-159)."""

    # Required fields
    trade_id: str
    asset_class: AssetClass
    notional: float
    market_value: float

    # Time parameters (year-fractions)
    maturity: float = 1.0  # M_i
    start_date: float = 0.0  # S_i
    end_date: float = 1.0  # E_i

    # Currency (IR and FX hedging-set keys)
    currency: Optional[str] = None
    currency_pair: Optional[str] = None

    # Credit / Equity
    reference_entity: Optional[str] = None
    is_index: bool = False
    credit_rating: Optional[CreditRating] = None
    index_grade: Optional[IndexGrade] = None

    # Commodity
    commodity_hedging_set: Optional[CommodityHedgingSet] = None
    commodity_type: Optional[CommodityType] = None

    # Option parameters
    is_option: bool = False
    option_type: Optional[OptionType] = None
    underlying_price: Optional[float] = None
    strike_price: Optional[float] = None
    exercise_date: Optional[float] = None

    # CDO tranche
    is_cdo_tranche: bool = False
    attachment_point: Optional[float] = None
    detachment_point: Optional[float] = None

    # Position
    position: Position = Position.LONG

    # Transaction type
    transaction_type: TransactionType = TransactionType.REGULAR

    def __post_init__(self):
        """Validate trade parameters (raises :class:`ValidationError`)."""
        # Enum-type validation: invalid enums otherwise silently default to another
        # meaning (e.g. an unknown asset class is dropped during aggregation).
        if not isinstance(self.asset_class, AssetClass):
            raise ValidationError(f"asset_class must be an AssetClass, got {self.asset_class!r}")
        if not isinstance(self.position, Position):
            raise ValidationError(f"position must be a Position, got {self.position!r}")
        if not isinstance(self.transaction_type, TransactionType):
            raise ValidationError(
                f"transaction_type must be a TransactionType, got {self.transaction_type!r}")
        # BASIS/VOLATILITY require separate hedging sets and SF multipliers
        # (paragraphs 162-163) that are not implemented; reject rather than
        # silently treating them as REGULAR.
        # TODO(saccr): implement BASIS/VOLATILITY hedging-set treatment.
        if self.transaction_type != TransactionType.REGULAR:
            raise ValidationError(
                f"transaction_type {self.transaction_type.name} is not yet supported")

        validate_positive(self.notional, "notional")

        # Maturity must be a valid, non-negative number BEFORE the regulatory floor,
        # so a negative/invalid maturity is not masked by the 10-business-day floor.
        if self.maturity is None or not is_valid_number(self.maturity):
            raise ValidationError(f"maturity must be a finite number, got {self.maturity}")
        if self.maturity < 0:
            raise ValidationError(f"maturity must be non-negative, got {self.maturity}")
        if self.maturity < MIN_MATURITY:
            self.maturity = MIN_MATURITY

        if self.start_date < 0:
            raise ValidationError(f"start_date must be non-negative, got {self.start_date}")
        if self.end_date < self.start_date:
            raise ValidationError(
                f"end_date ({self.end_date}) must be >= start_date ({self.start_date})")

        # Asset-class-specific requirements.
        if self.asset_class == AssetClass.INTEREST_RATE:
            if not self.currency:
                raise ValidationError("IR trades require currency")

        elif self.asset_class == AssetClass.FX:
            if not self.currency_pair:
                raise ValidationError("FX trades require currency_pair")
            if not _CCY_PAIR_RE.match(self.currency_pair):
                raise ValidationError(
                    f"currency_pair must be canonical 'CCY1/CCY2', got {self.currency_pair!r}")

        elif self.asset_class == AssetClass.CREDIT:
            if not self.reference_entity:
                raise ValidationError("Credit trades require reference_entity")
            if self.is_index and not self.index_grade:
                raise ValidationError("Credit index trades require index_grade")
            if not self.is_index and not self.credit_rating:
                raise ValidationError("Single-name credit trades require credit_rating")

        elif self.asset_class == AssetClass.EQUITY:
            if not self.reference_entity:
                raise ValidationError("Equity trades require reference_entity")

        elif self.asset_class == AssetClass.COMMODITY:
            if not self.commodity_type:
                raise ValidationError("Commodity trades require commodity_type")

        # Option parameters.
        if self.is_option:
            if self.option_type is None:
                raise ValidationError("Options require option_type")
            if not isinstance(self.option_type, OptionType):
                raise ValidationError(
                    f"option_type must be an OptionType, got {self.option_type!r}")
            if self.underlying_price is None:
                raise ValidationError("Options require underlying_price")
            if self.strike_price is None:
                raise ValidationError("Options require strike_price")
            if self.exercise_date is None:
                raise ValidationError("Options require exercise_date")

        # CDO tranche parameters.
        if self.is_cdo_tranche:
            if self.attachment_point is None or self.detachment_point is None:
                raise ValidationError(
                    "CDO tranches require attachment and detachment points")
