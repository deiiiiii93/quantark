"""
Base class for FX delta-one (linear) products.
"""

from datetime import datetime
from typing import Optional

from quantark.util.calendar import BusinessDayConvention, Calendar
from quantark.util.exceptions import ValidationError
from ..base_fx_product import BaseFxProduct
from ..currency_pair import CurrencyPair


class BaseFxDeltaOneProduct(BaseFxProduct):
    """
    Base class for linear FX products (spot, forward, swap).

    Delta-one products are specified with contractual dates; business-day
    adjustment uses the product's calendar (default: weekends-only) and
    business day convention (default: MODIFIED_FOLLOWING), matching standard
    FX market practice.

    Attributes:
        notional_base: Amount of the base (foreign) currency exchanged
        calendar: Business day calendar for date adjustments
        business_day_convention: Convention for adjusting contractual dates
    """

    def __init__(
        self,
        currency_pair: CurrencyPair,
        notional_base: float,
        calendar: Optional[Calendar] = None,
        business_day_convention: BusinessDayConvention = (
            BusinessDayConvention.MODIFIED_FOLLOWING
        ),
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
    ):
        super().__init__(
            currency_pair=currency_pair, maturity=maturity, expiry_date=expiry_date
        )
        self.notional_base = notional_base
        self.calendar = calendar if calendar is not None else Calendar()
        self.business_day_convention = business_day_convention

    @property
    def is_linear(self) -> bool:
        """Delta-one products are linear in spot."""
        return True

    def adjust_date(self, date: datetime) -> datetime:
        """Apply the product's business day convention to a date."""
        return self.calendar.adjust_date(date, self.business_day_convention)

    def _validate_notional(self) -> None:
        if self.notional_base <= 0:
            raise ValidationError(
                f"Base notional must be positive, got {self.notional_base}"
            )
