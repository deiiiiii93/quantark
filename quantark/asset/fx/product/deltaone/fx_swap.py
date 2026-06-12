"""
FX swap (near leg + far leg).
"""

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from quantark.util.calendar import BusinessDayConvention, Calendar
from quantark.util.exceptions import ValidationError
from ..currency_pair import CurrencyPair
from .base_fx_deltaone import BaseFxDeltaOneProduct

if TYPE_CHECKING:
    from quantark.priceenv import FxPricingEnvironment


class FxSwap(BaseFxDeltaOneProduct):
    """
    FX swap: sell the base currency at `near_rate` on the near date and buy
    it back at `far_rate` on the far date (both business-day adjusted).

    Attributes:
        near_rate: Exchange rate of the near leg (quote per base)
        far_rate: Exchange rate of the far leg (quote per base)
        near_date: Contractual near settlement date
        far_date: Contractual far settlement date
        swap_points: far_rate - near_rate
    """

    def __init__(
        self,
        currency_pair: CurrencyPair,
        notional_base: float,
        near_rate: float,
        far_rate: float,
        near_date: datetime,
        far_date: datetime,
        trade_date: Optional[datetime] = None,
        calendar: Optional[Calendar] = None,
        business_day_convention: BusinessDayConvention = (
            BusinessDayConvention.MODIFIED_FOLLOWING
        ),
    ):
        super().__init__(
            currency_pair=currency_pair,
            notional_base=notional_base,
            calendar=calendar,
            business_day_convention=business_day_convention,
        )
        self.near_rate = near_rate
        self.far_rate = far_rate
        self.near_date = near_date
        self.far_date = far_date
        self.trade_date = trade_date
        self.adjusted_near_date = self.adjust_date(near_date)
        self.adjusted_far_date = self.adjust_date(far_date)
        self.expiry_date = self.adjusted_far_date
        self.validate()

    @property
    def swap_points(self) -> float:
        """Swap points: far rate minus near rate."""
        return self.far_rate - self.near_rate

    def is_near_leg_expired(self, fx_env: "FxPricingEnvironment") -> bool:
        """True when the near leg settled before the valuation date."""
        return self.adjusted_near_date < fx_env.valuation_date

    def get_near_time(self, fx_env: "FxPricingEnvironment") -> float:
        """
        Year fraction from valuation to the adjusted near date.

        Raises:
            ValidationError: If the near leg has already settled
        """
        return self._year_fraction_to(self.adjusted_near_date, fx_env)

    def get_payoff(self, spot: float) -> float:
        """
        Net far-leg settlement value at the far date: N * (S - far_rate) in
        quote currency. The near leg is a fixed cashflow with no spot
        dependence at the far date.
        """
        return self.notional_base * (spot - self.far_rate)

    def validate(self) -> None:
        """Validate swap parameters."""
        self._validate_notional()
        if self.near_rate <= 0 or self.far_rate <= 0:
            raise ValidationError(
                f"Swap leg rates must be positive, got near={self.near_rate}, "
                f"far={self.far_rate}"
            )
        if self.adjusted_near_date >= self.adjusted_far_date:
            raise ValidationError(
                f"Near date ({self.adjusted_near_date.date()}) must be before "
                f"far date ({self.adjusted_far_date.date()})"
            )

    def __repr__(self):
        return (
            f"FxSwap({self.currency_pair}, N={self.notional_base:,.0f}, "
            f"near={self.near_rate:.6f}@{self.adjusted_near_date.date()}, "
            f"far={self.far_rate:.6f}@{self.adjusted_far_date.date()})"
        )
