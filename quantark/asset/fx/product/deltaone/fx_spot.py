"""
FX spot transaction.
"""

from datetime import datetime
from typing import Optional

from quantark.util.calendar import BusinessDayConvention, Calendar
from quantark.util.exceptions import ValidationError
from ..currency_pair import CurrencyPair
from .base_fx_deltaone import BaseFxDeltaOneProduct


class FxSpot(BaseFxDeltaOneProduct):
    """
    FX spot transaction: exchange `notional_base` units of the base currency
    at `contract_rate`, settling `settlement_days` business days after the
    value date (T+2 by market convention).

    Attributes:
        contract_rate: Agreed spot rate (quote per base)
        value_date: Trade/value date
        settlement_days: Business days until settlement
    """

    def __init__(
        self,
        currency_pair: CurrencyPair,
        notional_base: float,
        contract_rate: float,
        value_date: datetime,
        settlement_days: int = 2,
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
        self.contract_rate = contract_rate
        self.value_date = value_date
        self.settlement_days = settlement_days
        self.expiry_date = self.calendar.add_business_days(
            value_date, settlement_days
        )
        self.validate()

    def get_settlement_date(self) -> datetime:
        """Settlement date: value date advanced by the settlement lag."""
        return self.expiry_date

    def get_payoff(self, spot: float) -> float:
        """Net settlement value: N * (S - K) in quote currency."""
        return self.notional_base * (spot - self.contract_rate)

    def validate(self) -> None:
        """Validate spot transaction parameters."""
        self._validate_notional()
        if self.contract_rate <= 0:
            raise ValidationError(
                f"Contract rate must be positive, got {self.contract_rate}"
            )
        if self.settlement_days < 0:
            raise ValidationError(
                f"Settlement days must be non-negative, got {self.settlement_days}"
            )

    def __repr__(self):
        return (
            f"FxSpot({self.currency_pair}, N={self.notional_base:,.0f}, "
            f"rate={self.contract_rate:.6f}, settle={self.expiry_date.date()})"
        )
