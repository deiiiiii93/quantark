"""
FX forward contract.
"""

from datetime import datetime
from typing import Optional

from quantark.util.calendar import BusinessDayConvention, Calendar
from quantark.util.exceptions import ValidationError
from ..currency_pair import CurrencyPair
from .base_fx_deltaone import BaseFxDeltaOneProduct


class FxForward(BaseFxDeltaOneProduct):
    """
    Outright FX forward: buy `notional_base` units of the base currency at
    `contract_rate` on the (business-day adjusted) maturity date.

    Attributes:
        contract_rate: Agreed forward rate (quote per base)
        maturity_date: Contractual maturity date
        trade_date: Trade date (informational)
    """

    def __init__(
        self,
        currency_pair: CurrencyPair,
        notional_base: float,
        contract_rate: float,
        maturity_date: datetime,
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
        self.contract_rate = contract_rate
        self.maturity_date = maturity_date
        self.trade_date = trade_date
        self.expiry_date = self.adjust_date(maturity_date)
        self.validate()

    def get_adjusted_maturity_date(self) -> datetime:
        """Maturity date after business-day adjustment."""
        return self.expiry_date

    def get_payoff(self, spot: float) -> float:
        """Net settlement value at maturity: N * (S_T - K) in quote currency."""
        return self.notional_base * (spot - self.contract_rate)

    def validate(self) -> None:
        """Validate forward parameters."""
        self._validate_notional()
        if self.contract_rate <= 0:
            raise ValidationError(
                f"Contract rate must be positive, got {self.contract_rate}"
            )

    def __repr__(self):
        return (
            f"FxForward({self.currency_pair}, N={self.notional_base:,.0f}, "
            f"K={self.contract_rate:.6f}, maturity={self.expiry_date.date()})"
        )
