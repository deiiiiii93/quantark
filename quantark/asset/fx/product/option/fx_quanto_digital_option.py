"""
FX European quanto digital option.
"""

from datetime import datetime
from typing import Optional

from quantark.util.enum import FxPayoutCurrency, OptionType
from quantark.util.exceptions import ValidationError
from ..currency_pair import CurrencyPair
from .fx_digital_option import FxDigitalOption


class FxQuantoDigitalOption(FxDigitalOption):
    """
    FX digital option settled in a third currency.

    Pays a fixed domestic-currency amount converted to the settlement
    currency at the contractually fixed `quanto_fx_rate` when in the money.
    Only the cash-or-nothing (domestic payout) variant is supported.

    Attributes:
        quanto_fx_rate: Fixed contractual conversion rate from domestic
            (quote) currency to settlement currency
        settlement_ccy: Settlement currency code (informational)
    """

    def __init__(
        self,
        strike: float,
        option_type: OptionType,
        payout: float,
        quanto_fx_rate: float,
        currency_pair: Optional[CurrencyPair] = None,
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        delivery: Optional[float] = None,
        delivery_date: Optional[datetime] = None,
        payout_currency: FxPayoutCurrency = FxPayoutCurrency.DOMESTIC,
        participation_rate: float = 1.0,
        premium_currency: Optional[str] = None,
        premium_amount: Optional[float] = None,
        settlement_ccy: str = "SET",
    ):
        self.quanto_fx_rate = quanto_fx_rate
        self.settlement_ccy = settlement_ccy.upper()
        super().__init__(
            strike=strike,
            option_type=option_type,
            payout=payout,
            currency_pair=currency_pair,
            maturity=maturity,
            expiry_date=expiry_date,
            delivery=delivery,
            delivery_date=delivery_date,
            payout_currency=payout_currency,
            participation_rate=participation_rate,
            premium_currency=premium_currency,
            premium_amount=premium_amount,
        )

    def validate(self) -> None:
        """Validate quanto digital parameters."""
        super().validate()
        if self.quanto_fx_rate <= 0:
            raise ValidationError(
                f"Quanto FX rate must be positive, got {self.quanto_fx_rate}"
            )
        if self.payout_currency != FxPayoutCurrency.DOMESTIC:
            raise ValidationError(
                "Quanto digital options support only domestic (cash-or-nothing) "
                "payouts"
            )

    def get_payoff(self, spot: float) -> float:
        """
        Payoff at expiry in the settlement currency: the fixed domestic
        payout converted at the fixed quanto rate.
        """
        return super().get_payoff(spot) * self.quanto_fx_rate

    def __repr__(self):
        return (
            f"FxQuantoDigitalOption({self.currency_pair}, {self.option_type}, "
            f"K={self.strike:.6f}, payout={self.payout:,.0f}, "
            f"X_q={self.quanto_fx_rate:.6f} -> {self.settlement_ccy})"
        )
