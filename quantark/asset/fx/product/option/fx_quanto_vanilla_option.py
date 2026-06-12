"""
FX European quanto vanilla option.
"""

from datetime import datetime
from typing import Optional

from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError
from ..currency_pair import CurrencyPair
from .fx_vanilla_option import FxVanillaOption


class FxQuantoVanillaOption(FxVanillaOption):
    """
    FX European vanilla option settled in a third currency.

    The payoff is determined by the underlying currency pair but converted to
    the settlement currency at the contractually fixed `quanto_fx_rate`
    (settlement units per domestic unit) and discounted on the settlement
    currency's curve. For a EUR/USD option settled in JPY, the USD payoff is
    converted to JPY at the fixed rate.

    Attributes:
        quanto_fx_rate: Fixed contractual conversion rate from domestic
            (quote) currency to settlement currency
        settlement_ccy: Settlement currency code (informational)
    """

    def __init__(
        self,
        strike: float,
        option_type: OptionType,
        quanto_fx_rate: float,
        currency_pair: Optional[CurrencyPair] = None,
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        delivery: Optional[float] = None,
        delivery_date: Optional[datetime] = None,
        notional_foreign: Optional[float] = None,
        notional_domestic: Optional[float] = None,
        initial_spot: Optional[float] = None,
        participation_rate: float = 1.0,
        premium_currency: Optional[str] = None,
        premium_amount: Optional[float] = None,
        is_annualized: bool = False,
        tenor_days: Optional[float] = None,
        settlement_ccy: str = "SET",
    ):
        self.quanto_fx_rate = quanto_fx_rate
        self.settlement_ccy = settlement_ccy.upper()
        super().__init__(
            strike=strike,
            option_type=option_type,
            currency_pair=currency_pair,
            maturity=maturity,
            expiry_date=expiry_date,
            delivery=delivery,
            delivery_date=delivery_date,
            notional_foreign=notional_foreign,
            notional_domestic=notional_domestic,
            initial_spot=initial_spot,
            participation_rate=participation_rate,
            premium_currency=premium_currency,
            premium_amount=premium_amount,
            is_annualized=is_annualized,
            tenor_days=tenor_days,
        )

    def validate(self) -> None:
        """Validate quanto option parameters."""
        super().validate()
        if self.quanto_fx_rate <= 0:
            raise ValidationError(
                f"Quanto FX rate must be positive, got {self.quanto_fx_rate}"
            )

    def get_payoff(self, spot: float) -> float:
        """
        Payoff at expiry in the settlement currency: the domestic-currency
        vanilla payoff converted at the fixed quanto rate.
        """
        return super().get_payoff(spot) * self.quanto_fx_rate

    def __repr__(self):
        return (
            f"FxQuantoVanillaOption({self.currency_pair}, {self.option_type}, "
            f"K={self.strike:.6f}, N={self.notional:,.0f}, "
            f"X_q={self.quanto_fx_rate:.6f} -> {self.settlement_ccy})"
        )
