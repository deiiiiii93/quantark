"""
FX European digital (binary) option.
"""

from datetime import datetime
from typing import Optional

from quantark.util.enum import FxPayoutCurrency, OptionType
from quantark.util.exceptions import ValidationError
from ..base_fx_product import BaseFxProduct
from ..currency_pair import CurrencyPair


class FxDigitalOption(BaseFxProduct):
    """
    FX European digital option.

    Pays a fixed amount when in the money at expiry and nothing otherwise.
    The payout currency selects the variant:

    - DOMESTIC (cash-or-nothing): pays `payout` units of the domestic
      (quote) currency.
    - FOREIGN (asset-or-nothing): pays `payout` units of the foreign
      (base) currency, worth payout * S_T in domestic currency.

    Attributes:
        strike: Strike rate (quote per base)
        option_type: CALL (pays when S_T > K) or PUT (pays when S_T < K)
        payout: Fixed payout amount in the payout currency
        payout_currency: DOMESTIC or FOREIGN
        participation_rate: Payoff participation multiplier
        premium_currency: Currency the premium is paid in ("DOM" or "FOR")
        premium_amount: Contracted premium amount (for premium-adjusted delta)
    """

    def __init__(
        self,
        strike: float,
        option_type: OptionType,
        payout: float,
        currency_pair: Optional[CurrencyPair] = None,
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        delivery: Optional[float] = None,
        delivery_date: Optional[datetime] = None,
        payout_currency: FxPayoutCurrency = FxPayoutCurrency.DOMESTIC,
        participation_rate: float = 1.0,
        premium_currency: Optional[str] = None,
        premium_amount: Optional[float] = None,
    ):
        super().__init__(
            currency_pair=currency_pair,
            maturity=maturity,
            expiry_date=expiry_date,
            delivery=delivery,
            delivery_date=delivery_date,
        )
        self.strike = strike
        self.option_type = option_type
        self.payout = payout
        self.payout_currency = payout_currency
        self.participation_rate = participation_rate
        self.premium_currency = premium_currency
        self.premium_amount = premium_amount

        self.validate()

    def validate(self) -> None:
        """Validate digital option parameters."""
        if self.strike <= 0:
            raise ValidationError(f"Strike must be positive, got {self.strike}")
        if not isinstance(self.option_type, OptionType):
            raise ValidationError(f"Invalid option type: {self.option_type}")
        if self.payout <= 0:
            raise ValidationError(f"Payout must be positive, got {self.payout}")
        if not isinstance(self.payout_currency, FxPayoutCurrency):
            raise ValidationError(
                f"Invalid payout currency: {self.payout_currency}"
            )
        if self.participation_rate <= 0:
            raise ValidationError(
                f"Participation rate must be positive, got {self.participation_rate}"
            )
        self._validate_maturity_inputs()

    def is_call(self) -> bool:
        """True when the option pays for S_T above the strike."""
        return self.option_type == OptionType.CALL

    def get_payoff(self, spot: float) -> float:
        """
        Payoff at expiry in domestic (quote) currency.

        Cash-or-nothing: payout when in the money. Asset-or-nothing:
        payout * spot when in the money.
        """
        if spot < 0:
            raise ValidationError(f"Spot must be non-negative, got {spot}")
        in_the_money = spot > self.strike if self.is_call() else spot < self.strike
        if not in_the_money:
            return 0.0
        amount = self.payout
        if self.payout_currency == FxPayoutCurrency.FOREIGN:
            amount *= spot
        return amount * self.participation_rate

    def __repr__(self):
        return (
            f"FxDigitalOption({self.currency_pair}, {self.option_type}, "
            f"K={self.strike:.6f}, payout={self.payout:,.0f} "
            f"{self.payout_currency})"
        )
