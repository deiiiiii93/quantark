"""
FX European vanilla option.
"""

from datetime import datetime
from typing import Optional

from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance, is_close
from ..base_fx_product import BaseFxProduct
from ..currency_pair import CurrencyPair


class FxVanillaOption(BaseFxProduct):
    """
    FX European vanilla call or put option.

    The option is written on `notional` units of the foreign (base) currency;
    its value is expressed in the domestic (quote) currency. For a EUR/USD
    call on 1m EUR with strike 1.25, the payoff is 1m * max(S_T - 1.25, 0) USD.

    Notional may be specified in either currency (or both, which must be
    consistent with `initial_spot`, the spot rate at trade inception):
        - notional_foreign only: used directly
        - notional_domestic only: requires initial_spot; N = dom / initial_spot
        - both: initial_spot validated (or derived) as dom / for

    Attributes:
        strike: Strike rate (quote per base)
        option_type: CALL or PUT
        notional: Quantity of foreign (base) currency covered
        initial_spot: Spot rate at trade inception
        participation_rate: Payoff participation multiplier
        premium_currency: Currency the premium is paid in ("DOM" or "FOR")
        premium_amount: Contracted premium amount (for premium-adjusted delta)
        is_annualized: When True, price and Greeks are scaled by
            tenor_days / 365 (legacy deposit-embedding convention; opt-in)
        tenor_days: Contract tenor in days for annualization (defaults to the
            time to expiry when annualization is enabled)
    """

    def __init__(
        self,
        strike: float,
        option_type: OptionType,
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
        self.participation_rate = participation_rate
        self.premium_currency = premium_currency
        self.premium_amount = premium_amount
        self.is_annualized = is_annualized
        self.tenor_days = tenor_days

        self.notional, self.initial_spot, self.notional_domestic = (
            self._resolve_notional(notional_foreign, notional_domestic, initial_spot)
        )

        self.validate()

    @staticmethod
    def _resolve_notional(
        notional_foreign: Optional[float],
        notional_domestic: Optional[float],
        initial_spot: Optional[float],
    ) -> tuple:
        """
        Resolve the foreign-currency notional from the inputs.

        Returns:
            (notional_foreign, initial_spot, notional_domestic)
        """
        if notional_foreign is not None and notional_domestic is not None:
            implied_spot = notional_domestic / notional_foreign
            if initial_spot is not None:
                if not is_close(implied_spot, initial_spot, abs_tol=Tolerance.PRECISION):
                    raise ValidationError(
                        f"Inconsistent notionals: notional_domestic / "
                        f"notional_foreign = {implied_spot:.6f} but initial_spot "
                        f"= {initial_spot:.6f}"
                    )
            return notional_foreign, implied_spot, notional_domestic

        if notional_foreign is not None:
            spot0 = initial_spot
            domestic = notional_foreign * spot0 if spot0 is not None else None
            return notional_foreign, spot0, domestic

        if notional_domestic is not None:
            if initial_spot is None:
                raise ValidationError(
                    "initial_spot is required when only notional_domestic is provided"
                )
            return notional_domestic / initial_spot, initial_spot, notional_domestic

        raise ValidationError(
            "Either notional_foreign or notional_domestic must be provided"
        )

    def validate(self) -> None:
        """Validate option parameters."""
        if self.strike <= 0:
            raise ValidationError(f"Strike must be positive, got {self.strike}")
        if not isinstance(self.option_type, OptionType):
            raise ValidationError(f"Invalid option type: {self.option_type}")
        if self.notional <= 0:
            raise ValidationError(f"Notional must be positive, got {self.notional}")
        if self.participation_rate <= 0:
            raise ValidationError(
                f"Participation rate must be positive, got {self.participation_rate}"
            )
        if self.premium_currency is not None and self.premium_currency.upper() not in (
            "DOM",
            "FOR",
        ):
            raise ValidationError(
                f"premium_currency must be 'DOM' or 'FOR', got {self.premium_currency}"
            )
        if self.tenor_days is not None and self.tenor_days <= 0:
            raise ValidationError(
                f"tenor_days must be positive, got {self.tenor_days}"
            )
        self._validate_maturity_inputs()

    def is_call(self) -> bool:
        """True when the option is a call on the base currency."""
        return self.option_type == OptionType.CALL

    def is_put(self) -> bool:
        """True when the option is a put on the base currency."""
        return self.option_type == OptionType.PUT

    def get_payoff(self, spot: float) -> float:
        """
        Payoff at expiry in domestic (quote) currency.

        Call: N * max(S - K, 0); Put: N * max(K - S, 0); scaled by the
        participation rate.
        """
        if spot < 0:
            raise ValidationError(f"Spot must be non-negative, got {spot}")
        if self.is_call():
            intrinsic = max(spot - self.strike, 0.0)
        else:
            intrinsic = max(self.strike - spot, 0.0)
        return self.notional * intrinsic * self.participation_rate

    def annualization_factor(self, fx_env=None) -> float:
        """
        Scaling applied when is_annualized is enabled: tenor_days / 365,
        defaulting to the time to expiry when tenor_days is unset.
        """
        if not self.is_annualized:
            return 1.0
        if self.tenor_days is not None:
            return self.tenor_days / 365.0
        return self.get_maturity(fx_env)

    def __repr__(self):
        return (
            f"FxVanillaOption({self.currency_pair}, {self.option_type}, "
            f"K={self.strike:.6f}, N={self.notional:,.0f})"
        )
