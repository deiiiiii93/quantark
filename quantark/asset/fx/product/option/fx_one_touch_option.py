"""
FX one-touch option (expiry-pay).
"""

from datetime import datetime
from typing import Optional

from quantark.util.exceptions import ValidationError
from ..base_fx_product import BaseFxProduct
from ..currency_pair import CurrencyPair


class FxOneTouchOption(BaseFxProduct):
    """
    FX one-touch: pays a fixed domestic amount if the barrier is touched
    over the option's life (settled at expiry).

    Attributes:
        barrier: Barrier level (quote per base).
        is_up: True for an up-barrier (touched from below), False for a
            down-barrier (touched from above).
        payout: Domestic-currency amount paid on touch.
    """

    def __init__(
        self,
        barrier: float,
        is_up: bool,
        payout: float = 1.0,
        currency_pair: Optional[CurrencyPair] = None,
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        delivery: Optional[float] = None,
        delivery_date: Optional[datetime] = None,
    ):
        super().__init__(
            currency_pair=currency_pair,
            maturity=maturity,
            expiry_date=expiry_date,
            delivery=delivery,
            delivery_date=delivery_date,
        )
        self.barrier = barrier
        self.is_up = is_up
        self.payout = payout
        self.validate()

    def validate(self) -> None:
        if self.barrier <= 0:
            raise ValidationError(f"Barrier must be positive, got {self.barrier}")
        if self.payout <= 0:
            raise ValidationError(f"Payout must be positive, got {self.payout}")
        self._validate_maturity_inputs()

    def get_payoff(self, spot: float) -> float:
        """Terminal payoff for a spot at/beyond the barrier (MC cross-check).

        The analytic engine handles path-touch directly; this terminal form
        is used only by Monte-Carlo validation.
        """
        if spot < 0:
            raise ValidationError(f"Spot must be non-negative, got {spot}")
        touched = spot >= self.barrier if self.is_up else spot <= self.barrier
        return self.payout if touched else 0.0

    def __repr__(self):
        side = "up" if self.is_up else "down"
        return (
            f"FxOneTouchOption({self.currency_pair}, {side}, "
            f"H={self.barrier:.6f}, payout={self.payout:g})"
        )
