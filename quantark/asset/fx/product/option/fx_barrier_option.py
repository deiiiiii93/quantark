"""
FX single-barrier vanilla option (knock-out / knock-in, with optional rebate).
"""

from datetime import datetime
from typing import Optional

from quantark.util.enum import OptionType, FxBarrierType
from quantark.util.exceptions import ValidationError
from ..base_fx_product import BaseFxProduct
from ..currency_pair import CurrencyPair


class FxBarrierOption(BaseFxProduct):
    """
    FX single-barrier vanilla option.

    Attributes:
        strike: Strike rate (quote per base).
        barrier: Barrier level (quote per base).
        is_up: True for an up-barrier, False for a down-barrier.
        knock_type: KNOCK_OUT (dies on touch) or KNOCK_IN (born on touch).
        option_type: CALL or PUT of the underlying vanilla.
        rebate: Cash rebate amount in domestic currency. For KNOCK_OUT it is
            paid when the barrier is touched; for KNOCK_IN it is paid at expiry
            only if the barrier is never touched.
        rebate_at_hit: KNOCK_OUT only — pay the rebate at hit (True) vs at
            expiry (False). Must be False for KNOCK_IN.
    """

    def __init__(
        self,
        strike: float,
        barrier: float,
        is_up: bool,
        knock_type: FxBarrierType,
        option_type: OptionType,
        rebate: float = 0.0,
        rebate_at_hit: bool = False,
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
        self.strike = strike
        self.barrier = barrier
        self.is_up = is_up
        self.knock_type = knock_type
        self.option_type = option_type
        self.rebate = rebate
        self.rebate_at_hit = rebate_at_hit
        self.validate()

    def validate(self) -> None:
        if self.strike <= 0:
            raise ValidationError(f"Strike must be positive, got {self.strike}")
        if self.barrier <= 0:
            raise ValidationError(f"Barrier must be positive, got {self.barrier}")
        if not isinstance(self.knock_type, FxBarrierType):
            raise ValidationError(f"Invalid knock_type: {self.knock_type}")
        if not isinstance(self.option_type, OptionType):
            raise ValidationError(f"Invalid option_type: {self.option_type}")
        if self.rebate < 0:
            raise ValidationError(f"Rebate must be non-negative, got {self.rebate}")
        if self.knock_type == FxBarrierType.KNOCK_IN and self.rebate_at_hit:
            raise ValidationError(
                "rebate_at_hit is only valid for KNOCK_OUT; a knock-in rebate "
                "is paid at expiry when the barrier is never touched."
            )
        self._validate_maturity_inputs()

    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    def get_payoff(self, spot: float) -> float:
        """Unconditional vanilla terminal payoff (barrier handled by engine)."""
        if spot < 0:
            raise ValidationError(f"Spot must be non-negative, got {spot}")
        if self.is_call():
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)

    def __repr__(self):
        side = "up" if self.is_up else "down"
        return (
            f"FxBarrierOption({self.currency_pair}, {self.knock_type}, {side}, "
            f"{self.option_type}, K={self.strike:.6f}, H={self.barrier:.6f})"
        )
