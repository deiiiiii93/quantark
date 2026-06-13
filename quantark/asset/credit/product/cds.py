"""
Single-name Credit Default Swap (CDS) product.
"""
from dataclasses import dataclass
from enum import Enum

from quantark.util.exceptions import ValidationError
from .base_credit_product import BaseCreditProduct


class ProtectionSide(Enum):
    """Side of the CDS contract held by the position holder."""

    BUY = "Buy"   # long protection: pays premium, receives (1-R) on default
    SELL = "Sell"  # short protection: receives premium, pays (1-R) on default


@dataclass
class CDS(BaseCreditProduct):
    """
    Single-name Credit Default Swap.

    A reduced-form CDS: the protection buyer pays a running coupon spread on
    the notional until maturity or a credit event, and receives
    ``notional * (1 - recovery_rate)`` if the reference entity defaults.

    Values are expressed from the protection holder's perspective per ``side``.

    Attributes:
        notional: Contract notional.
        maturity: Time to maturity in years.
        recovery_rate: Recovery rate on default, in [0, 1].
        coupon_spread: Contractual running spread (decimal, e.g. 0.01 = 100bp).
        payment_freq: Premium payments per year (default 4 = quarterly).
        side: Protection side held (BUY = long protection).
    """

    notional: float
    maturity: float
    recovery_rate: float
    coupon_spread: float = 0.0
    payment_freq: int = 4
    side: ProtectionSide = ProtectionSide.BUY

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.notional <= 0:
            raise ValidationError(f"Notional must be positive, got {self.notional}")
        if self.maturity <= 0:
            raise ValidationError(f"Maturity must be positive, got {self.maturity}")
        if not 0.0 <= self.recovery_rate <= 1.0:
            raise ValidationError(
                f"Recovery rate must be in [0, 1], got {self.recovery_rate}"
            )
        if self.coupon_spread < 0:
            raise ValidationError(
                f"Coupon spread must be non-negative, got {self.coupon_spread}"
            )
        if self.payment_freq <= 0:
            raise ValidationError(
                f"Payment frequency must be positive, got {self.payment_freq}"
            )
        if not isinstance(self.side, ProtectionSide):
            raise ValidationError("side must be a ProtectionSide")

    @property
    def side_sign(self) -> float:
        """+1 for protection buyer, -1 for protection seller."""
        return 1.0 if self.side == ProtectionSide.BUY else -1.0
