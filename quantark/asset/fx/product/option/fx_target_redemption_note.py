"""
FX Target Redemption Note (digital-coupon TARN).

A note paying a **digital (cash-or-nothing) coupon** each period when the FX
fixing satisfies a condition, with a cumulative **coupon target**: once the
accumulated coupon reaches the target the note redeems (early), repaying
principal on that period's pay date; if the target is never reached the note
redeems at final maturity.

Per fixing ``t_i``:

    coupon_i = notional * coupon_rate * accrual_factor_i   if condition holds
             = 0                                            otherwise

The condition is either **one-sided** (``S_i >= strike`` when ``is_above``, else
``S_i <= strike``) or a **band** (``lower <= S_i <= upper``), both inclusive
(closed intervals). Coupons are domestic cash-or-nothing.

``target`` caps the cumulative coupon in domestic currency; on the period where
the cumulative coupon first reaches ``target`` the coupon is clipped so the
cumulative coupon equals ``target`` exactly, then the note redeems.

Principal handling is a config flag:
  - ``include_principal=True`` (default): the priced PV includes ``notional``
    discounted from the (path-dependent) redemption pay date.
  - ``include_principal=False``: only the coupon leg is priced (a digital coupon
    strip); the PV is **not** a full note value.

Path-dependent (the redemption couples the fixings) → Monte Carlo only;
priced by :class:`FxTargetRedemptionNoteMCEngine`.
"""

import math
from typing import List, Optional

from quantark.util.exceptions import ValidationError
from ..base_fx_product import BaseFxProduct
from ..currency_pair import CurrencyPair


class FxTargetRedemptionNote(BaseFxProduct):
    """
    FX Target Redemption Note (digital coupon strip with target redemption).

    Attributes:
        fixing_times: Ascending, unique, strictly-positive year-fraction fixing
            times.
        coupon_rate: Annualized coupon rate applied per period.
        notional: Note notional (domestic currency).
        target: Cumulative coupon-amount cap in domestic currency; ``math.inf``
            disables redemption.
        strike: One-sided coupon barrier (with ``is_above``); mutually exclusive
            with the ``lower``/``upper`` band.
        is_above: For the one-sided condition, pay when ``S_i >= strike`` (True)
            or ``S_i <= strike`` (False).
        lower, upper: Band coupon condition ``lower <= S_i <= upper``.
        accrual_factors: Per-period accrual factors (defaults to consecutive
            fixing-time differences with ``t_0 = 0``).
        pay_times: Per-fixing settlement times (defaults to ``fixing_times``).
        include_principal: Include discounted principal redemption in the PV.
    """

    def __init__(
        self,
        fixing_times: List[float],
        coupon_rate: float,
        notional: float = 1.0,
        target: float = math.inf,
        strike: Optional[float] = None,
        is_above: bool = True,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        accrual_factors: Optional[List[float]] = None,
        pay_times: Optional[List[float]] = None,
        include_principal: bool = True,
        currency_pair: Optional[CurrencyPair] = None,
    ):
        fixings = [float(t) for t in fixing_times]
        pays = [float(t) for t in pay_times] if pay_times is not None else list(fixings)
        if accrual_factors is not None:
            accruals = [float(a) for a in accrual_factors]
        else:
            prev = [0.0] + fixings[:-1]
            accruals = [t - p for t, p in zip(fixings, prev)]
        super().__init__(
            currency_pair=currency_pair,
            maturity=fixings[-1] if fixings else None,
            delivery=max(pays) if pays else None,
        )
        self.fixing_times = fixings
        self.pay_times = pays
        self.accrual_factors = accruals
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.target = float(target)
        self.strike = strike
        self.is_above = bool(is_above)
        self.lower = lower
        self.upper = upper
        self.include_principal = bool(include_principal)
        self.validate()

    def validate(self) -> None:
        for name, v in (("notional", self.notional), ("coupon_rate", self.coupon_rate)):
            if not math.isfinite(v) or v <= 0:
                raise ValidationError(f"{name} must be positive and finite, got {v}")
        if not (self.target > 0):  # rejects 0, negatives, NaN; inf passes
            raise ValidationError(f"target must be positive, got {self.target}")
        self._validate_condition()
        self._validate_schedule()

    def _validate_condition(self) -> None:
        band = self.lower is not None or self.upper is not None
        one_sided = self.strike is not None
        if band and one_sided:
            raise ValidationError(
                "specify either a one-sided strike or a band, not both"
            )
        if not band and not one_sided:
            raise ValidationError("a coupon condition (strike or band) is required")
        if one_sided and (not math.isfinite(self.strike) or self.strike <= 0):
            raise ValidationError(
                f"strike must be positive and finite, got {self.strike}"
            )
        if band:
            if self.lower is None or self.upper is None:
                raise ValidationError("band requires both lower and upper")
            if not math.isfinite(self.lower) or not math.isfinite(self.upper):
                raise ValidationError("band bounds must be finite")
            if self.lower <= 0 or self.upper <= 0:
                raise ValidationError("band bounds must be positive")
            if self.lower >= self.upper:
                raise ValidationError("band requires lower < upper")

    def _validate_schedule(self) -> None:
        fixings, pays, accruals = self.fixing_times, self.pay_times, self.accrual_factors
        if not fixings:
            raise ValidationError("fixing_times must be non-empty")
        if list(fixings) != sorted(fixings):
            raise ValidationError("fixing_times must be ascending")
        if len(set(fixings)) != len(fixings):
            raise ValidationError("fixing_times must be unique")
        if any((not math.isfinite(t)) or t <= 0.0 for t in fixings):
            raise ValidationError("fixing_times must be positive and finite")
        if len(pays) != len(fixings):
            raise ValidationError("pay_times must align 1:1 with fixing_times")
        for tf, tp in zip(fixings, pays):
            if not math.isfinite(tp):
                raise ValidationError(f"pay time {tp} must be finite")
            if tp < tf:
                raise ValidationError(f"pay time {tp} precedes its fixing {tf}")
        if list(pays) != sorted(pays):
            raise ValidationError("pay_times must be non-decreasing")
        if len(accruals) != len(fixings):
            raise ValidationError("accrual_factors must align 1:1 with fixing_times")
        if any((not math.isfinite(a)) or a <= 0.0 for a in accruals):
            raise ValidationError("accrual_factors must be positive and finite")

    @property
    def is_band(self) -> bool:
        return self.strike is None

    @property
    def is_unbounded_target(self) -> bool:
        return math.isinf(self.target)

    def full_coupon(self, period: int) -> float:
        """Undiscounted coupon paid in period ``period`` if the condition holds."""
        return self.notional * self.coupon_rate * self.accrual_factors[period]

    def coupon_condition(self, spot: float) -> bool:
        """True when ``spot`` satisfies the (inclusive) coupon condition."""
        if self.is_band:
            return self.lower <= spot <= self.upper
        return spot >= self.strike if self.is_above else spot <= self.strike

    def time_shift(self, time_bump: float) -> None:
        """Roll fixing/pay times (and maturity/delivery) forward for theta."""
        super().time_shift(time_bump)
        self.fixing_times = [t - time_bump for t in self.fixing_times]
        self.pay_times = [t - time_bump for t in self.pay_times]

    def get_payoff(self, spot: float) -> float:
        """First-period coupon at ``spot`` (path-independent utility)."""
        return self.full_coupon(0) if self.coupon_condition(spot) else 0.0

    def __repr__(self):
        tgt = "inf" if self.is_unbounded_target else f"{self.target:.4g}"
        cond = (
            f"[{self.lower:.4f},{self.upper:.4f}]"
            if self.is_band
            else f"{'>=' if self.is_above else '<='}{self.strike:.4f}"
        )
        return (
            f"FxTargetRedemptionNote({self.currency_pair}, cond S{cond}, "
            f"n_fix={len(self.fixing_times)}, target={tgt}, "
            f"principal={self.include_principal})"
        )
