"""
FX Target Redemption Forward (TARF).

A strip of periodic FX forwards struck at ``K``, with a *target redemption*
feature: the holder accumulates the **favorable** amount each fixing and the
structure terminates ("redeems") once that cumulative favorable amount reaches a
contractual ``target``. The unfavorable side may be **geared** (the classic
leveraged TARF), which is what gives the product its asymmetric risk profile.

Per fixing ``t_i`` (buy-base convention; sell-base flips the sign of ``S_i-K``):

    favorable_i   = notional * max(S_i - K, 0)        (accrues toward target)
    unfavorable_i = notional * max(K - S_i, 0)
    CF_i          = favorable_i - gearing * unfavorable_i   (settled at pay_i)

Target accumulation is on the **favorable amount only**, in **undiscounted
domestic currency** (``acc = sum favorable_i``). On the period where ``acc`` first
reaches ``target`` the favorable amount is clipped so the cumulative favorable
equals ``target`` exactly; that period's geared unfavorable cashflow is still
paid, then the deal redeems and no further fixings occur.

Path-dependent (the redemption couples the fixings) → Monte Carlo only;
priced by :class:`FxTarnForwardMCEngine`.
"""

import math
from typing import List, Optional

from quantark.util.exceptions import ValidationError
from ..base_fx_product import BaseFxProduct
from ..currency_pair import CurrencyPair


class FxTargetRedemptionForward(BaseFxProduct):
    """
    FX Target Redemption Forward (geared forward strip with target redemption).

    Attributes:
        strike: Forward strike ``K`` (quote per base).
        fixing_times: Ascending, unique, strictly-positive year-fraction fixing
            times.
        pay_times: Per-fixing settlement times (defaults to ``fixing_times``);
            each must be on or after its fixing.
        notional: Base-currency notional (cashflows are in domestic currency).
        target: Cumulative favorable-amount cap in undiscounted domestic
            currency; ``math.inf`` disables redemption (a plain geared strip).
        gearing: Multiplier on the unfavorable side (default 1.0).
        is_buy_base: True buys the base ccy at ``K`` (favorable when ``S>K``);
            False sells it (favorable when ``S<K``).
    """

    def __init__(
        self,
        strike: float,
        fixing_times: List[float],
        notional: float = 1.0,
        target: float = math.inf,
        gearing: float = 1.0,
        is_buy_base: bool = True,
        pay_times: Optional[List[float]] = None,
        currency_pair: Optional[CurrencyPair] = None,
    ):
        fixings = [float(t) for t in fixing_times]
        pays = [float(t) for t in pay_times] if pay_times is not None else list(fixings)
        super().__init__(
            currency_pair=currency_pair,
            maturity=fixings[-1] if fixings else None,
            delivery=max(pays) if pays else None,
        )
        self.strike = strike
        self.fixing_times = fixings
        self.pay_times = pays
        self.notional = notional
        self.target = float(target)
        self.gearing = gearing
        self.is_buy_base = bool(is_buy_base)
        self.validate()

    def validate(self) -> None:
        for name, v in (("strike", self.strike), ("notional", self.notional)):
            if not math.isfinite(v) or v <= 0:
                raise ValidationError(f"{name} must be positive and finite, got {v}")
        if not math.isfinite(self.gearing) or self.gearing < 0:
            raise ValidationError(
                f"gearing must be non-negative and finite, got {self.gearing}"
            )
        if not (self.target > 0):  # rejects 0, negatives, and NaN; inf passes
            raise ValidationError(f"target must be positive, got {self.target}")
        self._validate_schedule()

    def _validate_schedule(self) -> None:
        fixings, pays = self.fixing_times, self.pay_times
        if not fixings:
            raise ValidationError("fixing_times must be non-empty")
        if list(fixings) != sorted(fixings):
            raise ValidationError("fixing_times must be ascending")
        if len(set(fixings)) != len(fixings):
            raise ValidationError("fixing_times must be unique")
        if any(t <= 0.0 for t in fixings):
            raise ValidationError("fixing_times must be strictly positive")
        if len(pays) != len(fixings):
            raise ValidationError("pay_times must align 1:1 with fixing_times")
        for tf, tp in zip(fixings, pays):
            if tp < tf:
                raise ValidationError(
                    f"pay time {tp} precedes its fixing {tf}"
                )
        if list(pays) != sorted(pays):
            raise ValidationError("pay_times must be non-decreasing")

    def time_shift(self, time_bump: float) -> None:
        """Roll fixing/pay times (and maturity/delivery) forward for theta."""
        super().time_shift(time_bump)
        self.fixing_times = [t - time_bump for t in self.fixing_times]
        self.pay_times = [t - time_bump for t in self.pay_times]

    def period_cashflow(self, spot: float) -> float:
        """Undiscounted single-fixing net cashflow (favorable - geared unfavorable).

        Path-independent utility; the redemption/target logic lives in the engine.
        """
        if self.is_buy_base:
            fav, unf = max(spot - self.strike, 0.0), max(self.strike - spot, 0.0)
        else:
            fav, unf = max(self.strike - spot, 0.0), max(spot - self.strike, 0.0)
        return self.notional * (fav - self.gearing * unf)

    def get_payoff(self, spot: float) -> float:
        """Per-fixing net cashflow at ``spot`` (see :meth:`period_cashflow`)."""
        return self.period_cashflow(spot)

    @property
    def is_unbounded_target(self) -> bool:
        return math.isinf(self.target)

    def __repr__(self):
        tgt = "inf" if self.is_unbounded_target else f"{self.target:.4g}"
        side = "buy" if self.is_buy_base else "sell"
        return (
            f"FxTargetRedemptionForward({self.currency_pair}, {side}-base, "
            f"K={self.strike:.6f}, n_fix={len(self.fixing_times)}, "
            f"target={tgt}, gearing={self.gearing})"
        )
