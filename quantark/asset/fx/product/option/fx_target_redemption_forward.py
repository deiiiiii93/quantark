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
from datetime import datetime
from typing import List, Optional, Tuple

from quantark.util.exceptions import ValidationError
from ..currency_pair import CurrencyPair
from .target_redemption_base import TargetRedemptionProduct
from .target_redemption_state import TargetRedemptionState


class FxTargetRedemptionForward(TargetRedemptionProduct):
    """
    FX Target Redemption Forward (geared forward strip with target redemption).

    Attributes:
        strike: Forward strike ``K`` (quote per base).
        fixing_times: Ascending, unique year-fraction fixing times of the
            fixings not yet banked; a non-positive time is an unbanked past
            fixing. An empty list is a terminal booking.
        pay_times: Per-fixing settlement times (defaults to ``fixing_times``);
            each must be on or after its fixing.
        notional: Base-currency notional (cashflows are in domestic currency).
        target: Cumulative favorable-amount cap in undiscounted domestic
            currency; ``math.inf`` disables redemption (a plain geared strip).
        gearing: Multiplier on the unfavorable side (default 1.0).
        is_buy_base: True buys the base ccy at ``K`` (favorable when ``S>K``);
            False sells it (favorable when ``S<K``).
        state: Mandatory realized state (``TargetRedemptionState(0.0)`` for a
            contract booked at inception).
        fixing_dates, pay_dates: Dated form (exclusive with ``fixing_times`` /
            ``pay_times``): the whole contractual schedule; ``state.as_of`` is
            the date through which fixings are banked.
    """

    def __init__(
        self,
        strike: float,
        fixing_times: Optional[List[float]] = None,
        notional: float = 1.0,
        target: float = math.inf,
        gearing: float = 1.0,
        is_buy_base: bool = True,
        pay_times: Optional[List[float]] = None,
        currency_pair: Optional[CurrencyPair] = None,
        *,
        state: Optional[TargetRedemptionState] = None,
        fixing_dates: Optional[List[datetime]] = None,
        pay_dates: Optional[List[datetime]] = None,
    ):
        self.strike = strike
        self.notional = notional
        self.target = float(target)
        self.gearing = gearing
        self.is_buy_base = bool(is_buy_base)
        state = self._require_state(state)
        if self._schedule_form(fixing_times, fixing_dates):
            if pay_times is not None:
                raise ValidationError("pay_times belongs to the numeric form; a dated TARF takes pay_dates")
            self._init_dated(fixing_dates, pay_dates, currency_pair, state)
            self.validate()
            self._refresh_dated_horizon()
        else:
            if pay_dates is not None:
                raise ValidationError("pay_dates belongs to the dated form; a numeric TARF takes pay_times")
            self._init_numeric([float(t) for t in fixing_times], pay_times, currency_pair, state)
            self.validate()
            self._refresh_numeric_horizon()
        self._warn_if_terminal_booking()

    def _validate_terms(self) -> None:
        for name, v in (("strike", self.strike), ("notional", self.notional)):
            if not math.isfinite(v) or v <= 0:
                raise ValidationError(f"{name} must be positive and finite, got {v}")
        if not math.isfinite(self.gearing) or self.gearing < 0:
            raise ValidationError(
                f"gearing must be non-negative and finite, got {self.gearing}"
            )
        if not (self.target > 0):  # rejects 0, negatives, and NaN; inf passes
            raise ValidationError(f"target must be positive, got {self.target}")

    def _fixing_outcome(self, factor, spot, accumulated) -> Tuple[float, float]:
        """The engine's arithmetic for one fixing: favourable amount clipped to the remaining target, net cash."""
        sign = 1.0 if self.is_buy_base else -1.0
        diff = sign * (spot - self.strike)
        fav = self.notional * max(diff, 0.0)
        unf = self.notional * max(-diff, 0.0)
        fav_eff = min(fav, self.target - accumulated)
        return fav_eff, fav_eff - self.gearing * unf

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

    def __repr__(self):
        tgt = "inf" if self.is_unbounded_target else f"{self.target:.4g}"
        side = "buy" if self.is_buy_base else "sell"
        return (
            f"FxTargetRedemptionForward({self.currency_pair}, {side}-base, "
            f"K={self.strike:.6f}, n_fix={len(self.fixing_times if self.is_numeric else self.fixing_dates)}, "
            f"target={tgt}, gearing={self.gearing})"
        )
