"""Digital Coupon Note (DCN) autocallable product (spec WP1.2).

Payoff semantics (problem §3.1): daily KI (sticky) gates all future fixed
coupons; monthly KO has priority over the fixed coupon on its date; loss leg
pays -(N/S0)·part·max(K_loss - S_T, 0) at the independent settlement_date.
PV sign: direction_sign (+1 BUYER / -1 SELLER) applied uniformly to the PV
and every leg PV (mirrors the cashleg buyer-perspective sign convention).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import validate_positive

from .dcn_schedule import DCNSchedule

_DAYS_PER_YEAR = 365.0  # ACT/365F fixed by the problem


class DCNDirection(Enum):
    """Trade side; PV is reported from the buyer's perspective times sign."""

    BUYER = 1
    SELLER = -1


class DCNOption(BaseEquityProduct):
    """DCN product: instrument fields + derived absolute levels.

    Payoff evaluation stays engine-side (the path kernel in
    engine/mc/dcn_payoff.py and the PDE events in
    engine/pde/dcn_pde_solver.py); the BaseEquityProduct hooks below cover
    the repository-wide product contract (unified Greeks, dispatch).
    """

    def __init__(
        self,
        notional: float,
        initial_price: float,
        direction: DCNDirection,
        coupon_barrier_ratio: float,
        ko_barrier_ratio: float,
        ki_barrier_ratio: float,
        ki_put_strike_ratio: float,
        coupon_rate: float,
        ko_coupon_rate: float,
        participation: float,
        coupon_counted_days: int,
        coupon_days_denom: int,
        schedule: DCNSchedule,
        settlement_date: datetime,
        knocked_in_at_valuation: bool = False,
    ):
        self.notional = float(notional)
        self.initial_price = float(initial_price)
        self.direction = direction
        self.coupon_barrier_ratio = float(coupon_barrier_ratio)
        self.ko_barrier_ratio = float(ko_barrier_ratio)
        self.ki_barrier_ratio = float(ki_barrier_ratio)
        self.ki_put_strike_ratio = float(ki_put_strike_ratio)
        self.coupon_rate = float(coupon_rate)
        self.ko_coupon_rate = float(ko_coupon_rate)
        self.participation = float(participation)
        self.coupon_counted_days = int(coupon_counted_days)
        self.coupon_days_denom = int(coupon_days_denom)
        self.schedule = schedule
        self.settlement_date = settlement_date
        self.knocked_in_at_valuation = bool(knocked_in_at_valuation)
        self.validate()

    # -- derived levels (ratios are the inputs; levels never independent) --
    @property
    def coupon_barrier(self) -> float:
        return self.coupon_barrier_ratio * self.initial_price

    @property
    def ko_barrier(self) -> float:
        return self.ko_barrier_ratio * self.initial_price

    @property
    def ki_barrier(self) -> float:
        return self.ki_barrier_ratio * self.initial_price

    @property
    def k_loss(self) -> float:
        return self.ki_put_strike_ratio * self.initial_price

    @property
    def accrual_per_period(self) -> float:
        # 30/360 by contract: deliberately NOT derived from actual obs dates
        return self.coupon_counted_days / float(self.coupon_days_denom)

    @property
    def direction_sign(self) -> float:
        return float(self.direction.value)

    # -- BaseEquityProduct contract -----------------------------------------
    def get_maturity(self, pricing_env=None) -> float:
        """ACT/365F years from the valuation date to the final observation."""
        s = self.schedule
        return (
            s.monthly[-1].observation_date - s.valuation_date
        ).days / _DAYS_PER_YEAR

    def get_payoff(self, spot: float, knocked_in: bool = False) -> float:
        """Final-observation-date cashflow EXCLUDING coupons (signed).

        The DCN payoff is path-dependent; this hook returns only the
        terminal loss-leg amount for a given knocked-in state (the piece a
        terminal-spot payoff can express): 0 when never knocked in, else
        -(N/S0)*participation*max(K_loss - spot, 0), times direction_sign.
        Engines never call this — they use the path kernel.
        """
        if not knocked_in:
            return 0.0
        return (
            self.direction_sign
            * -(self.notional / self.initial_price)
            * self.participation
            * max(self.k_loss - float(spot), 0.0)
        )

    def validate(self) -> None:
        validate_positive(self.notional, "notional")
        validate_positive(self.initial_price, "initial_price")
        validate_positive(self.participation, "participation")
        validate_positive(self.ki_put_strike_ratio, "ki_put_strike_ratio")
        for name in ("coupon_barrier_ratio", "ko_barrier_ratio",
                     "ki_barrier_ratio"):
            validate_positive(getattr(self, name), name)
        if self.coupon_rate < 0 or self.ko_coupon_rate < 0:
            raise ValidationError("coupon rates must be >= 0")
        if self.coupon_counted_days <= 0 or self.coupon_days_denom <= 0:
            raise ValidationError("accrual day counts must be positive")
        if not (self.ki_barrier_ratio < self.coupon_barrier_ratio
                and self.ki_barrier_ratio < self.ko_barrier_ratio):
            raise ValidationError(
                "ki_barrier_ratio must be strictly below both the coupon "
                "and KO barrier ratios"
            )
        if not isinstance(self.direction, DCNDirection):
            raise ValidationError(f"invalid direction: {self.direction!r}")
        last_obs = self.schedule.monthly[-1].observation_date
        if self.settlement_date < last_obs:
            raise ValidationError(
                "loss-leg settlement_date before final observation date"
            )
