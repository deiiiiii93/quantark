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

Accrual: numeric form - the first listed period starts at ``accrual_start`` (or the caller states explicit
factors); dated form - period ``i`` runs from ``initial_date`` / the previous fixing date to this fixing date in the
contract ``day_count_convention`` (default ACT/365), or explicit factors. No start is assumed. The contract carries
its realized ``state``.

Path-dependent (the redemption couples the fixings) → Monte Carlo only;
priced by :class:`FxTargetRedemptionNoteMCEngine`.
"""

import math
from datetime import datetime
from typing import List, Optional, Tuple

from quantark.util.calendar import DayCountConvention, calculate_year_fraction
from quantark.util.exceptions import ValidationError
from ..currency_pair import CurrencyPair
from .target_redemption_base import TargetRedemptionProduct
from .target_redemption_state import TargetRedemptionState


class FxTargetRedemptionNote(TargetRedemptionProduct):
    """
    FX Target Redemption Note (digital coupon strip with target redemption).

    Numeric form: ``fixing_times`` / ``pay_times`` are years from valuation and list only the fixings not yet
    banked (a non-positive time is an unbanked past fixing; an empty list is a terminal booking).
    Dated form: ``fixing_dates`` / ``pay_dates`` list the whole contractual schedule; ``state.fixings_banked`` counts
    the banked fixings (a booking stating only ``state.as_of`` banks those dated on or before it); times and factors
    resolve against the pricing environment.

    Attributes:
        fixing_times: Ascending (non-decreasing: periods may share a fixing time), finite year-fraction fixing
            times.
        coupon_rate: Annualized coupon rate applied per period.
        notional: Note notional (domestic currency).
        target: Cumulative coupon-amount cap in domestic currency; ``math.inf`` disables redemption.
        strike: One-sided coupon barrier (with ``is_above``); mutually exclusive with the ``lower``/``upper`` band.
        is_above: For the one-sided condition, pay when ``S_i >= strike`` (True) or ``S_i <= strike`` (False).
        lower, upper: Band coupon condition ``lower <= S_i <= upper``.
        accrual_factors: Explicit per-listed-period factors; or ``accrual_start``, the time the first listed
            period began (years from valuation, negative when before) - exactly one is required.
        pay_times: Per-fixing settlement times (defaults to ``fixing_times``).
        include_principal: Include discounted principal redemption in the PV.
        state: Mandatory ``TargetRedemptionState`` (``TargetRedemptionState(0.0)`` at inception).
        fixing_dates, pay_dates: Dated schedule (exclusive with ``fixing_times`` / ``pay_times``).
        initial_date: Dated form: start of the first period (exclusive with ``accrual_factors``).
        day_count_convention: Dated form: the coupon accrual day count for ``initial_date``-derived factors
            (``None`` = ACT/365).
    """

    _has_accrual = True
    _pending_must_be_positive = True

    def __init__(
        self,
        fixing_times: Optional[List[float]] = None,
        coupon_rate: Optional[float] = None,
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
        *,
        state: Optional[TargetRedemptionState] = None,
        accrual_start: Optional[float] = None,
        fixing_dates: Optional[List[datetime]] = None,
        pay_dates: Optional[List[datetime]] = None,
        initial_date: Optional[datetime] = None,
        day_count_convention: Optional[DayCountConvention] = None,
    ):
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.target = float(target)
        self.strike = strike
        self.is_above = bool(is_above)
        self.lower = lower
        self.upper = upper
        self.include_principal = bool(include_principal)
        state = self._require_state(state)
        if self._schedule_form(fixing_times, fixing_dates):
            for name, value in (("pay_times", pay_times), ("accrual_start", accrual_start)):
                if value is not None:
                    raise ValidationError(
                        f"{name} belongs to the numeric form; a dated TARN takes pay_dates and initial_date")
            if initial_date is None and accrual_factors is None:
                raise ValidationError(
                    "a dated TARN derives its periods from initial_date or states explicit accrual_factors")
            if initial_date is not None and accrual_factors is not None:
                raise ValidationError("give initial_date or accrual_factors, not both")
            if accrual_factors is not None and day_count_convention is not None:
                raise ValidationError(
                    "day_count_convention sets the derived accrual factors; it cannot be combined with explicit "
                    "accrual_factors")
            self.initial_date = initial_date
            self.day_count_convention = day_count_convention
            self.accrual_factors = [float(a) for a in accrual_factors] if accrual_factors is not None else None
            self._init_dated(fixing_dates, pay_dates, currency_pair, state)
            self.validate()
            self._refresh_dated_horizon()
        else:
            for name, value in (("pay_dates", pay_dates), ("initial_date", initial_date),
                                ("day_count_convention", day_count_convention)):
                if value is not None:
                    raise ValidationError(
                        f"{name} belongs to the dated form; a numeric TARN takes pay_times and accrual_start")
            self.initial_date = None
            self.day_count_convention = None
            fixings = [float(t) for t in fixing_times]
            self.accrual_factors = self._numeric_accruals(fixings, accrual_start, accrual_factors, state)
            self._init_numeric(fixings, pay_times, currency_pair, state)
            self.validate()
            self._refresh_numeric_horizon()
        self._warn_if_terminal_booking()

    @staticmethod
    def _numeric_accruals(fixings, accrual_start, accrual_factors, state) -> List[float]:
        """The listed periods' factors: explicit, or ``t_i - t_{i-1}`` with ``t_{-1} = accrual_start`` (spec Q2a)."""
        if accrual_start is None and accrual_factors is None:
            raise ValidationError(
                "a numeric TARN must state when its first listed period began: pass accrual_start (years from "
                "valuation, negative when it began before) or explicit accrual_factors - no start is assumed"
            )
        if accrual_start is not None and accrual_factors is not None:
            raise ValidationError("give accrual_start or accrual_factors, not both")
        if accrual_factors is not None:
            return [float(a) for a in accrual_factors]
        start = float(accrual_start)
        if not math.isfinite(start):
            raise ValidationError(f"accrual_start must be finite, got {start}")
        if fixings and not start < fixings[0]:
            raise ValidationError(f"accrual_start {start} must be before the first listed fixing {fixings[0]}")
        if start > 0.0 and state.accumulated > 0.0:
            raise ValidationError(
                "accrual_start > 0 with accumulated > 0: periods are contiguous, so a contract that has banked a "
                "fixing has a listed period that began at or before valuation"
            )
        prev = [start] + fixings[:-1]
        return [t - p for t, p in zip(fixings, prev)]

    def _validate_dated_accrual(self) -> None:
        if self.initial_date is not None and not self.initial_date < self.fixing_dates[0]:
            raise ValidationError(
                f"initial_date {self.initial_date} must be before the first fixing date {self.fixing_dates[0]}")
        if self.initial_date is not None and len(set(self.fixing_dates)) != len(self.fixing_dates):
            # user ruling 2026-10-10 on R26 (b): the DEFAULT derivation's later period on a shared fixing date runs
            # from that date to itself; a contract stating its own accrual is not refused (Task 8d narrows this
            # condition to the default derivation and adds accrual_end_dates to the message)
            raise ValidationError(
                "periods sharing a fixing date need their own accrual: under the default derivation (initial_date, "
                "then each previous fixing date) the later period would have zero length - state accrual_factors")
        if self.accrual_factors is not None:
            if len(self.accrual_factors) != len(self.fixing_dates):
                raise ValidationError("accrual_factors must align 1:1 with fixing_dates")
            if any((not math.isfinite(a)) or a <= 0.0 for a in self.accrual_factors):
                raise ValidationError("accrual_factors must be positive and finite")

    def _dated_factor(self, index: int, fx_env) -> float:
        """Explicit factor, else ``initial_date`` / previous fixing date -> this fixing date in the contract day count
        (``None`` = ACT_365), resolved against the environment's calendar (a BUSINESS_DAYS count needs it)."""
        if self.accrual_factors is not None:
            return self.accrual_factors[index]
        start = self.initial_date if index == 0 else self.fixing_dates[index - 1]
        convention = DayCountConvention.ACT_365 if self.day_count_convention is None else self.day_count_convention
        return calculate_year_fraction(start, self.fixing_dates[index], convention, fx_env.bus_days_in_year,
                                       calendar=fx_env.calendar)

    def _validate_terms(self) -> None:
        if self.coupon_rate is None:
            raise ValidationError("coupon_rate is required")
        for name, v in (("notional", self.notional), ("coupon_rate", self.coupon_rate)):
            if not math.isfinite(v) or v <= 0:
                raise ValidationError(f"{name} must be positive and finite, got {v}")
        if not (self.target > 0):  # rejects 0, negatives, NaN; inf passes
            raise ValidationError(f"target must be positive, got {self.target}")
        self._validate_condition()

    def coupon_amount(self, factor: float) -> float:
        """Undiscounted coupon of a period with accrual ``factor`` when the condition holds."""
        return self.notional * self.coupon_rate * factor

    def full_coupon(self, period: int) -> float:
        """Undiscounted coupon of listed period ``period`` if the condition holds."""
        if self.accrual_factors is None:
            raise ValidationError(
                "a dated TARN without explicit accrual_factors resolves its factors against an environment: use "
                "resolve(fx_env).accrual_factors")
        return self.coupon_amount(self.accrual_factors[period])

    def _fixing_outcome(self, factor, spot, accumulated) -> Tuple[float, float]:
        coupon = self.coupon_amount(factor) if self.coupon_condition(spot) else 0.0
        c = min(coupon, self.target - accumulated)
        return c, c

    def _principal_amount(self) -> float:
        return self.notional if self.include_principal else 0.0

    def get_payoff(self, spot: float) -> float:
        """First listed (dated: scheduled) period's coupon at ``spot`` (path-independent utility)."""
        if self.is_numeric and not self.accrual_factors:
            raise ValidationError("a terminal TARN has no next coupon")
        return self.full_coupon(0) if self.coupon_condition(spot) else 0.0

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

    @property
    def is_band(self) -> bool:
        return self.strike is None

    def coupon_condition(self, spot: float) -> bool:
        """True when ``spot`` satisfies the (inclusive) coupon condition."""
        if self.is_band:
            return self.lower <= spot <= self.upper
        return spot >= self.strike if self.is_above else spot <= self.strike

    def __repr__(self):
        tgt = "inf" if self.is_unbounded_target else f"{self.target:.4g}"
        cond = (
            f"[{self.lower:.4f},{self.upper:.4f}]"
            if self.is_band
            else f"{'>=' if self.is_above else '<='}{self.strike:.4f}"
        )
        return (
            f"FxTargetRedemptionNote({self.currency_pair}, cond S{cond}, "
            f"n_fix={len(self.fixing_times if self.is_numeric else self.fixing_dates)}, target={tgt}, "
            f"principal={self.include_principal})"
        )
