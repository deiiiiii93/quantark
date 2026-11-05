"""Instrument quotes for curve construction.

Spec: ``.ship/2026-11-05-curve-construction/spec.md`` §Contracts ``quotes.py``.

Every quote is a frozen dataclass that carries **exactly one clock**: either
``tenor_years`` (numeric axis, bootstrap run with ``valuation_date=None``) or
dates (dated axis, bootstrap run with a ``valuation_date``). Mixing raises
``ValidationError`` — the repo's date-vs-numeric clock rule. Pillar times are
always year fractions on the calendar ACT/365F axis ``(date − valuation).days/365``,
the convention of the rate and bond engines.

Dated swap and bond quotes build the repo's own products (``to_swap`` →
``InterestRateSwap``, ``to_bond`` → ``FixedBond``) so the bootstrapper strips,
the engine-consistency tests and the QuantLib cross-check all price the very
same cash flows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import ClassVar, List, Optional, Sequence, Union

from quantark.param.index import RateIndex
from quantark.param.rrf.projection import act365f
from quantark.util.calendar import BusinessDayConvention, CalendarType, DayCountConvention
from quantark.util.enum import PaymentFrequency
from quantark.util.exceptions import ValidationError

PERIOD_COUNT_ATOL: float = 1e-9
"""A numeric swap/bond tenor must be a whole number of coupon periods within this."""


# --------------------------------------------------------------------------- helpers


def _finite(value, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return out


def _check_clock(cls_name: str, tenor_years, dates: dict) -> None:
    has_tenor = tenor_years is not None
    has_dates = any(v is not None for v in dates.values())
    if has_tenor == has_dates:
        names = ", ".join(dates)
        raise ValidationError(
            f"{cls_name} needs exactly one clock: tenor_years (numeric) or dates ({names}); "
            f"got tenor_years={tenor_years!r}, dates={ {k: v for k, v in dates.items() if v is not None} }"
        )
    if has_tenor:
        t = _finite(tenor_years, f"{cls_name}.tenor_years")
        if t <= 0.0:
            raise ValidationError(f"{cls_name}.tenor_years must be positive, got {t}")


def whole_periods(tenor_years: float, frequency: PaymentFrequency) -> int:
    """Number of coupon periods in ``tenor_years``; raises unless it is a whole number."""
    n = float(tenor_years) * frequency.value
    k = int(round(n))
    if k < 1 or abs(n - k) > PERIOD_COUNT_ATOL:
        raise ValidationError(
            f"tenor {tenor_years} years is not a whole number of {frequency.name} periods"
        )
    return k


def _run_clock_matches(quote, valuation_date: Optional[datetime]) -> None:
    if quote.is_dated and valuation_date is None:
        raise ValidationError(f"{quote!r} is dated but the run has no valuation_date (numeric clock)")
    if not quote.is_dated and valuation_date is not None:
        raise ValidationError(f"{quote!r} is numeric but the run has valuation_date={valuation_date:%Y-%m-%d}")


def _after(valuation_date: datetime, date: datetime, what: str) -> float:
    t = act365f(valuation_date, date)
    if t <= 0.0:
        raise ValidationError(f"{what} {date:%Y-%m-%d} is not after the valuation date {valuation_date:%Y-%m-%d}")
    return t


# --------------------------------------------------------------------------- quotes


@dataclass(frozen=True)
class DepositQuote:
    """Money-market / repo deposit: a simple rate ``r`` over ``[start, maturity]``.

    ``DF(t_end) = DF(t_start) / (1 + r·τ)``; ``τ`` is ``tenor_years`` on the numeric
    axis or the day-count fraction of the dated period. ``start_date`` defaults to
    the valuation date. Default ``day_count`` ACT/365 (the axis convention);
    ACT/360 markets override.
    """

    rate: float
    tenor_years: Optional[float] = None
    start_date: Optional[datetime] = None
    maturity_date: Optional[datetime] = None
    day_count: DayCountConvention = DayCountConvention.ACT_365
    label: Optional[str] = None

    def __post_init__(self) -> None:
        _finite(self.rate, "DepositQuote.rate")
        _check_clock("DepositQuote", self.tenor_years, {"start_date": self.start_date, "maturity_date": self.maturity_date})
        if self.start_date is not None and self.maturity_date is None:
            raise ValidationError("DepositQuote.start_date needs a maturity_date")
        if not isinstance(self.day_count, DayCountConvention):
            raise ValidationError(f"day_count must be a DayCountConvention, got {self.day_count!r}")
        if self.tenor_years is not None and 1.0 + self.rate * self.tenor_years <= 0.0:
            raise ValidationError(
                f"1 + rate·tenor must be positive, got rate={self.rate}, tenor={self.tenor_years}"
            )

    @property
    def is_dated(self) -> bool:
        return self.maturity_date is not None

    def pillar_time(self, valuation_date: Optional[datetime]) -> float:
        _run_clock_matches(self, valuation_date)
        if not self.is_dated:
            return float(self.tenor_years)
        assert valuation_date is not None and self.maturity_date is not None
        return _after(valuation_date, self.maturity_date, "deposit maturity")


@dataclass(frozen=True)
class SwapQuote:
    """Par interest-rate swap: fixed ``fixed_rate`` against ``index`` flat.

    Numeric: ``tenor_years`` must be a whole number of ``payment_frequency`` periods
    (idealised schedule, ``dcf = 1/frequency`` on both legs). Dated: the swap runs
    from ``effective_date`` (default: the valuation date) to ``maturity_date`` on
    the quote's calendar and roll convention; day counts default to the index's.
    """

    fixed_rate: float
    index: RateIndex
    tenor_years: Optional[float] = None
    effective_date: Optional[datetime] = None
    maturity_date: Optional[datetime] = None
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY
    fixed_day_count: Optional[DayCountConvention] = None
    float_day_count: Optional[DayCountConvention] = None
    calendar_type: Optional[CalendarType] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    label: Optional[str] = None

    def __post_init__(self) -> None:
        _finite(self.fixed_rate, "SwapQuote.fixed_rate")
        if not isinstance(self.index, RateIndex):
            raise ValidationError(f"index must be a RateIndex, got {self.index!r}")
        if not isinstance(self.payment_frequency, PaymentFrequency):
            raise ValidationError(f"payment_frequency must be a PaymentFrequency, got {self.payment_frequency!r}")
        _check_clock("SwapQuote", self.tenor_years, {"effective_date": self.effective_date, "maturity_date": self.maturity_date})
        if self.effective_date is not None and self.maturity_date is None:
            raise ValidationError("SwapQuote.effective_date needs a maturity_date")
        if self.tenor_years is not None:
            whole_periods(self.tenor_years, self.payment_frequency)

    @property
    def is_dated(self) -> bool:
        return self.maturity_date is not None

    @property
    def resolved_fixed_day_count(self) -> DayCountConvention:
        return self.fixed_day_count or self.index.day_count_convention

    @property
    def resolved_float_day_count(self) -> DayCountConvention:
        return self.float_day_count or self.index.day_count_convention

    def periods(self) -> int:
        """Number of coupon periods (numeric quotes)."""
        if self.is_dated:
            raise ValidationError("periods() is defined for numeric quotes only")
        return whole_periods(float(self.tenor_years), self.payment_frequency)

    def to_swap(self, valuation_date: datetime):
        """The dated quote as an ``InterestRateSwap`` (payer, notional 1.0, spread 0).

        The legs receive a ``dataclasses.replace`` copy of the index, because
        ``FloatingLeg.__post_init__`` installs its own fixing store on the index it
        is given and must not overwrite the caller's shared ``RateIndex``.
        """
        if not self.is_dated:
            raise ValidationError("to_swap needs a dated SwapQuote")
        if valuation_date is None:
            raise ValidationError("to_swap needs the run's valuation_date")
        from quantark.asset.rate.product.irs import (  # local: keep quantark.param free of asset imports
            FixedLeg,
            FloatingLeg,
            InterestRateSwap,
            NotionalSchedule,
            SwapDirection,
        )
        from quantark.util.calendar import create_calendar

        effective = self.effective_date or valuation_date
        assert self.maturity_date is not None
        if self.maturity_date <= effective:
            raise ValidationError("SwapQuote.maturity_date must be after the effective date")
        calendar = create_calendar(self.calendar_type or self.index.calendar_type)
        notional = NotionalSchedule.constant(1.0)
        fixed = FixedLeg(
            start_date=effective,
            end_date=self.maturity_date,
            notional_schedule=notional,
            fixed_rate=self.fixed_rate,
            payment_frequency=self.payment_frequency,
            day_count_convention=self.resolved_fixed_day_count,
            calendar=calendar,
            business_day_convention=self.business_day_convention,
        )
        floating = FloatingLeg(
            start_date=effective,
            end_date=self.maturity_date,
            notional_schedule=notional,
            index=replace(self.index),
            spread=0.0,
            payment_frequency=self.payment_frequency,
            day_count_convention=self.resolved_float_day_count,
            calendar=calendar,
            business_day_convention=self.business_day_convention,
        )
        return InterestRateSwap(
            fixed_leg=fixed, floating_leg=floating, direction=SwapDirection.PAYER, effective_date=effective
        )

    def pillar_time(self, valuation_date: Optional[datetime]) -> float:
        """Last payment time: ``n/frequency`` numeric, the product's last payment dated."""
        _run_clock_matches(self, valuation_date)
        if not self.is_dated:
            return self.periods() / self.payment_frequency.value
        assert valuation_date is not None
        swap = self.to_swap(valuation_date)
        last = max(
            [cf.payment_date for cf in swap.fixed_leg.get_cashflows(valuation_date)]
            + [cf.payment_date for cf in swap.floating_leg.get_cashflows(valuation_date)]
        )
        return _after(valuation_date, last, "swap last payment")


@dataclass(frozen=True)
class BondQuote:
    """Fixed-coupon bond quoted as a clean price per 100 face.

    Numeric: ``tenor_years`` is a whole number of coupon periods, valued on a
    coupon date (accrued 0, so dirty = clean). Dated: ``issue_date`` and
    ``maturity_date`` anchor the schedule; dirty = clean + the bond's own accrued.
    """

    coupon_rate: float
    clean_price: float
    payment_frequency: PaymentFrequency
    tenor_years: Optional[float] = None
    issue_date: Optional[datetime] = None
    maturity_date: Optional[datetime] = None
    day_count: DayCountConvention = DayCountConvention.ACT_ACT_ISDA
    calendar_type: CalendarType = CalendarType.NONE
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    label: Optional[str] = None

    FACE: ClassVar[float] = 100.0

    def __post_init__(self) -> None:
        coupon = _finite(self.coupon_rate, "BondQuote.coupon_rate")
        if coupon < 0.0:
            raise ValidationError(f"coupon_rate must be non-negative, got {coupon}")
        price = _finite(self.clean_price, "BondQuote.clean_price")
        if price <= 0.0:
            raise ValidationError(f"clean_price must be positive, got {price}")
        if not isinstance(self.payment_frequency, PaymentFrequency):
            raise ValidationError(f"payment_frequency must be a PaymentFrequency, got {self.payment_frequency!r}")
        _check_clock("BondQuote", self.tenor_years, {"issue_date": self.issue_date, "maturity_date": self.maturity_date})
        if self.is_dated and (self.issue_date is None or self.maturity_date is None):
            raise ValidationError("a dated BondQuote needs both issue_date and maturity_date")
        if self.tenor_years is not None:
            whole_periods(self.tenor_years, self.payment_frequency)

    @property
    def is_dated(self) -> bool:
        return self.issue_date is not None or self.maturity_date is not None

    def periods(self) -> int:
        if self.is_dated:
            raise ValidationError("periods() is defined for numeric quotes only")
        return whole_periods(float(self.tenor_years), self.payment_frequency)

    def to_bond(self, valuation_date: datetime):
        """The dated quote as a ``FixedBond`` on 100 face."""
        if not self.is_dated:
            raise ValidationError("to_bond needs a dated BondQuote")
        if valuation_date is None:
            raise ValidationError("to_bond needs the run's valuation_date")
        from quantark.asset.bond.product.couponbond.fixed_bond import FixedBond  # local: see to_swap
        from quantark.util.calendar import create_calendar

        assert self.issue_date is not None and self.maturity_date is not None
        if self.maturity_date <= valuation_date:
            raise ValidationError("BondQuote.maturity_date must be after the valuation date")
        return FixedBond(
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            denominator=self.FACE,
            coupon_rate=self.coupon_rate,
            payment_frequency=self.payment_frequency,
            day_count_convention=self.day_count,
            calendar=create_calendar(self.calendar_type),
            business_day_convention=self.business_day_convention,
        )

    def pillar_time(self, valuation_date: Optional[datetime]) -> float:
        _run_clock_matches(self, valuation_date)
        if not self.is_dated:
            return self.periods() / self.payment_frequency.value
        assert valuation_date is not None
        flows = self.to_bond(valuation_date).get_cashflows(valuation_date)
        if not flows:
            raise ValidationError("the bond has no cash flows after the valuation date")
        return _after(valuation_date, max(cf.payment_date for cf in flows), "bond last payment")


@dataclass(frozen=True)
class FuturesQuote:
    """Index futures price ``F`` at one expiry: carry node ``B = ln(F / S)``."""

    price: float
    tenor_years: Optional[float] = None
    expiry_date: Optional[datetime] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        price = _finite(self.price, "FuturesQuote.price")
        if price <= 0.0:
            raise ValidationError(f"price must be positive, got {price}")
        _check_clock("FuturesQuote", self.tenor_years, {"expiry_date": self.expiry_date})

    @property
    def is_dated(self) -> bool:
        return self.expiry_date is not None

    def pillar_time(self, valuation_date: Optional[datetime]) -> float:
        _run_clock_matches(self, valuation_date)
        if not self.is_dated:
            return float(self.tenor_years)
        assert valuation_date is not None and self.expiry_date is not None
        return _after(valuation_date, self.expiry_date, "futures expiry")

    @classmethod
    def from_index_futures_curve(cls, curve, dated: bool = False) -> List["FuturesQuote"]:
        """One quote per contract of an ``IndexFuturesCurve``.

        Numeric (default): ``tenor_years`` from each quote's ``maturity``. Dated:
        ``expiry_date`` from each quote, which every contract must carry.
        """
        out: List[FuturesQuote] = []
        for q in curve.quotes:
            if dated:
                if q.expiry_date is None:
                    raise ValidationError(f"futures contract {q.contract!r} has no expiry_date for a dated run")
                out.append(cls(price=float(q.price), expiry_date=q.expiry_date, label=q.contract))
            else:
                out.append(cls(price=float(q.price), tenor_years=float(q.maturity), label=q.contract))
        return out


@dataclass(frozen=True)
class OptionPairQuote:
    """Call/put pair at one strike and expiry: ``F = K + (C − P)/DF(T)`` by put-call parity."""

    strike: float
    call_price: float
    put_price: float
    tenor_years: Optional[float] = None
    expiry_date: Optional[datetime] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        strike = _finite(self.strike, "OptionPairQuote.strike")
        if strike <= 0.0:
            raise ValidationError(f"strike must be positive, got {strike}")
        for name in ("call_price", "put_price"):
            v = _finite(getattr(self, name), f"OptionPairQuote.{name}")
            if v < 0.0:
                raise ValidationError(f"{name} must be non-negative, got {v}")
        _check_clock("OptionPairQuote", self.tenor_years, {"expiry_date": self.expiry_date})

    @property
    def is_dated(self) -> bool:
        return self.expiry_date is not None

    def pillar_time(self, valuation_date: Optional[datetime]) -> float:
        _run_clock_matches(self, valuation_date)
        if not self.is_dated:
            return float(self.tenor_years)
        assert valuation_date is not None and self.expiry_date is not None
        return _after(valuation_date, self.expiry_date, "option expiry")


RateQuote = Union[DepositQuote, SwapQuote, BondQuote]
CarryQuote = Union[FuturesQuote, OptionPairQuote]
Quote = Union[RateQuote, CarryQuote]


def check_run_clock(quotes: Sequence[Quote], valuation_date: Optional[datetime]) -> None:
    """Every quote must be on the run's clock; mixing raises ``ValidationError``."""
    if not quotes:
        raise ValidationError("at least one quote is required")
    for q in quotes:
        _run_clock_matches(q, valuation_date)
