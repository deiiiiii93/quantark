"""Cash-flow strips: one quote → the flows the bootstrapper reprices.

Spec: ``.ship/2026-11-05-curve-construction/spec.md`` §Contracts ``strip.py``.

A strip is the clock-free image of an instrument on the ACT/365F axis. Its
``residual(discount, projection)`` is zero when the curves reprice the quote:

* deposit: ``P(t_start) − P(t_end)·(1 + r·τ)`` on the PROJECTION curve (so a
  deposit can seed a projection curve; in a discount bootstrap both arguments
  are the curve under construction);
* swap (unit notional): ``Σ fixed_k·D(t_pay_k) − Σ (L_k + s)·dcf_k·D(t_pay_k)`` with
  ``L_k = simple_forward_rate(projection, t_start_k, t_end_k, dcf_k)`` — the same
  helper the rate engines use;
* bond (per 100 face): ``Σ cf_k·D(t_pay_k) − dirty``, ``dirty = clean + accrued``.

Numeric strips are the idealised twins (regular periods, ``dcf = 1/frequency``,
payment at accrual end, accrued 0). Dated strips are read off the products the
quote builds (``to_swap`` / ``to_bond``), so they coincide with the engines by
construction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from quantark.param.curve.quotes import BondQuote, DepositQuote, SwapQuote
from quantark.param.rrf.projection import act365f, simple_forward_rate
from quantark.param.rrf.rate_curve import RateCurve
from quantark.util.calendar import calculate_day_count_fraction
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class FixedFlow:
    """A known amount paid at ``t_pay``."""

    t_pay: float
    amount: float


@dataclass(frozen=True)
class FloatFlow:
    """A floating coupon ``notional·(L + spread)·dcf`` paid at ``t_pay``, ``L`` over ``[t_start, t_end]``."""

    t_pay: float
    t_start: float
    t_end: float
    dcf: float
    notional: float
    spread: float

    def rate(self, projection: RateCurve) -> float:
        """Projected simple rate on ``projection``."""
        return simple_forward_rate(projection, self.t_start, self.t_end, self.dcf)

    def amount(self, projection: RateCurve) -> float:
        return self.notional * (self.rate(projection) + self.spread) * self.dcf


@dataclass(frozen=True)
class InstrumentStrip(ABC):
    """One quote's flows on the axis; ``pillar_time`` is its last payment time."""

    quote: object
    pillar_time: float

    @property
    @abstractmethod
    def kind(self) -> str: ...

    @abstractmethod
    def residual(self, discount: RateCurve, projection: RateCurve) -> float:
        """Model value minus quoted value; zero when the curves reprice the quote."""


@dataclass(frozen=True)
class DepositStrip(InstrumentStrip):
    t_start: float
    t_end: float
    tau: float
    rate: float

    @property
    def kind(self) -> str:
        return "deposit"

    def residual(self, discount: RateCurve, projection: RateCurve) -> float:
        p_start = 1.0 if self.t_start == 0.0 else float(projection.get_discount_factor(self.t_start))
        p_end = float(projection.get_discount_factor(self.t_end))
        return p_start - p_end * (1.0 + self.rate * self.tau)


@dataclass(frozen=True)
class SwapStrip(InstrumentStrip):
    fixed_flows: Tuple[FixedFlow, ...]
    float_flows: Tuple[FloatFlow, ...]

    @property
    def kind(self) -> str:
        return "swap"

    def fixed_pv(self, discount: RateCurve) -> float:
        return sum(f.amount * float(discount.get_discount_factor(f.t_pay)) for f in self.fixed_flows)

    def float_pv(self, discount: RateCurve, projection: RateCurve) -> float:
        return sum(
            f.amount(projection) * float(discount.get_discount_factor(f.t_pay)) for f in self.float_flows
        )

    def residual(self, discount: RateCurve, projection: RateCurve) -> float:
        return self.fixed_pv(discount) - self.float_pv(discount, projection)


@dataclass(frozen=True)
class BondStrip(InstrumentStrip):
    flows: Tuple[FixedFlow, ...]
    dirty_price: float

    @property
    def kind(self) -> str:
        return "bond"

    def residual(self, discount: RateCurve, projection: RateCurve) -> float:
        pv = sum(f.amount * float(discount.get_discount_factor(f.t_pay)) for f in self.flows)
        return pv - self.dirty_price


# --------------------------------------------------------------------------- builders


def _deposit_strip(q: DepositQuote, valuation_date: Optional[datetime]) -> DepositStrip:
    if not q.is_dated:
        tenor = float(q.tenor_years)
        return DepositStrip(quote=q, pillar_time=tenor, t_start=0.0, t_end=tenor, tau=tenor, rate=float(q.rate))
    assert valuation_date is not None and q.maturity_date is not None
    start = q.start_date or valuation_date
    t_start = act365f(valuation_date, start)
    if t_start < 0.0:
        raise ValidationError(
            f"deposit start {start:%Y-%m-%d} is before the valuation date {valuation_date:%Y-%m-%d}"
        )
    t_end = q.pillar_time(valuation_date)
    if t_end <= t_start:
        raise ValidationError("deposit maturity must be after its start")
    tau = float(calculate_day_count_fraction(start, q.maturity_date, q.day_count))
    if 1.0 + q.rate * tau <= 0.0:
        raise ValidationError(f"1 + rate·τ must be positive, got rate={q.rate}, τ={tau}")
    return DepositStrip(quote=q, pillar_time=t_end, t_start=t_start, t_end=t_end, tau=tau, rate=float(q.rate))


def _swap_strip(q: SwapQuote, valuation_date: Optional[datetime]) -> SwapStrip:
    if not q.is_dated:
        n = q.periods()
        f = q.payment_frequency.value
        dcf = 1.0 / f
        fixed = tuple(FixedFlow(t_pay=k / f, amount=float(q.fixed_rate) * dcf) for k in range(1, n + 1))
        floating = tuple(
            FloatFlow(t_pay=k / f, t_start=(k - 1) / f, t_end=k / f, dcf=dcf, notional=1.0, spread=0.0)
            for k in range(1, n + 1)
        )
        return SwapStrip(quote=q, pillar_time=n / f, fixed_flows=fixed, float_flows=floating)
    assert valuation_date is not None
    swap = q.to_swap(valuation_date)
    fixed = tuple(
        FixedFlow(t_pay=act365f(valuation_date, cf.payment_date), amount=float(cf.amount))
        for cf in swap.fixed_leg.get_cashflows(valuation_date)
    )
    floating = tuple(
        FloatFlow(
            t_pay=act365f(valuation_date, cf.payment_date),
            t_start=act365f(valuation_date, cf.accrual_start_date),
            t_end=act365f(valuation_date, cf.accrual_end_date),
            dcf=float(cf.day_count_fraction),
            notional=float(cf.notional),
            spread=float(cf.spread),
        )
        for cf in swap.floating_leg.get_floating_cashflows(valuation_date)
    )
    if not fixed or not floating:
        raise ValidationError("the swap has no cash flows after the valuation date")
    pillar = max([f.t_pay for f in fixed] + [f.t_pay for f in floating])
    return SwapStrip(quote=q, pillar_time=pillar, fixed_flows=fixed, float_flows=floating)


def _bond_strip(q: BondQuote, valuation_date: Optional[datetime]) -> BondStrip:
    face = q.FACE
    if not q.is_dated:
        n = q.periods()
        f = q.payment_frequency.value
        coupon = face * float(q.coupon_rate) / f
        flows = tuple(
            FixedFlow(t_pay=k / f, amount=coupon + (face if k == n else 0.0)) for k in range(1, n + 1)
        )
        return BondStrip(quote=q, pillar_time=n / f, flows=flows, dirty_price=float(q.clean_price))
    assert valuation_date is not None
    bond = q.to_bond(valuation_date)
    cfs = bond.get_cashflows(valuation_date)
    if not cfs:
        raise ValidationError("the bond has no cash flows after the valuation date")
    flows = tuple(FixedFlow(t_pay=act365f(valuation_date, cf.payment_date), amount=float(cf.amount)) for cf in cfs)
    dirty = float(q.clean_price) + float(bond.calculate_accrued_interest(valuation_date))
    return BondStrip(quote=q, pillar_time=max(f.t_pay for f in flows), flows=flows, dirty_price=dirty)


def build_strip(quote, valuation_date: Optional[datetime]) -> InstrumentStrip:
    """The strip of a rate quote on the run's clock (``None`` = numeric)."""
    if quote.is_dated != (valuation_date is not None):
        clock = "dated" if quote.is_dated else "numeric"
        run = "numeric (valuation_date=None)" if valuation_date is None else f"dated ({valuation_date:%Y-%m-%d})"
        raise ValidationError(f"{clock} quote {quote!r} in a {run} run: one clock per run")
    if isinstance(quote, DepositQuote):
        return _deposit_strip(quote, valuation_date)
    if isinstance(quote, SwapQuote):
        return _swap_strip(quote, valuation_date)
    if isinstance(quote, BondQuote):
        return _bond_strip(quote, valuation_date)
    raise ValidationError(f"not a rate instrument quote: {quote!r}")
