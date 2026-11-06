"""The note: per-inception 2Y snowball terms, the coupon solve, its realised lifecycle, daily greeks.

Term sheet (the 2Y carry-curve study's, re-struck at each inception): strike =
KO reference = inception spot; KO 100% on the month anniversaries 3..24 rolled
to the next SSE trading day (the last is maturity); KI 75% discrete on every SSE
trading day in (inception, maturity]; coupon = KO coupon = rebate; principal
excluded; one unit = ``NOTIONAL`` (``contract_multiplier = NOTIONAL/S0``).
The note is booked NUMERIC (ACT/365F times from inception) and aged by
``AutocallableLifecycleTracker`` from ``start_date = inception``.

Sign convention: every value here is HOLDER-side (the engine prices the long
note); ``NoteDay.cash`` is what the note pays the holder that day (> 0 a KO
coupon or maturity rebate, < 0 a knocked-in loss at maturity).  The desk is
short; the books negate.
"""

from __future__ import annotations

import calendar as _calendar
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from quantark.asset.equity.lifecycle.autocallable import AutocallableLifecycleTracker
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.asset.equity.riskmeasures.greeks.bump_envs import resolve_bump_engine
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.curve.carry_curve import CarryCurve
from quantark.priceenv.pricing_environment import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_exp, safe_log

from . import ACT, NOTIONAL, RATE, StudyDataError
from .chain import DayChain
from .tails import build_curve

KO_PCT = 1.00
KI_PCT = 0.75
LOCKOUT_MONTHS = 3
MATURITY_MONTHS = 24
#: |PV| tolerance of the coupon solve, as a fraction of notional.
COUPON_PV_TOL_FRACTION = 1e-6
COUPON_PROBES = (0.05, 0.25)
COUPON_TAIL = "hold_forward"
#: ``calculate_carry_curve_buckets`` settings of the 2Y study.
SPOT_BUMP_REL = 0.01
FORWARD_BUMP_REL = 1e-3
#: One vol point (absolute vol), the vega step and unit.
VOL_BUMP = 0.01

_SSE = create_calendar(CalendarType.CHINA_SSE)


def _is_trading(d: date) -> bool:
    return _SSE.is_business_day(datetime(d.year, d.month, d.day))


def _next_trading(d: date) -> date:
    while not _is_trading(d):
        d += timedelta(days=1)
    return d


def add_months(d: date, months: int) -> date:
    """Same day-of-month ``months`` later, clipped to the month end."""
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, _calendar.monthrange(year, month)[1]))


# ---------------------------------------------------------------------------
# Inceptions and terms
# ---------------------------------------------------------------------------


def inception_dates(chains: Sequence[DayChain], anchors: Mapping[date, float]) -> List[date]:
    """First trading day (chain day) of each calendar quarter, kept when it has an anchor."""
    firsts: Dict[Tuple[int, int], date] = {}
    for dc in chains:
        key = (dc.day.year, (dc.day.month - 1) // 3)
        if key not in firsts or dc.day < firsts[key]:
            firsts[key] = dc.day
    return sorted(d for d in firsts.values() if d in anchors)


@dataclass(frozen=True)
class Terms:
    """The 2Y schedule on the SSE calendar, ACT/365F from ``inception``."""

    inception: date
    maturity_date: date
    maturity_years: float
    ko_dates: Tuple[date, ...]
    ko_times: Tuple[float, ...]
    ki_dates: Tuple[date, ...]
    ki_times: Tuple[float, ...]


def build_terms(inception: date) -> Terms:
    """KO on month anniversaries 3..24 rolled to the next SSE trading day (last = maturity);
    KI on every SSE trading day in (inception, maturity]."""
    if not _is_trading(inception):
        raise ValidationError(f"inception {inception} is not an SSE trading day")

    def yf(d: date) -> float:
        return (d - inception).days / ACT

    maturity = _next_trading(add_months(inception, MATURITY_MONTHS))
    ko_dates = sorted({min(_next_trading(add_months(inception, m)), maturity)
                       for m in range(LOCKOUT_MONTHS, MATURITY_MONTHS + 1)})
    if ko_dates[-1] != maturity:
        raise StudyDataError("last KO observation must be the maturity date")
    ki_dates = []
    d = inception + timedelta(days=1)
    while d <= maturity:
        if _is_trading(d):
            ki_dates.append(d)
        d += timedelta(days=1)
    return Terms(
        inception=inception,
        maturity_date=maturity,
        maturity_years=yf(maturity),
        ko_dates=tuple(ko_dates),
        ko_times=tuple(yf(x) for x in ko_dates),
        ki_dates=tuple(ki_dates),
        ki_times=tuple(yf(x) for x in ki_dates),
    )


def build_note(coupon: float, s0: float, terms: Terms):
    """Standard snowball off ``s0`` (one unit = ``NOTIONAL``), booked numeric."""
    return create_standard_snowball(
        initial_price=s0,
        strike=s0,
        maturity=terms.maturity_years,
        contract_multiplier=NOTIONAL / s0,
        ko_barrier=KO_PCT * s0,
        ko_rate=float(coupon),
        ki_barrier=KI_PCT * s0,
        num_observations=len(terms.ko_times),
        is_reverse=False,
        ko_observation_dates=list(terms.ko_times),
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=list(terms.ki_times),
        rebate_rate=float(coupon),
        include_principal=False,
    )


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


def make_env(dc: DayChain, div_yield, spot: Optional[float] = None) -> PricingEnvironment:
    """Flat vol ``dc.mo.vol``, flat ``r``, ``valuation_date = dc.day``."""
    return PricingEnvironment(
        rate_curve=FlatRateCurve(RATE),
        valuation_date=datetime(dc.day.year, dc.day.month, dc.day.day),
        spot_quote=SpotQuote(spot=dc.spot if spot is None else float(spot)),
        vol_surface=FlatVolSurface(volatility=dc.mo.vol),
        div_yield=div_yield,
    )


def solve_coupon(dc: DayChain, engine) -> float:
    """Coupon with PV = 0 on ``dc`` under ``hold_forward`` (fresh note struck at ``dc.spot``).

    PV is affine in the coupon (no trigger reads it): two probes, one root, one
    re-price check against ``COUPON_PV_TOL_FRACTION·NOTIONAL`` (fails closed).
    """
    terms = build_terms(dc.day)
    env = make_env(dc, build_curve(dc, COUPON_TAIL, None).to_dividend_yield(FlatRateCurve(RATE)))

    def pv_at(c: float) -> float:
        return float(engine.price(build_note(c, dc.spot, terms), env))

    c0, c1 = COUPON_PROBES
    v0, v1 = pv_at(c0), pv_at(c1)
    annuity = (v1 - v0) / (c1 - c0)
    if not annuity > 0.0:
        raise ValidationError(f"{dc.day}: PV not increasing in the coupon (dPV/dc = {annuity})")
    coupon = c0 - v0 / annuity
    pv = pv_at(coupon)
    tol = COUPON_PV_TOL_FRACTION * NOTIONAL
    if abs(pv) > tol:
        raise ValidationError(f"{dc.day}: coupon {coupon:.6f} re-prices to {pv:.4f} > tol {tol:.4f}")
    return float(coupon)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteDay:
    """One close of the note's life (holder side).

    ``cash``: ledger cash determined that day, paid TO the holder (> 0) or by
    the holder (< 0). ``product``: the aged pricing copy, ``None`` once dead.
    """

    alive: bool
    cash: float
    knocked_in: bool
    end_reason: Optional[str]
    product: Any


class NoteLife:
    """A numeric note's realised life on the study's daily closes.

    Wraps ``AutocallableLifecycleTracker(product, quantity=1.0, start_date=inception)``.
    Each close after inception: observe (KO / KI), then settle a maturity due
    today, both on ``tracker.product_for_lifecycle()``, then the pricing copy
    ``tracker.product_for_pricing(day, env)`` (the replay engine's order). The
    inception close is the run's first valuation: no observation,
    ``initial=True``.
    """

    def __init__(self, product, inception: date) -> None:
        self.inception = inception
        self.tracker = AutocallableLifecycleTracker(
            product=product, quantity=1.0, start_date=pd.Timestamp(inception)
        )
        self._last_day: Optional[date] = None

    @property
    def lifecycle(self):
        """The tracker's state; pass it to the engine as ``lifecycle_state``."""
        return self.tracker.lifecycle

    def step(self, dc: DayChain, env: PricingEnvironment) -> NoteDay:
        day = dc.day
        if self._last_day is None:
            if day != self.inception:
                raise ValidationError(f"the first step must be the inception {self.inception}, got {day}")
        elif day <= self._last_day:
            raise ValidationError(f"steps must move forward: {day} after {self._last_day}")
        if not self.lifecycle.alive:
            raise ValidationError(f"{day}: the note ended on {self._last_day}; nothing to step")
        self._last_day = day
        ts = pd.Timestamp(day)
        if day == self.inception:
            product = self.tracker.product_for_pricing(ts, env, initial=True)
            return NoteDay(True, 0.0, bool(self.lifecycle.knocked_in), None, product)

        ledger = self.lifecycle.ledger
        n_before = len(ledger.cashflows)
        self.tracker.observe(ts, self.tracker.product_for_lifecycle(), env, dc.spot)
        self.tracker.settle_maturity_if_due(ts, self.tracker.product_for_lifecycle(), env, dc.spot)
        # quantity 1.0: a ledger amount is the holder's cash for one note (positive = received)
        cash = float(sum(cf.amount for cf in ledger.cashflows[n_before:]))
        state = self.lifecycle
        if not state.alive:
            reason = "ko" if state.knocked_out else ("maturity" if state.matured else None)
            if reason is None:
                raise ValidationError(f"{day}: note dead without a KO or maturity")
            return NoteDay(False, cash, bool(state.knocked_in), reason, None)
        product = self.tracker.product_for_pricing(ts, env)
        return NoteDay(True, cash, bool(state.knocked_in), None, product)


# ---------------------------------------------------------------------------
# Daily greeks (the bucket report's formulas and call order, no rho)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DayGreeks:
    """Holder-side PV and deltas; ``pillar_lots = −δ_i/m_i`` (holder lots; the books negate).

    ``vega``: holder-side CNY per vol point (0.01 absolute vol), central; ``None`` when not computed.
    Diagnostic only (nothing hedges it).
    """

    pv: float
    spot_delta: float
    pinned_delta: float
    pillar_delta: Tuple[float, ...]
    pillar_lots: Tuple[float, ...]
    vega: Optional[float] = None


def daily_greeks(product, env: PricingEnvironment, curve: CarryCurve, engine, lifecycle_state,
                 lot_mults: Sequence[float], *, with_vega: bool = True) -> DayGreeks:
    """PV, spot delta, pinned-forward delta and pillar deltas of ``product`` on ``curve``; then vega.

    The same formulas as ``calculate_carry_curve_buckets`` (spot step 1%, forward
    step 0.1%) and the same price order, so the engine's single-slot cache
    behaves identically: base; spot ±; pinned-forward ± (``B_i − ln(S'/S)``);
    each pillar ``F_i ±``. Rho is not computed. With ``with_vega`` the flat vol
    is then bumped ±``VOL_BUMP`` on the base curve and spot — priced AFTER the
    delta prices, so they are unchanged. ``env.div_yield`` is replaced by
    ``curve``'s view; ``env`` itself is never mutated.
    """
    nodes = curve.nodes
    if len(lot_mults) != len(nodes):
        raise ValidationError(f"lot_mults has {len(lot_mults)} entries for {len(nodes)} nodes")
    rate_curve = env.rate_curve
    spot = float(env.spot_quote.spot)

    def env_for(div_yield, spot_value: float = spot) -> PricingEnvironment:
        e = deepcopy(env)
        e.div_yield = div_yield
        e.spot_quote.spot = spot_value
        return e

    base_div = curve.to_dividend_yield(rate_curve)
    base_env = env_for(base_div)
    bump_engine = resolve_bump_engine(product, base_env, engine)

    def pv(div_yield, spot_value: float = spot) -> float:
        return float(bump_engine.price(product, env_for(div_yield, spot_value), lifecycle_state=lifecycle_state))

    base_pv = float(bump_engine.price(product, base_env, lifecycle_state=lifecycle_state))

    s_up, s_dn = spot * (1.0 + SPOT_BUMP_REL), spot * (1.0 - SPOT_BUMP_REL)
    spot_delta = (pv(base_div, s_up) - pv(base_div, s_dn)) / (s_up - s_dn)

    def pinned(s_new: float):
        shift = float(safe_log(s_new / spot))
        return curve.with_nodes([(t, b - shift) for t, b in nodes]).to_dividend_yield(rate_curve)

    pinned_delta = (pv(pinned(s_up), s_up) - pv(pinned(s_dn), s_dn)) / (s_up - s_dn)

    up_log, dn_log = float(safe_log(1.0 + FORWARD_BUMP_REL)), float(safe_log(1.0 - FORWARD_BUMP_REL))
    deltas: List[float] = []
    lots: List[float] = []
    for i, (t_i, b_i) in enumerate(nodes):
        f_i = spot * float(safe_exp(b_i))

        def moved(new_b: float):
            m = list(nodes)
            m[i] = (t_i, new_b)
            return curve.with_nodes(m).to_dividend_yield(rate_curve)

        delta = (pv(moved(b_i + up_log)) - pv(moved(b_i + dn_log))) / (2.0 * FORWARD_BUMP_REL * f_i)
        deltas.append(delta)
        lots.append(-delta / float(lot_mults[i]))

    vega: Optional[float] = None
    if with_vega:
        surface = env.vol_surface
        if not isinstance(surface, FlatVolSurface):
            raise ValidationError(f"vega bumps a flat vol; got {type(surface).__name__}")

        def at_vol(vol: float) -> float:
            e = env_for(base_div)
            e.vol_surface = FlatVolSurface(volatility=vol)
            return float(bump_engine.price(product, e, lifecycle_state=lifecycle_state))

        sigma = float(surface.volatility)
        # per vol point: (PV(σ+h) − PV(σ−h)) / 2 with h = one point
        vega = (at_vol(sigma + VOL_BUMP) - at_vol(sigma - VOL_BUMP)) / 2.0
    return DayGreeks(base_pv, spot_delta, pinned_delta, tuple(deltas), tuple(lots), vega)
