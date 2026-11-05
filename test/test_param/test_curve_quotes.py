"""Instrument quotes, the clock rule and cash-flow strips (spec 2026-11-05, T5).

Covers AC11 (quote level), the quote-level ValidationErrors of AC9 / AC14, and
the strip ↔ product equivalences the bootstrapper relies on.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from quantark.asset.equity.market import IndexFuturesCurve, IndexFuturesQuote
from quantark.asset.rate import IRSDiscountEngine
from quantark.param import (
    BondQuote,
    CurveInterpolation,
    DepositQuote,
    FlatRateCurve,
    FuturesQuote,
    OptionPairQuote,
    SwapQuote,
    ZeroCurve,
)
from quantark.param.curve.strip import (
    BondStrip,
    DepositStrip,
    FixedFlow,
    FloatFlow,
    InstrumentStrip,
    SwapStrip,
    build_strip,
)
from quantark.param.index import REPO_7D, SHIBOR_3M
from quantark.param.rrf.projection import act365f, simple_forward_rate
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import BusinessDayConvention, CalendarType, DayCountConvention
from quantark.util.enum import PaymentFrequency
from quantark.util.exceptions import ValidationError

VD = datetime(2026, 11, 5)  # Thursday
PILLARS = [(0.25, 0.030), (0.5, 0.031), (1.0, 0.033), (2.0, 0.0345), (5.0, 0.036), (10.0, 0.035)]
TERM = ZeroCurve(PILLARS, interpolation=CurveInterpolation.LOG_CUBIC_DF)
FLAT = FlatRateCurve(0.03)


# --------------------------------------------------------------------------- clock rule on quotes (AC11)


def test_exactly_one_clock_per_quote():
    with pytest.raises(ValidationError):
        DepositQuote(rate=0.02)
    with pytest.raises(ValidationError):
        DepositQuote(rate=0.02, tenor_years=0.25, maturity_date=datetime(2027, 2, 5))
    with pytest.raises(ValidationError):
        DepositQuote(rate=0.02, start_date=VD)  # a start date needs a maturity date
    with pytest.raises(ValidationError):
        SwapQuote(fixed_rate=0.03, index=SHIBOR_3M)
    with pytest.raises(ValidationError):
        SwapQuote(fixed_rate=0.03, index=SHIBOR_3M, tenor_years=2.0, maturity_date=datetime(2028, 11, 5))
    with pytest.raises(ValidationError):
        BondQuote(coupon_rate=0.03, clean_price=99.0, payment_frequency=PaymentFrequency.ANNUAL)
    with pytest.raises(ValidationError):
        FuturesQuote(price=4000.0)
    with pytest.raises(ValidationError):
        OptionPairQuote(strike=4000.0, call_price=100.0, put_price=120.0)
    # dated bonds need both schedule anchors
    with pytest.raises(ValidationError):
        BondQuote(coupon_rate=0.03, clean_price=99.0, payment_frequency=PaymentFrequency.ANNUAL,
                  maturity_date=datetime(2031, 11, 5))


def test_is_dated_and_pillar_time():
    dep = DepositQuote(rate=0.02, tenor_years=0.25)
    assert dep.is_dated is False and dep.pillar_time(None) == 0.25
    with pytest.raises(ValidationError):
        dep.pillar_time(VD)  # numeric quote in a dated run
    dated = DepositQuote(rate=0.02, maturity_date=datetime(2027, 2, 5))
    assert dated.is_dated is True
    assert dated.pillar_time(VD) == act365f(VD, datetime(2027, 2, 5)) == 92 / 365.0
    with pytest.raises(ValidationError):
        dated.pillar_time(None)  # dated quote in a numeric run
    fut = FuturesQuote(price=4100.0, expiry_date=datetime(2027, 3, 19))
    assert fut.pillar_time(VD) == act365f(VD, datetime(2027, 3, 19))
    opt = OptionPairQuote(strike=4000.0, call_price=110.0, put_price=95.0, tenor_years=0.5)
    assert opt.pillar_time(None) == 0.5


# --------------------------------------------------------------------------- quote validation (AC9 / AC14)


def test_quote_validation():
    with pytest.raises(ValidationError):
        DepositQuote(rate=-3.0, tenor_years=1.0)  # 1 + r*tau <= 0
    with pytest.raises(ValidationError):
        DepositQuote(rate=float("nan"), tenor_years=1.0)
    with pytest.raises(ValidationError):
        DepositQuote(rate=0.02, tenor_years=0.0)
    with pytest.raises(ValidationError):
        SwapQuote(fixed_rate=0.03, index=SHIBOR_3M, tenor_years=1.1)  # not a whole number of quarters
    SwapQuote(fixed_rate=0.03, index=SHIBOR_3M, tenor_years=2.5)  # 10 quarters: fine
    SwapQuote(fixed_rate=0.03, index=SHIBOR_3M, tenor_years=13 / 12, payment_frequency=PaymentFrequency.MONTHLY)
    with pytest.raises(ValidationError):
        SwapQuote(fixed_rate=0.03, index=SHIBOR_3M, tenor_years=1.1, payment_frequency=PaymentFrequency.MONTHLY)
    with pytest.raises(ValidationError):
        BondQuote(coupon_rate=-0.01, clean_price=99.0, payment_frequency=PaymentFrequency.ANNUAL, tenor_years=2.0)
    with pytest.raises(ValidationError):
        BondQuote(coupon_rate=0.03, clean_price=0.0, payment_frequency=PaymentFrequency.ANNUAL, tenor_years=2.0)
    with pytest.raises(ValidationError):
        BondQuote(coupon_rate=0.03, clean_price=99.0, payment_frequency=PaymentFrequency.SEMI_ANNUAL, tenor_years=2.25)
    with pytest.raises(ValidationError):
        FuturesQuote(price=-1.0, tenor_years=0.5)
    with pytest.raises(ValidationError):
        OptionPairQuote(strike=0.0, call_price=1.0, put_price=1.0, tenor_years=0.5)
    with pytest.raises(ValidationError):
        OptionPairQuote(strike=100.0, call_price=-1.0, put_price=1.0, tenor_years=0.5)
    with pytest.raises(ValidationError):
        SwapQuote(fixed_rate=0.03, index="SHIBOR_3M", tenor_years=1.0)  # type: ignore[arg-type]


def test_futures_quotes_from_index_futures_curve():
    curve = IndexFuturesCurve(
        underlying="CSI1000", spot=6000.0,
        quotes=[
            IndexFuturesQuote("IM2612", 0.12, 5940.0, 200.0, expiry_date=datetime(2026, 12, 18)),
            IndexFuturesQuote("IM2703", 0.37, 5880.0, 200.0, expiry_date=datetime(2027, 3, 19)),
            IndexFuturesQuote("IM2706", 0.62, 5820.0, 200.0),
        ],
    )
    numeric = FuturesQuote.from_index_futures_curve(curve)
    assert [q.tenor_years for q in numeric] == [0.12, 0.37, 0.62]
    assert [q.price for q in numeric] == [5940.0, 5880.0, 5820.0]
    assert [q.label for q in numeric] == ["IM2612", "IM2703", "IM2706"]
    with pytest.raises(ValidationError):
        FuturesQuote.from_index_futures_curve(curve, dated=True)  # IM2706 has no expiry_date


# --------------------------------------------------------------------------- numeric strips


def test_numeric_deposit_strip():
    strip = build_strip(DepositQuote(rate=0.02, tenor_years=0.25), None)
    assert isinstance(strip, DepositStrip) and isinstance(strip, InstrumentStrip)
    assert strip.pillar_time == 0.25 and strip.t_start == 0.0 and strip.t_end == 0.25 and strip.tau == 0.25
    # the curve implied by the quote has zero residual
    z = np.log(1.0 + 0.02 * 0.25) / 0.25
    implied = ZeroCurve([(0.25, z)])
    assert strip.residual(implied, implied) == pytest.approx(0.0, abs=1e-16)
    # deposits read the PROJECTION argument
    assert strip.residual(FLAT, implied) == pytest.approx(0.0, abs=1e-16)
    assert strip.residual(implied, FLAT) != pytest.approx(0.0, abs=1e-6)


def test_numeric_swap_strip_shape_and_par():
    q = SwapQuote(fixed_rate=0.03, index=SHIBOR_3M, tenor_years=2.0)
    strip = build_strip(q, None)
    assert isinstance(strip, SwapStrip)
    assert strip.pillar_time == 2.0
    assert len(strip.fixed_flows) == 8 and len(strip.float_flows) == 8
    for k, (fx, fl) in enumerate(zip(strip.fixed_flows, strip.float_flows), start=1):
        assert isinstance(fx, FixedFlow) and isinstance(fl, FloatFlow)
        assert fx.t_pay == fl.t_pay == k / 4
        assert fx.amount == pytest.approx(0.03 * 0.25)
        assert fl.t_start == (k - 1) / 4 and fl.t_end == k / 4 and fl.dcf == 0.25
        assert fl.notional == 1.0 and fl.spread == 0.0
    # single-curve par rate on a flat curve: (1 - D_n) / sum(dcf * D_k)
    dfs = [FLAT.get_discount_factor(k / 4) for k in range(1, 9)]
    par = (1.0 - dfs[-1]) / sum(0.25 * d for d in dfs)
    par_strip = build_strip(SwapQuote(fixed_rate=par, index=SHIBOR_3M, tenor_years=2.0), None)
    assert par_strip.residual(FLAT, FLAT) == pytest.approx(0.0, abs=1e-15)
    assert strip.residual(FLAT, FLAT) == pytest.approx((0.03 - par) * sum(0.25 * d for d in dfs), rel=1e-12)


def test_numeric_bond_strip_shape_and_par():
    q = BondQuote(coupon_rate=0.04, clean_price=100.0, payment_frequency=PaymentFrequency.SEMI_ANNUAL, tenor_years=2.0)
    strip = build_strip(q, None)
    assert isinstance(strip, BondStrip)
    assert strip.pillar_time == 2.0 and strip.dirty_price == 100.0
    assert [f.t_pay for f in strip.flows] == [0.5, 1.0, 1.5, 2.0]
    assert [f.amount for f in strip.flows] == pytest.approx([2.0, 2.0, 2.0, 102.0])
    pv = sum(f.amount * FLAT.get_discount_factor(f.t_pay) for f in strip.flows)
    assert strip.residual(FLAT, FLAT) == pytest.approx(pv - 100.0, rel=1e-14)
    at_par = BondQuote(coupon_rate=0.04, clean_price=pv, payment_frequency=PaymentFrequency.SEMI_ANNUAL, tenor_years=2.0)
    assert build_strip(at_par, None).residual(FLAT, FLAT) == pytest.approx(0.0, abs=1e-12)


def test_strip_clock_mismatch_raises():
    with pytest.raises(ValidationError):
        build_strip(DepositQuote(rate=0.02, tenor_years=0.25), VD)
    with pytest.raises(ValidationError):
        build_strip(DepositQuote(rate=0.02, maturity_date=datetime(2027, 2, 5)), None)
    with pytest.raises(ValidationError):
        build_strip(FuturesQuote(price=1.0, tenor_years=0.5), None)  # not a rate instrument


# --------------------------------------------------------------------------- dated strips == the products


def test_dated_deposit_strip_uses_day_count_and_validates():
    q = DepositQuote(rate=0.02, maturity_date=datetime(2027, 2, 5), day_count=DayCountConvention.ACT_360)
    strip = build_strip(q, VD)
    assert strip.t_start == 0.0 and strip.t_end == 92 / 365.0 and strip.tau == pytest.approx(92 / 360.0)
    fwd_start = DepositQuote(rate=0.02, start_date=datetime(2026, 12, 7), maturity_date=datetime(2027, 3, 8))
    fs = build_strip(fwd_start, VD)
    assert fs.t_start == act365f(VD, datetime(2026, 12, 7)) and fs.tau == pytest.approx(91 / 365.0)
    with pytest.raises(ValidationError):
        build_strip(DepositQuote(rate=-5.0, maturity_date=datetime(2027, 11, 5)), VD)  # 1 + r*tau <= 0
    with pytest.raises(ValidationError):
        build_strip(DepositQuote(rate=0.02, start_date=datetime(2026, 10, 1), maturity_date=datetime(2027, 2, 5)), VD)


def test_dated_swap_strip_equals_product_cashflows_and_engine():
    q = SwapQuote(fixed_rate=0.031, index=SHIBOR_3M, maturity_date=datetime(2029, 11, 5),
                  calendar_type=CalendarType.NONE, business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING)
    strip = build_strip(q, VD)
    swap = q.to_swap(VD)
    fixed_cfs = swap.fixed_leg.get_cashflows(VD)
    float_cfs = swap.floating_leg.get_floating_cashflows(VD)
    assert len(strip.fixed_flows) == len(fixed_cfs) == 12
    assert len(strip.float_flows) == len(float_cfs) == 12
    for fx, cf in zip(strip.fixed_flows, fixed_cfs):
        assert fx.t_pay == act365f(VD, cf.payment_date) and fx.amount == cf.amount
    for fl, cf in zip(strip.float_flows, float_cfs):
        assert fl.t_pay == act365f(VD, cf.payment_date)
        assert fl.t_start == act365f(VD, cf.accrual_start_date) and fl.t_end == act365f(VD, cf.accrual_end_date)
        assert fl.dcf == cf.day_count_fraction and fl.notional == 1.0 and fl.spread == 0.0
    assert strip.pillar_time == max(act365f(VD, cf.payment_date) for cf in fixed_cfs + list(float_cfs))
    assert q.pillar_time(VD) == strip.pillar_time
    # the engine on the same product: payer NPV = float - fixed = -residual (dual-curve)
    env = PricingEnvironment(rate_curve=FLAT, valuation_date=VD)
    engine = IRSDiscountEngine(env, projection_curve=TERM)
    assert engine.npv(swap) == pytest.approx(-strip.residual(FLAT, TERM), abs=1e-12)
    # float flows project through simple_forward_rate on the projection curve
    for fl in strip.float_flows:
        assert fl.rate(TERM) == simple_forward_rate(TERM, fl.t_start, fl.t_end, fl.dcf)


def test_to_swap_leaves_callers_index_untouched():
    store_before = SHIBOR_3M._fixing_store
    q = SwapQuote(fixed_rate=0.03, index=SHIBOR_3M, maturity_date=datetime(2028, 11, 5))
    swap = q.to_swap(VD)
    assert swap.floating_leg.index is not SHIBOR_3M
    assert swap.floating_leg.index == SHIBOR_3M  # same identity by name
    assert SHIBOR_3M._fixing_store is store_before
    assert swap.fixed_leg.day_count_convention is SHIBOR_3M.day_count_convention
    assert swap.floating_leg.day_count_convention is SHIBOR_3M.day_count_convention
    q2 = SwapQuote(fixed_rate=0.03, index=REPO_7D, maturity_date=datetime(2028, 11, 5),
                   fixed_day_count=DayCountConvention.ACT_360, payment_frequency=PaymentFrequency.ANNUAL)
    swap2 = q2.to_swap(VD)
    assert swap2.fixed_leg.day_count_convention is DayCountConvention.ACT_360
    assert swap2.floating_leg.day_count_convention is REPO_7D.day_count_convention
    assert len(swap2.fixed_leg.get_cashflows(VD)) == 2


def test_dated_bond_strip_equals_product_and_engine():
    # coupons 20 Mar / 20 Sep: the valuation date (5 Nov) sits mid-period, so accrued > 0
    q = BondQuote(coupon_rate=0.035, clean_price=99.2, payment_frequency=PaymentFrequency.SEMI_ANNUAL,
                  issue_date=datetime(2024, 3, 20), maturity_date=datetime(2029, 3, 20),
                  day_count=DayCountConvention.ACT_ACT_ISDA)
    strip = build_strip(q, VD)
    bond = q.to_bond(VD)
    cfs = bond.get_cashflows(VD)
    assert bond.denominator == 100.0
    assert len(strip.flows) == len(cfs) == 5
    for f, cf in zip(strip.flows, cfs):
        assert f.t_pay == act365f(VD, cf.payment_date) and f.amount == cf.amount
    accrued = bond.calculate_accrued_interest(VD)
    assert accrued > 0.0
    assert strip.dirty_price == pytest.approx(99.2 + accrued)
    assert strip.pillar_time == act365f(VD, cfs[-1].payment_date) == q.pillar_time(VD)
    env = PricingEnvironment(rate_curve=TERM, valuation_date=VD)
    price = BondDiscountEngine(env).price(bond)
    assert strip.residual(TERM, TERM) == pytest.approx(price - strip.dirty_price, abs=1e-12)
