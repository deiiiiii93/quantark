"""Bond and rate DV01-style bumps must PARALLEL-shift the rate curve, not
replace it with ``FlatRateCurve(r(1y) +/- bump)``.

Same bug class as the equity scalar rho (test_rho_term_structure_bump.py):
on a term curve the flat replacement differs from the base by
r(1y) - r(t) + bump at every t, so the finite difference measures a curve
reshaping. ``RateCurve.parallel_shifted`` is the one primitive every site
uses; a flat curve stays flat (legacy floats, bitwise).
"""
import math
from datetime import datetime

import pytest

from quantark.param.rrf import FlatRateCurve
from quantark.param.rrf.rate_curve import LinearRateCurve, ParallelShiftRateCurve
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock

BP = 1e-4
VD = datetime(2024, 1, 15)
STEEP = [(0.25, 0.010), (1.0, 0.030), (5.0, 0.050), (10.0, 0.060)]


def _curve():
    return LinearRateCurve(STEEP)


def _env(curve=None, **kw):
    return PricingEnvironment(rate_curve=curve or _curve(), valuation_date=VD, **kw)


def _shifted_env(shift, **kw):
    return _env(_curve().parallel_shifted(shift), **kw)


# --- the primitive ----------------------------------------------------------

def test_parallel_shifted_flat_curve_stays_flat_bitwise():
    c = FlatRateCurve(0.0356)
    up, down = c.parallel_shifted(BP), c.parallel_shifted(-BP)
    assert type(up) is FlatRateCurve and up.rate == 0.0356 + BP
    assert type(down) is FlatRateCurve and down.rate == 0.0356 - BP
    assert c.rate == 0.0356


def test_parallel_shifted_term_curve_preserves_shape():
    c = _curve()
    up = c.parallel_shifted(BP)
    assert isinstance(up, ParallelShiftRateCurve)
    for t in (0.1, 0.25, 0.7, 1.0, 3.0, 5.0, 8.0, 10.0, 12.0):
        assert up.get_rate(t) == pytest.approx(c.get_rate(t) + BP, abs=1e-12)
        assert up.get_discount_factor(t) == pytest.approx(
            c.get_discount_factor(t) * math.exp(-BP * t), rel=1e-14
        )


def test_parallel_shifted_trading_clock_shifts_calendar_inner():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), datetime(2026, 2, 9), datetime(2028, 2, 9))
    wrapped = TradingClockRateCurve(_curve(), m)
    up = wrapped.parallel_shifted(BP)
    assert type(up) is TradingClockRateCurve and up.time_map is m
    for t_td in (0.1, 0.5, 1.0, 1.7):
        t_cal = float(m.to_calendar(t_td))
        assert up.get_discount_factor(t_td) == pytest.approx(
            wrapped.get_discount_factor(t_td) * math.exp(-BP * t_cal), rel=1e-14
        )


# --- rate engines -----------------------------------------------------------

def _swaption():
    from quantark.asset.rate.product.swaption import create_payer_swaption
    from quantark.param.index import SOFR_3M

    return create_payer_swaption(
        exercise_date=datetime(2025, 1, 15), swap_tenor_years=5,
        notional=10_000_000, fixed_rate=0.05, index=SOFR_3M,
    )


def test_swaption_dv01_on_term_curve_is_parallel_shift():
    from quantark.asset.rate.engine.swaption_engine import SwaptionEngine

    s = _swaption()
    up = SwaptionEngine(_shifted_env(BP), vol=0.20).price(s, VD)
    down = SwaptionEngine(_shifted_env(-BP), vol=0.20).price(s, VD)
    assert SwaptionEngine(_env(), vol=0.20).dv01(s) == pytest.approx((down - up) / (2 * BP), rel=1e-12)


def test_cap_dv01_on_term_curve_is_parallel_shift():
    from quantark.asset.rate.engine.cap_floor_engine import CapFloorEngine
    from quantark.asset.rate.product.cap_floor import create_cap
    from quantark.param.index import SOFR_3M

    cap = create_cap(start_date=datetime(2024, 3, 15), end_date=datetime(2026, 3, 15),
                     notional=10_000_000, strike=0.05, index=SOFR_3M)
    up = CapFloorEngine(_shifted_env(BP), vol=0.20).price(cap, VD)
    down = CapFloorEngine(_shifted_env(-BP), vol=0.20).price(cap, VD)
    assert CapFloorEngine(_env(), vol=0.20).dv01(cap) == pytest.approx((down - up) / (2 * BP), rel=1e-12)


def test_fra_dv01_on_term_curve_is_parallel_shift():
    from quantark.asset.rate.engine.fra_engine import FRAEngine
    from quantark.asset.rate.product.fra import create_fra
    from quantark.param.index import SOFR_3M

    fra = create_fra(trade_date=VD, settlement_date=datetime(2024, 4, 15), tenor_months=3,
                     notional=10_000_000, fixed_rate=0.05, index=SOFR_3M)
    up = FRAEngine(_shifted_env(BP)).price(fra, VD)
    down = FRAEngine(_shifted_env(-BP)).price(fra, VD)
    assert FRAEngine(_env()).dv01(fra) == pytest.approx((down - up) / (2 * BP), rel=1e-12)


def _irs():
    from quantark.asset.rate.product.irs import SwapDirection, create_vanilla_irs
    from quantark.param.index import SOFR_3M

    return create_vanilla_irs(
        effective_date=VD, maturity_date=datetime(2029, 1, 15), denominator=10_000_000.0,
        fixed_rate=0.045, index=SOFR_3M, direction=SwapDirection.PAYER,
    )


def test_irs_dv01_on_term_curve_is_parallel_shift():
    from quantark.asset.rate.engine.irs_discount_engine import IRSDiscountEngine

    swap = _irs()
    up = IRSDiscountEngine(_shifted_env(BP)).price(swap, VD)
    down = IRSDiscountEngine(_shifted_env(-BP)).price(swap, VD)
    assert IRSDiscountEngine(_env()).dv01(swap) == pytest.approx((down - up) / (2 * BP), rel=1e-12)


def test_irs_dv01_dual_curve_shifts_both_and_restores_projection():
    from quantark.asset.rate.engine.irs_discount_engine import IRSDiscountEngine

    swap = _irs()
    disc = _curve()
    proj = LinearRateCurve([(t, r + 0.002) for t, r in STEEP])       # 20bp basis
    engine = IRSDiscountEngine(_env(disc), projection_curve=proj)
    up = IRSDiscountEngine(_env(disc.parallel_shifted(BP)),
                           projection_curve=proj.parallel_shifted(BP)).price(swap, VD)
    down = IRSDiscountEngine(_env(disc.parallel_shifted(-BP)),
                             projection_curve=proj.parallel_shifted(-BP)).price(swap, VD)
    assert engine.dv01(swap) == pytest.approx((down - up) / (2 * BP), rel=1e-12)
    assert engine.pricing_env.rate_curve is disc
    assert engine.projection_curve is proj                       # not the discount curve


def test_irs_convexity_on_term_curve_is_parallel_shift():
    from quantark.asset.rate.engine.irs_discount_engine import IRSDiscountEngine

    swap = _irs()
    engine = IRSDiscountEngine(_env())
    base = engine.price(swap, VD)
    up = IRSDiscountEngine(_shifted_env(BP)).price(swap, VD)
    down = IRSDiscountEngine(_shifted_env(-BP)).price(swap, VD)
    expected = (up + down - 2 * base) / (base * BP * BP)
    assert engine.convexity(swap) == pytest.approx(expected, rel=1e-9)


def test_irs_key_rate_durations_on_term_curve_are_the_parallel_value():
    """The documented approximation bumps ALL rates; on a term curve that is
    the parallel shift for every tenor, not a flattening at r(tenor)."""
    from quantark.asset.rate.engine.irs_discount_engine import IRSDiscountEngine

    swap = _irs()
    engine = IRSDiscountEngine(_env())
    npv = abs(engine.price(swap, VD))
    up = IRSDiscountEngine(_shifted_env(BP)).price(swap, VD)
    down = IRSDiscountEngine(_shifted_env(-BP)).price(swap, VD)
    expected = (down - up) / (2 * BP * npv)
    krds = engine.key_rate_durations(swap, key_tenors=[1.0, 5.0])
    assert krds[1.0] == pytest.approx(expected, rel=1e-12)
    assert krds[5.0] == pytest.approx(expected, rel=1e-12)


# --- bond option greeks -----------------------------------------------------

def _bond_option():
    from quantark.asset.bond.product.couponbond.fixed_bond import create_simple_fixed_bond
    from quantark.asset.bond.product.option.euro_short_term_bond_option import (
        EuroShortTermBondOption,
    )
    from quantark.util.enum import OptionType, PaymentFrequency

    bond = create_simple_fixed_bond(
        issue_date=datetime(2023, 1, 1), maturity_date=datetime(2028, 1, 1),
        denominator=1000.0, coupon_rate=0.05, payment_frequency=PaymentFrequency.SEMI_ANNUAL,
    )
    return EuroShortTermBondOption(underlying=bond, strike=1000.0,
                                   expiry_date=datetime(2025, 1, 1), option_type=OptionType.CALL)


def _bond_env(shift=0.0):
    from quantark.param.vol import FlatVolSurface

    return PricingEnvironment(rate_curve=_curve().parallel_shifted(shift) if shift else _curve(),
                              valuation_date=datetime(2024, 1, 1),
                              vol_surface=FlatVolSurface(volatility=0.10))


def test_bond_option_numerical_dv01_and_rho_on_term_curve_are_parallel_shifts():
    from quantark.asset.bond.engine.analytical.black_engine import BlackBondOptionEngine
    from quantark.asset.bond.riskmeasures.bond_greeks_calculator import BondGreeksCalculator

    opt, vd = _bond_option(), datetime(2024, 1, 1)
    base = BlackBondOptionEngine(_bond_env()).price(opt, 0.10, vd)
    up_1bp = BlackBondOptionEngine(_bond_env(BP)).price(opt, 0.10, vd)
    up_1pct = BlackBondOptionEngine(_bond_env(0.01)).price(opt, 0.10, vd)
    g = BondGreeksCalculator().calculate_numerical_greeks(opt, _bond_env(), volatility=0.10)
    assert g["dv01"] == pytest.approx(base - up_1bp, rel=1e-12)
    assert g["rho"] == pytest.approx(up_1pct - base, rel=1e-12)
    sens = BondGreeksCalculator().calculate_bond_sensitivities(opt, _bond_env(), volatility=0.10)
    assert sens["option_dv01"] == pytest.approx(base - up_1bp, rel=1e-12)


def test_bond_option_numerical_delta_gamma_on_term_curve_use_parallel_shifts():
    from quantark.asset.bond.engine.analytical.black_engine import BlackBondOptionEngine
    from quantark.asset.bond.riskmeasures.bond_greeks_calculator import BondGreeksCalculator

    opt, vd = _bond_option(), datetime(2024, 1, 1)
    bump = 0.01 * _curve().get_rate(1.0)          # calculator: bump_size * r(1y)
    base = BlackBondOptionEngine(_bond_env()).price(opt, 0.10, vd)
    r_up = BlackBondOptionEngine(_bond_env(bump)).price_with_details(opt, 0.10, vd)
    r_dn = BlackBondOptionEngine(_bond_env(-bump)).price_with_details(opt, 0.10, vd)
    dF = r_up.forward_bond_price - r_dn.forward_bond_price
    g = BondGreeksCalculator().calculate_numerical_greeks(opt, _bond_env(), volatility=0.10)
    assert g["delta"] == pytest.approx((r_up.price - r_dn.price) / dF, rel=1e-12)
    assert g["gamma"] == pytest.approx((r_up.price - 2 * base + r_dn.price) / ((dF / 2) ** 2), rel=1e-9)


# --- bond forward / futures / FRN -------------------------------------------

def _bond(issue, maturity, coupon):
    from quantark.asset.bond.product.couponbond.fixed_bond import create_simple_fixed_bond
    from quantark.util.enum import PaymentFrequency

    return create_simple_fixed_bond(issue_date=issue, maturity_date=maturity, denominator=100.0,
                                    coupon_rate=coupon, payment_frequency=PaymentFrequency.SEMI_ANNUAL)


def test_bond_forward_dv01_on_term_curve_is_parallel_shift():
    from quantark.asset.bond.engine.analytical.bond_forward_engine import BondForwardEngine
    from quantark.asset.bond.product.forward.bond_forward import BondForward

    fwd = BondForward(underlying=_bond(datetime(2023, 1, 15), datetime(2033, 1, 15), 0.05),
                      delivery_date=datetime(2024, 6, 15), repo_rate=0.045, is_long=True,
                      contract_size=100_000)
    base = BondForwardEngine(_env()).price(fwd, VD).forward_dirty_price
    up = BondForwardEngine(_shifted_env(BP)).price(fwd, VD).forward_dirty_price
    greeks = BondForwardEngine(_env()).calculate_greeks(fwd, VD)
    assert greeks["dv01"] == pytest.approx(base - up, rel=1e-12)


def test_bond_futures_dv01_on_term_curve_is_parallel_shift():
    from quantark.asset.bond.engine.analytical.bond_futures_engine import BondFuturesEngine
    from quantark.asset.bond.product.futures.bond_futures import BondFutures, DeliverableBond

    vd = datetime(2024, 11, 15)
    fut = BondFutures(
        delivery_date=datetime(2025, 3, 15),
        deliverable_basket=[
            DeliverableBond(_bond(datetime(2021, 6, 15), datetime(2031, 6, 15), 0.05)),
            DeliverableBond(_bond(datetime(2022, 9, 15), datetime(2032, 9, 15), 0.035)),
        ],
        futures_price=112.50, contract_size=100_000,
    )
    env = PricingEnvironment(rate_curve=_curve(), valuation_date=vd)
    env_up = PricingEnvironment(rate_curve=_curve().parallel_shifted(BP), valuation_date=vd)
    base = BondFuturesEngine(env, repo_rate=0.045).price(fut, vd)
    up = BondFuturesEngine(env_up, repo_rate=0.045).price(fut, vd, calculate_greeks=False)
    expected = (base.theoretical_futures_price - up.theoretical_futures_price) \
        / fut.get_conversion_factor(base.ctd_bond_index)
    assert base.dv01 == pytest.approx(expected, rel=1e-12)


def test_frn_effective_duration_on_term_curve_is_parallel_shift():
    from quantark.asset.bond.engine.discount.frn_engine import FRNDiscountEngine
    from quantark.asset.bond.product.couponbond.frn import create_simple_frn
    from quantark.param.index import SOFR_3M
    from quantark.util.enum import PaymentFrequency

    vd = datetime(2024, 1, 1)
    frn = create_simple_frn(issue_date=vd, maturity_date=datetime(2029, 1, 1), denominator=1_000_000.0,
                            index=SOFR_3M, spread=0.0050, payment_frequency=PaymentFrequency.QUARTERLY)
    env = PricingEnvironment(rate_curve=_curve(), valuation_date=vd)
    p0 = FRNDiscountEngine(env).price(frn, vd, vd)
    p_up = FRNDiscountEngine(PricingEnvironment(rate_curve=_curve().parallel_shifted(BP), valuation_date=vd)).price(frn, vd, vd)
    p_dn = FRNDiscountEngine(PricingEnvironment(rate_curve=_curve().parallel_shifted(-BP), valuation_date=vd)).price(frn, vd, vd)
    assert FRNDiscountEngine(env).effective_duration(frn, vd, vd) == pytest.approx(
        -(p_up - p_dn) / (2 * BP * p0), rel=1e-12
    )
