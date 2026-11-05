"""T7 — dated-mode engine consistency and environment integration (AC10, AC13).

A curve bootstrapped from dated quotes must reprice the very products the quotes
build, through the repo's own engines: ``BondDiscountEngine`` (dirty price =
clean + accrued), ``IRSDiscountEngine`` (par swaps at zero NPV, single- and
dual-curve) and ``FRAEngine`` (the simple forward on the projection curve).
A ``ZeroCurve`` + carry view that encode a flat ``r``/``q`` must price a vanilla
option through ``BlackScholesEngine`` like ``FlatRateCurve``/``ContinuousDividendYield``.

Dates are chosen away from month-end weekends (a MODIFIED_FOLLOWING roll-back
before an unadjusted accrual end makes ``CashFlow.__post_init__`` raise) and
every maturity is ≤ 2030 because the holiday files cover 2020–2030.
"""
from __future__ import annotations

import math
from datetime import datetime

import pytest

from quantark.asset.bond.engine.discount import BondDiscountEngine
from quantark.asset.equity.engine import BlackScholesEngine
from quantark.asset.equity.product import EuropeanVanillaOption
from quantark.asset.rate.engine import FRAEngine, IRSDiscountEngine
from quantark.asset.rate.product.fra import ForwardRateAgreement
from quantark.param import (
    BondQuote,
    BootstrapConfig,
    ContinuousDividendYield,
    CurveInterpolation,
    DepositQuote,
    FlatRateCurve,
    FlatVolSurface,
    FuturesQuote,
    SpotQuote,
    SwapQuote,
    ZeroCurve,
    bootstrap_discount_curve,
    bootstrap_projection_curve,
    build_carry_curve,
    reprice_quotes,
    simple_forward_rate,
)
from quantark.param.index import REPO_7D, SHIBOR_3M
from quantark.param.rrf.projection import act365f
from quantark.priceenv.pricing_environment import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import OptionType, PaymentFrequency
from quantark.util.exceptions import ValidationError

VD = datetime(2026, 11, 5)  # a Thursday

DEPOSITS = [
    DepositQuote(rate=0.0180, maturity_date=datetime(2026, 12, 7), label="1M"),
    DepositQuote(rate=0.0190, maturity_date=datetime(2027, 2, 5), label="3M"),
    DepositQuote(rate=0.0205, maturity_date=datetime(2027, 5, 5), label="6M"),
]
SWAPS = [
    SwapQuote(fixed_rate=0.0220, index=SHIBOR_3M, maturity_date=datetime(2027, 11, 5), label="1Y"),
    SwapQuote(fixed_rate=0.0262, index=SHIBOR_3M, maturity_date=datetime(2029, 11, 5), label="3Y"),
    SwapQuote(fixed_rate=0.0285, index=SHIBOR_3M, maturity_date=datetime(2030, 11, 5), label="4Y"),
]
BONDS = [
    BondQuote(coupon_rate=0.025, clean_price=100.4, payment_frequency=PaymentFrequency.SEMI_ANNUAL,
              issue_date=datetime(2024, 3, 20), maturity_date=datetime(2028, 3, 20), label="2Y-bond"),
    BondQuote(coupon_rate=0.030, clean_price=100.9, payment_frequency=PaymentFrequency.SEMI_ANNUAL,
              issue_date=datetime(2025, 6, 10), maturity_date=datetime(2030, 6, 10), label="5Y-bond"),
]
DATED_QUOTES = DEPOSITS + SWAPS + BONDS

SCHEMES = list(CurveInterpolation)


def _rate_quotes():
    return DEPOSITS + SWAPS


@pytest.mark.parametrize("scheme", SCHEMES)
def test_dated_pillars_are_act365f_times(scheme):
    curve = bootstrap_discount_curve(DATED_QUOTES, BootstrapConfig(interpolation=scheme), valuation_date=VD)
    expected = sorted(q.pillar_time(VD) for q in DATED_QUOTES)
    assert list(curve.tenors) == expected
    for q in DEPOSITS + SWAPS + BONDS:
        d = q.maturity_date
        assert q.pillar_time(VD) == act365f(VD, d) == (d - VD).days / 365.0
    assert max(abs(r) for r in reprice_quotes(DATED_QUOTES, curve, valuation_date=VD)) <= 1e-12
    # AC11 end to end: a bumped dated curve (pillar-shifted ZeroCurve or, for
    # MONOTONE_CONVEX, the anchored wrapper) still carries the clock and is
    # refused by a numeric run
    bumped = curve.parallel_shifted(1e-4)
    assert bumped.valuation_date == VD
    assert max(abs(r) for r in reprice_quotes(DATED_QUOTES, bumped, valuation_date=VD)) > 1e-6
    with pytest.raises(ValidationError):
        reprice_quotes(DATED_QUOTES, bumped, valuation_date=None)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_ac10a_bond_engine_reprices_clean_plus_accrued(scheme):
    discount = bootstrap_discount_curve(DATED_QUOTES, BootstrapConfig(interpolation=scheme), valuation_date=VD)
    engine = BondDiscountEngine(PricingEnvironment(rate_curve=discount, valuation_date=VD))
    for q in BONDS:
        bond = q.to_bond(VD)
        accrued = bond.calculate_accrued_interest(VD)
        assert accrued > 0.0  # the valuation date sits inside a coupon period
        assert engine.price(bond, VD) == pytest.approx(q.clean_price + accrued, abs=1e-10)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_ac10b_single_curve_swaps_have_zero_npv(scheme):
    discount = bootstrap_discount_curve(_rate_quotes(), BootstrapConfig(interpolation=scheme), valuation_date=VD)
    env = PricingEnvironment(rate_curve=discount, valuation_date=VD)
    engine = IRSDiscountEngine(env, projection_curve=discount)
    for q in SWAPS:
        swap = q.to_swap(VD)
        assert abs(engine.npv(swap, VD)) <= 1e-9 * 1.0  # notional 1.0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_ac10b_dual_curve_swaps_have_zero_npv(scheme):
    cfg = BootstrapConfig(interpolation=scheme)
    discount = bootstrap_discount_curve(_rate_quotes(), cfg, valuation_date=VD)
    # a different discount curve: the same shape shifted down 50 bp
    other_discount = discount.parallel_shifted(-0.005)
    projection = bootstrap_projection_curve(_rate_quotes(), other_discount, cfg, valuation_date=VD)
    assert max(abs(a - b) for a, b in zip(projection.rates, discount.rates)) > 1e-6
    env = PricingEnvironment(rate_curve=other_discount, valuation_date=VD)
    engine = IRSDiscountEngine(env, projection_curve=projection)
    for q in SWAPS:
        swap = q.to_swap(VD)
        assert abs(engine.npv(swap, VD)) <= 1e-9 * 1.0
    # and the engine really consults the projection curve: single-curve on the
    # shifted discount alone is far from par
    assert abs(IRSDiscountEngine(env).npv(SWAPS[-1].to_swap(VD), VD)) > 1e-4


def test_ac10c_fra_forward_rate_is_the_simple_forward_on_the_projection_curve():
    cfg = BootstrapConfig(interpolation=CurveInterpolation.LOG_LINEAR_DF)
    discount = bootstrap_discount_curve(_rate_quotes(), cfg, valuation_date=VD)
    other_discount = discount.parallel_shifted(-0.005)
    projection = bootstrap_projection_curve(_rate_quotes(), other_discount, cfg, valuation_date=VD)
    # accrual window straddling the 6M (2027-05-05) and 1Y (2027-11-05) pillars
    fra = ForwardRateAgreement(
        notional=10_000_000.0,
        fixed_rate=0.02,
        accrual_start=datetime(2027, 3, 5),
        accrual_end=datetime(2027, 12, 6),
        index=SHIBOR_3M,
        day_count_convention=DayCountConvention.ACT_360,
    )
    env = PricingEnvironment(rate_curve=other_discount, valuation_date=VD)
    engine = FRAEngine(env, projection_curve=projection)
    t1, t2 = act365f(VD, fra.accrual_start), act365f(VD, fra.accrual_end)
    assert t1 < 0.5 < t2 and t1 < 1.0 < t2
    expected = simple_forward_rate(projection, t1, t2, fra.day_count_fraction())
    assert engine.forward_rate(fra, VD) == pytest.approx(expected, abs=1e-14)
    # dual-curve: the discount curve alone gives a different forward
    assert abs(expected - simple_forward_rate(other_discount, t1, t2, fra.day_count_fraction())) > 1e-5


def test_ac13_flat_zero_curve_and_carry_view_match_flat_inputs():
    r, q, s0, sigma, T = 0.03, 0.02, 100.0, 0.20, 1.5
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0]
    # simple deposit rates that encode a flat continuously-compounded r
    deposits = [DepositQuote(rate=(math.exp(r * t) - 1.0) / t, tenor_years=t) for t in tenors]
    zero = bootstrap_discount_curve(deposits, BootstrapConfig(interpolation=CurveInterpolation.LINEAR_ZERO))
    assert isinstance(zero, ZeroCurve)
    futures = [FuturesQuote(price=s0 * math.exp((r - q) * t), tenor_years=t) for t in tenors]
    carry = build_carry_curve(s0, futures, zero)
    view = carry.to_dividend_yield(zero)

    option = EuropeanVanillaOption(strike=105.0, option_type=OptionType.CALL, maturity=T)
    common = dict(valuation_date=VD, spot_quote=SpotQuote(spot=s0), vol_surface=FlatVolSurface(volatility=sigma))
    env_curves = PricingEnvironment(rate_curve=zero, div_yield=view, **common)
    env_flat = PricingEnvironment(rate_curve=FlatRateCurve(r), div_yield=ContinuousDividendYield(q), **common)
    engine = BlackScholesEngine()
    assert engine.price(option, env_curves) == pytest.approx(engine.price(option, env_flat), abs=1e-10)
    assert zero.get_rate(T) == pytest.approx(r, abs=1e-14)
    assert view.get_yield(T) == pytest.approx(q, abs=1e-14)
