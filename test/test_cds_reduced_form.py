"""Tests for single-name CDS pricing under a constant hazard rate."""
import math
from datetime import datetime

import pytest

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.product import CDS, ProtectionSide
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.priceenv import CreditPricingEnvironment


def _env(rate=0.0, hazard=0.02):
    return CreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=rate),
        hazard_curve=FlatHazardCurve(hazard_rate=hazard),
    )


def test_protection_leg_closed_form_zero_rate_zero_recovery():
    # PL = N * (1 - R) * (1 - exp(-lambda*T)) when r = 0
    cds = CDS(notional=1_000_000, maturity=5.0, recovery_rate=0.0, coupon_spread=0.01)
    res = CDSReducedFormEngine().calculate(cds, _env(rate=0.0, hazard=0.03))
    expected = 1_000_000 * (1 - math.exp(-0.03 * 5.0))
    assert res["protection_leg"] == pytest.approx(expected, rel=1e-3)


def test_protection_leg_scales_with_loss_given_default():
    base = CDS(notional=1_000_000, maturity=5.0, recovery_rate=0.0, coupon_spread=0.01)
    half = CDS(notional=1_000_000, maturity=5.0, recovery_rate=0.5, coupon_spread=0.01)
    eng, env = CDSReducedFormEngine(), _env(rate=0.02, hazard=0.03)
    assert eng.calculate(half, env)["protection_leg"] == pytest.approx(
        0.5 * eng.calculate(base, env)["protection_leg"], rel=1e-9
    )


def test_pv_is_zero_at_fair_spread():
    eng, env = CDSReducedFormEngine(), _env(rate=0.03, hazard=0.025)
    cds = CDS(notional=1_000_000, maturity=5.0, recovery_rate=0.4, coupon_spread=0.01)
    fs = eng.fair_spread(cds, env)
    at_fair = CDS(notional=1_000_000, maturity=5.0, recovery_rate=0.4, coupon_spread=fs)
    assert eng.price(at_fair, env) == pytest.approx(0.0, abs=1.0)


def test_fair_spread_credit_triangle_approximation():
    # s ~= lambda * (1 - R) for small lambda
    eng, env = CDSReducedFormEngine(), _env(rate=0.0, hazard=0.01)
    cds = CDS(notional=1_000_000, maturity=5.0, recovery_rate=0.4, coupon_spread=0.0)
    fs = eng.fair_spread(cds, env)
    assert fs == pytest.approx(0.01 * (1 - 0.4), rel=0.05)


def test_fair_spread_increases_with_hazard():
    eng = CDSReducedFormEngine()
    cds = CDS(notional=1_000_000, maturity=5.0, recovery_rate=0.4, coupon_spread=0.0)
    fs_low = eng.fair_spread(cds, _env(hazard=0.01))
    fs_high = eng.fair_spread(cds, _env(hazard=0.05))
    assert fs_high > fs_low


def test_protection_buyer_and_seller_pv_are_opposite():
    eng, env = CDSReducedFormEngine(), _env(rate=0.03, hazard=0.04)
    buyer = CDS(notional=1_000_000, maturity=5.0, recovery_rate=0.4,
                coupon_spread=0.01, side=ProtectionSide.BUY)
    seller = CDS(notional=1_000_000, maturity=5.0, recovery_rate=0.4,
                 coupon_spread=0.01, side=ProtectionSide.SELL)
    assert eng.price(buyer, env) == pytest.approx(-eng.price(seller, env), rel=1e-9)
