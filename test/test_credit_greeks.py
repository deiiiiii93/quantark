"""Tests for credit risk measures (CS01, IR01, Rec01)."""
from datetime import datetime

import pytest

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.product import CDS, ProtectionSide
from quantark.asset.credit.riskmeasures import CreditGreeksCalculator
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.priceenv import CreditPricingEnvironment


def _env(rate=0.03, hazard=0.02):
    return CreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=rate),
        hazard_curve=FlatHazardCurve(hazard_rate=hazard),
    )


def _cds(side=ProtectionSide.BUY, spread=0.01):
    return CDS(notional=10_000_000, maturity=5.0, recovery_rate=0.4,
               coupon_spread=spread, side=side)


def test_cs01_positive_for_protection_buyer():
    calc, eng = CreditGreeksCalculator(), CDSReducedFormEngine()
    cs01 = calc.cs01(_cds(side=ProtectionSide.BUY), _env(), eng)
    assert cs01 > 0  # buyer gains when spreads widen


def test_cs01_opposite_signs_for_buyer_and_seller():
    calc, eng = CreditGreeksCalculator(), CDSReducedFormEngine()
    buy = calc.cs01(_cds(side=ProtectionSide.BUY), _env(), eng)
    sell = calc.cs01(_cds(side=ProtectionSide.SELL), _env(), eng)
    assert buy == pytest.approx(-sell, rel=1e-6)


def test_rec01_negative_for_protection_buyer():
    # Higher recovery -> lower loss given default -> protection worth less.
    calc, eng = CreditGreeksCalculator(), CDSReducedFormEngine()
    rec01 = calc.rec01(_cds(side=ProtectionSide.BUY), _env(), eng)
    assert rec01 < 0


def test_calculate_returns_all_measures():
    calc, eng = CreditGreeksCalculator(), CDSReducedFormEngine()
    greeks = calc.calculate(_cds(), _env(), eng)
    for key in ("price", "cs01", "ir01", "rec01"):
        assert key in greeks


def test_cs01_scales_with_notional():
    calc, eng = CreditGreeksCalculator(), CDSReducedFormEngine()
    small = calc.cs01(CDS(notional=1_000_000, maturity=5.0, recovery_rate=0.4,
                          coupon_spread=0.01), _env(), eng)
    big = calc.cs01(CDS(notional=10_000_000, maturity=5.0, recovery_rate=0.4,
                        coupon_spread=0.01), _env(), eng)
    assert big == pytest.approx(10 * small, rel=1e-6)
