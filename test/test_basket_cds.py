"""Tests for basket CDS (copula Monte-Carlo) pricing."""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.engine.mc import BasketCDSEngine
from quantark.asset.credit.product import CDS, BasketCDS, BasketType, CopulaType
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.priceenv import BasketCreditPricingEnvironment, CreditPricingEnvironment

N = 5
NOTIONAL = 10_000_000.0
MATURITY = 5.0
RECOVERY = 0.4
HAZARD = 0.02
RATE = 0.03


def _corr(n, rho):
    m = np.full((n, n), rho)
    np.fill_diagonal(m, 1.0)
    return m


def _basket(rho=0.3, basket_type=BasketType.FTD, n_to_default=1):
    product = BasketCDS(
        notional=NOTIONAL, maturity=MATURITY, recovery_rates=[RECOVERY] * N,
        basket_type=basket_type, n_to_default=n_to_default,
        correlation_matrix=_corr(N, rho),
    )
    env = BasketCreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=RATE),
        hazard_curves=[FlatHazardCurve(hazard_rate=HAZARD)] * N,
    )
    return product, env


def _single_name_spread():
    env = CreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=RATE),
        hazard_curve=FlatHazardCurve(hazard_rate=HAZARD),
    )
    cds = CDS(notional=NOTIONAL, maturity=MATURITY, recovery_rate=RECOVERY, coupon_spread=0.0)
    return CDSReducedFormEngine().fair_spread(cds, env)


def test_ftd_spread_between_single_name_and_sum():
    eng = BasketCDSEngine(n_simulations=40_000, seed=7)
    sn = _single_name_spread()
    ftd = eng.fair_spread(*_basket(rho=0.3))
    assert sn < ftd < N * sn


def test_ftd_spread_decreases_with_correlation():
    eng = BasketCDSEngine(n_simulations=40_000, seed=7)
    low_corr = eng.fair_spread(*_basket(rho=0.05))
    high_corr = eng.fair_spread(*_basket(rho=0.95))
    assert high_corr < low_corr


def test_second_to_default_cheaper_than_first():
    eng = BasketCDSEngine(n_simulations=40_000, seed=7)
    first = eng.fair_spread(*_basket(basket_type=BasketType.NTD, n_to_default=1))
    second = eng.fair_spread(*_basket(basket_type=BasketType.NTD, n_to_default=2))
    assert second < first


def test_ntd_first_matches_ftd():
    eng = BasketCDSEngine(n_simulations=40_000, seed=7)
    ftd = eng.fair_spread(*_basket(basket_type=BasketType.FTD))
    ntd1 = eng.fair_spread(*_basket(basket_type=BasketType.NTD, n_to_default=1))
    assert ntd1 == pytest.approx(ftd, rel=1e-9)


def test_pv_zero_at_fair_spread():
    eng = BasketCDSEngine(n_simulations=20_000, seed=11)
    product, env = _basket()
    fs = eng.fair_spread(product, env)
    priced = BasketCDS(
        notional=NOTIONAL, maturity=MATURITY, recovery_rates=[RECOVERY] * N,
        basket_type=BasketType.FTD, correlation_matrix=_corr(N, 0.3), coupon_spread=fs,
    )
    # Same seed -> same default-time sample -> PV exactly zero at the par spread.
    assert eng.price(priced, env) == pytest.approx(0.0, abs=NOTIONAL * 1e-6)


def test_t_copula_runs_and_is_positive():
    eng = BasketCDSEngine(n_simulations=20_000, seed=3)
    product, env = _basket()
    product.copula_type = CopulaType.STUDENT_T
    assert eng.fair_spread(product, env) > 0
