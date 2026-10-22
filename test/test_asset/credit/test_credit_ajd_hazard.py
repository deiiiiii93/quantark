"""Tests for the AJD (CIR-jump) stochastic hazard-rate curve."""
import math
from datetime import datetime

import pytest

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.product import CDS
from quantark.param import FlatRateCurve
from quantark.param.credit import AJDHazardCurve, FlatHazardCurve, HazardCurve
from quantark.priceenv import CreditPricingEnvironment


def test_ajd_is_a_hazard_curve():
    assert isinstance(
        AJDHazardCurve(lambda0=0.02, kappa=0.5, theta=0.02, sigma=0.1), HazardCurve
    )


def test_ajd_survival_is_monotone_decreasing_and_starts_at_one():
    curve = AJDHazardCurve(lambda0=0.03, kappa=0.5, theta=0.04, sigma=0.1, gamma=0.05)
    assert curve.get_survival_probability(0.0) == pytest.approx(1.0, abs=1e-6)
    s1 = curve.get_survival_probability(1.0)
    s5 = curve.get_survival_probability(5.0)
    assert 0.0 < s5 < s1 < 1.0


def test_ajd_reduces_to_flat_hazard_in_degenerate_limit():
    # sigma=0, gamma=0, kappa->0: dbeta/dt = -1 => beta(t) = -t, alpha(t) -> 0,
    # so S(t) -> exp(-lambda0 * t), matching a flat hazard curve.
    lam = 0.025
    ajd = AJDHazardCurve(lambda0=lam, kappa=1e-7, theta=lam, sigma=0.0, gamma=0.0)
    flat = FlatHazardCurve(hazard_rate=lam)
    for t in (1.0, 3.0, 5.0):
        assert ajd.get_survival_probability(t) == pytest.approx(
            flat.get_survival_probability(t), rel=1e-3
        )


def test_ajd_default_density_is_negative_dsurvival():
    curve = AJDHazardCurve(lambda0=0.03, kappa=0.5, theta=0.04, sigma=0.1)
    t = 2.0
    dt = 0.01
    fd = (curve.get_survival_probability(t - dt)
          - curve.get_survival_probability(t + dt)) / (2 * dt)
    assert curve.get_default_density(t) == pytest.approx(fd, rel=1e-2)


def test_ajd_cds_prices_with_same_engine():
    env = CreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=0.03),
        hazard_curve=AJDHazardCurve(lambda0=0.02, kappa=0.5, theta=0.03,
                                    sigma=0.1, gamma=0.05, mu_jump=0.02),
    )
    cds = CDS(notional=1_000_000, maturity=5.0, recovery_rate=0.4, coupon_spread=0.0)
    fs = CDSReducedFormEngine().fair_spread(cds, env)
    assert fs > 0
