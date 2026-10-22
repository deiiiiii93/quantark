"""Tests for the one-step-survival (OSS) estimator on LocalVolSnowballMCEngine.

The OSS estimator (Glasserman-Staum conditioning at KO dates, Alm et al. branch
factorization) must agree with the plain estimator in expectation (A/B PV gate),
report consistent event probabilities, produce bump-stable CRN finite-difference
Greeks, and refuse the product/engine combinations its factorization does not
cover.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc import LocalVolSnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import SnowballOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol.surface import LocalVolSurface

STRIKES = np.geomspace(40.0, 250.0, 31)
TIMES = np.array([0.0, 1 / 52, 1 / 12, 0.25, 0.5, 0.75, 1.0])


def _steep_surface():
    term = 0.22 + 0.12 * np.exp(-3.5 * TIMES)
    skew = np.clip((STRIKES / 100.0) ** (-0.9), 0.45, 2.2)
    return LocalVolSurface(STRIKES.copy(), TIMES.copy(), term[:, None] * skew[None, :])


def _product(**barrier_overrides):
    barrier_kwargs = dict(
        ko_barrier=103.0, ko_rate=0.15,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=75.0, ki_continuous=True,
    )
    barrier_kwargs.update(barrier_overrides)
    return SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(**barrier_kwargs),
        contract_multiplier=1.0, maturity=1.0,
    )


def _env(s0=100.0):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.0),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=s0),
        vol_surface=FlatVolSurface(0.25),
        div_yield=ContinuousDividendYield(0.0),
    )


def _engine(estimator="plain", num_paths=60_000, seed=42, **kwargs):
    return LocalVolSnowballMCEngine(
        params=MCParams(num_paths=num_paths, seed=seed),
        local_vol_surface=_steep_surface(),
        estimator=estimator,
        **kwargs,
    )


def test_oss_pv_matches_plain_within_noise():
    product, env = _product(), _env()
    e_plain = _engine("plain")
    p_plain = e_plain.price(product, env)
    e_oss = _engine("one_step_survival")
    p_oss = e_oss.price(product, env)
    tol = 5.0 * float(
        np.hypot(e_plain.get_last_std_error(), e_oss.get_last_std_error())
    )
    assert abs(p_oss - p_plain) < tol


def test_oss_event_probabilities_consistent():
    product, env = _product(), _env()
    e_plain = _engine("plain")
    e_plain.price(product, env)
    r_plain = e_plain.get_last_result()
    e_oss = _engine("one_step_survival")
    e_oss.price(product, env)
    r_oss = e_oss.get_last_result()
    assert abs(r_oss.ko_probability - r_plain.ko_probability) < 0.02
    assert abs(r_oss.v1_probability - r_plain.v1_probability) < 0.02
    total = r_oss.ko_probability + r_oss.v0_probability + r_oss.v1_probability
    assert abs(total - 1.0) < 1e-6


def test_oss_with_substeps_prices_and_steers_correctly():
    """The KO-date mapping onto the refined grid: a wrong mapping steers the
    wrong step and trips the hard-KO consistency check (NumericalError)."""
    product, env = _product(), _env()
    e1 = _engine("one_step_survival", num_paths=30_000)
    e2 = _engine("one_step_survival", num_paths=30_000, substeps_per_interval=2)
    p1, p2 = e1.price(product, env), e2.price(product, env)
    tol = 5.0 * float(np.hypot(e1.get_last_std_error(), e2.get_last_std_error()))
    assert abs(p1 - p2) < tol + 0.05  # small residual for the substep bias


def test_oss_quasi_mode_agrees_with_pseudo():
    product, env = _product(), _env()
    e_p = _engine("one_step_survival", num_paths=40_000)
    p_pseudo = e_p.price(product, env)
    e_q = _engine("one_step_survival", num_paths=40_000,
                  method=MonteCarloMethod.QUASI)
    p_quasi = e_q.price(product, env)
    tol = 5.0 * float(np.hypot(e_p.get_last_std_error(),
                               max(e_q.get_last_std_error(), 1e-6)))
    assert abs(p_quasi - p_pseudo) < tol


def test_oss_crn_gamma_is_bump_stable():
    """CRN finite-difference gamma must be stable from a 2% bump down to a
    0.1% bump under OSS (the estimator is Lipschitz in spot)."""
    product = _product()

    def gamma(estimator, rel_bump):
        e0 = _engine(estimator)
        p0 = e0.price(product, _env(100.0))
        pu = _engine(estimator).price(product, _env(100.0 * (1 + rel_bump)))
        pd = _engine(estimator).price(product, _env(100.0 * (1 - rel_bump)))
        return (pu - 2 * p0 + pd) / (100.0 * rel_bump) ** 2

    g_small = gamma("one_step_survival", 0.001)
    g_large = gamma("one_step_survival", 0.02)
    assert abs(g_small - g_large) < 0.03
    assert np.isfinite(g_small)


def test_oss_rejects_unsupported_engine_configs():
    with pytest.raises(ValidationError):
        _engine("one_step_survival", method=MonteCarloMethod.RANDOMIZED_QUASI)
    with pytest.raises(ValidationError):
        LocalVolSnowballMCEngine(
            params=MCParams(num_paths=1000, seed=1, use_antithetic=True),
            local_vol_surface=_steep_surface(),
            estimator="one_step_survival",
        )
    with pytest.raises(ValidationError):
        _engine("bogus")


def test_oss_rejects_unsupported_products():
    env = _env()
    e = _engine("one_step_survival", num_paths=1000)
    with pytest.raises(ValidationError):
        e.price(_product(disable_ko_after_ki=True), env)
    with pytest.raises(ValidationError):
        e.price(
            _product(ko_observation_type=ObservationType.CONTINUOUS), env
        )


def test_oss_rejects_event_stats():
    e = _engine("one_step_survival", num_paths=1000)
    with pytest.raises(ValidationError):
        e.calculate_event_stats(_product(), _env())
