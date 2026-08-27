"""Tests for the Richardson pair extrapolation harness."""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc import (
    LocalVolSnowballMCEngine,
    RichardsonPairResult,
    richardson_pair_price,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import SnowballOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol.surface import LocalVolSurface


class _StubEngine:
    def __init__(self, price, se):
        self._price, self._se = price, se

    def price(self, product, env):
        return self._price

    def get_last_std_error(self):
        return self._se


def test_pair_arithmetic_exact():
    def factory(n):
        return {1: _StubEngine(10.0, 0.5), 2: _StubEngine(9.4, 0.3)}[n]

    r = richardson_pair_price(factory, None, None, substeps=1)
    assert isinstance(r, RichardsonPairResult)
    assert r.price == 2.0 * 9.4 - 10.0
    assert r.coarse_substeps == 1 and r.fine_substeps == 2
    assert r.std_error == pytest.approx(np.sqrt(4 * 0.3**2 + 0.5**2))


def test_pair_without_stderr_reports_none():
    class _Bare:
        def price(self, product, env):
            return 1.0

    r = richardson_pair_price(lambda n: _Bare(), None, None)
    assert r.price == 1.0 and r.std_error is None


def test_pair_rejects_shared_leg_seed():
    """A factory that seeds both legs identically couples their draw streams,
    invalidating the independent-legs std_error -- it must raise, not report."""

    class _Seeded:
        class params:
            seed = 42

        def price(self, product, env):  # pragma: no cover - never reached
            return 1.0

    with pytest.raises(ValidationError):
        richardson_pair_price(lambda n: _Seeded(), None, None)


def test_pair_rejects_bad_substeps():
    with pytest.raises(ValidationError):
        richardson_pair_price(lambda n: _StubEngine(1.0, 0.1), None, None, substeps=0)
    with pytest.raises(ValidationError):
        richardson_pair_price(lambda n: _StubEngine(1.0, 0.1), None, None, substeps=True)


def test_pair_is_noop_on_flat_surface_snowball():
    """Flat surface: the LV step is exact at any substep count, so the pair must
    agree with a single run within combined MC noise."""
    strikes = np.geomspace(40.0, 250.0, 21)
    times = np.array([0.0, 0.5, 1.0])
    surf = LocalVolSurface(strikes, times, np.full((3, 21), 0.25))
    product = SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15,
            ko_observation_dates=[i / 12 for i in range(1, 13)],
            ki_barrier=75.0, ki_continuous=True,
        ),
        contract_multiplier=1.0, maturity=1.0,
    )
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.0),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(0.25),
        div_yield=ContinuousDividendYield(0.0),
    )

    def factory(n):
        return LocalVolSnowballMCEngine(
            params=MCParams(num_paths=60_000, seed=100 + n),
            local_vol_surface=surf,
            substeps_per_interval=n,
        )

    r = richardson_pair_price(factory, product, env, substeps=1)
    single = factory(1)
    p1 = single.price(product, env)
    tol = 5.0 * float(np.hypot(r.std_error, single.get_last_std_error()))
    assert abs(r.price - p1) < tol
