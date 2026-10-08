"""The same accrual in the product, every engine and the intraday path (spec §8 'One answer everywhere')."""
from datetime import timedelta

import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.accrual import ko_accrual_factor
from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.intraday.events import phoenix_coupon_fractions

from .conftest import T0, dates, env, phoenix, snowball, times


def test_knock_out_at_the_valuation_instant_pays_the_resolvers_cash():
    """An aged contract (offset 0.25) whose first record is at time zero, spot above the barrier."""
    p = snowball(ko=times(0.0, 0.25), maturity=0.25, accrual=AccrualConfig(accrued_offset=0.25))
    cash = 100.0 + 100.0 * 0.12 * ko_accrual_factor(p, p.barrier_config.ko_observation_schedule.records[0])
    e = env(spot=110.0)
    for engine in (SnowballMCEngine(params=MCParams(seed=1, num_paths=512)), SnowballPDESolver(PDEParams()),
                   SnowballQuadEngine(params=QuadParams(grid_points=401))):
        assert engine.price(p, e) == pytest.approx(cash, rel=0, abs=1e-12), type(engine).__name__


@pytest.mark.parametrize("dated", [False, True])
def test_phoenix_fractions_agree(dated):
    kw = dict(ko=dates(30, 61, 91), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=91)) if dated \
        else dict(ko=times(1 / 12, 2 / 12, 3 / 12), accrual=AccrualConfig(accrued_offset=0.1, first_period_start=0.05))
    p = phoenix(**kw)
    product = p.get_coupon_period_year_fractions(env())
    intraday = phoenix_coupon_fractions(p, [None] * 3)
    assert product == intraday == [p.coupon_period_fraction(i) for i in range(3)]
    assert intraday[0] == (30 / 365 if dated else 0.1 + 1 / 12 - 0.05)


def _spy_fractions(monkeypatch, product_type):
    seen = []
    real = product_type.coupon_period_fraction
    def spy(self, i):
        seen.append(i)
        return real(self, i)
    monkeypatch.setattr(product_type, "coupon_period_fraction", spy)
    return seen


def test_quad_v2_coupons_are_the_resolvers(monkeypatch):
    from quantark.asset.equity.engine.quad.v2.adapters import compile_contract
    from quantark.asset.equity.param.quad_v2_params import QuadV2Params
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    p = phoenix(ko=dates(30, 61, 91), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=91))
    seen = _spy_fractions(monkeypatch, PhoenixOption)
    compile_contract(p, env(), QuadV2Params())
    assert seen == [0, 1, 2]


def test_quad_v2_fractions_follow_source_index_after_phase(monkeypatch):
    """Review Focus 4: valued on the second observation date, after its events: coupons of record 2 only."""
    from quantark.asset.equity.engine.quad.v2.adapters import compile_contract
    from quantark.asset.equity.param.quad_v2_params import QuadV2Params
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    p = phoenix(ko=dates(30, 61, 91), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=91))
    seen = _spy_fractions(monkeypatch, PhoenixOption)
    compile_contract(p, env(T0 + timedelta(days=61)), QuadV2Params(), event_phase="after")
    assert seen == [2]
