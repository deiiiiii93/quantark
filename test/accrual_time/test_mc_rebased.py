"""MC's REBASED route pays the resolver's cash, per path (spec §5, §6)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.accrual import contract_tenor, knock_in_anchor_for, ko_accrual_factor
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, PayoffConfig
from quantark.util.enum import PostKOScheduleMode as M

from .conftest import T0, env, ko_reset, times


def _price_parts(product):
    engine = SnowballMCEngine(params=MCParams(seed=7, num_paths=4096))
    e = env(spot=100.0)
    engine.price(product, e)
    return engine, e


def test_rebased_mc_ko_cash_is_the_resolvers(monkeypatch):
    p = ko_reset(M.REBASED, accrual=AccrualConfig(accrued_offset=0.3))
    seen = []
    real = type(p).compute_ko_accrual_factor
    def spy(self, t, rec, env_, *, knock_in=None):
        value = real(self, t, rec, env_, knock_in=knock_in)
        seen.append((rec.accrual.end, None if knock_in is None else knock_in.fixing, value))
        return value
    monkeypatch.setattr(type(p), "compute_ko_accrual_factor", spy)
    _price_parts(p)
    rebased = [s for s in seen if s[1] is not None]
    assert rebased and all(v == f + end for end, f, v in rebased)


def test_rebased_v1_tenor_is_per_path_anchor():
    p = ko_reset(M.REBASED, accrual=AccrualConfig(is_annualized_ki=True), payoff_config=PayoffConfig(rebate_rate=0.05),
                 initial_date=T0)   # the pre-existing flag rule wants an initial_date; numeric periods ignore it
    ki = p.barrier_config.ki_observation_schedule.records
    tenors = {contract_tenor(p, None, "post", knock_in=knock_in_anchor_for(r, i)) for i, r in enumerate(ki)}
    assert tenors == {1.0, 1.25}  # final offset 0.75 after fixings 0.25 and 0.5
    _price_parts(p)   # used to raise "ki_time is required for rebased post-KO maturity"


def test_unannualized_rebased_needs_no_anchor():
    p = ko_reset(M.REBASED, accrual=AccrualConfig(is_annualized=False))
    post = p.post_barrier_config.ko_observation_schedule.records[0]
    assert ko_accrual_factor(p, post) == 1.0
    _price_parts(p)
