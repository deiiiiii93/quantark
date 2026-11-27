"""MC's REBASED route pays the resolver's cash, per path (spec §5, §6)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.accrual import contract_tenor, knock_in_anchor_for, ko_accrual_factor
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, PayoffConfig
from quantark.util.enum import PostKOScheduleMode as M

from .conftest import env, ko_reset, times


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


def test_rebased_v1_tenor_runs_to_the_exercise_whatever_the_anchor():
    """Owner ruling O2 (2026-11-27): the knocked-in tenor runs to the exercise date (the maturity 1.25) on every path;
    it hung from each path's knock-in anchor before (0.25 / 0.5 + the final offset 0.75: 1.0 / 1.25)."""
    # numeric clock alone (clock rule 2026-10-26: no initial_date on a numeric contract)
    p = ko_reset(M.REBASED, accrual=AccrualConfig(is_annualized_ki=True), payoff_config=PayoffConfig(rebate_rate=0.05))
    ki = p.barrier_config.ki_observation_schedule.records
    tenors = {contract_tenor(p, None, "post", knock_in=knock_in_anchor_for(r, i)) for i, r in enumerate(ki)}
    assert tenors == {1.25} == {contract_tenor(p, None, "post")}
    _price_parts(p)   # used to raise "ki_time is required for rebased post-KO maturity"


def test_unannualized_rebased_needs_no_anchor():
    p = ko_reset(M.REBASED, accrual=AccrualConfig(is_annualized=False))
    post = p.post_barrier_config.ko_observation_schedule.records[0]
    assert ko_accrual_factor(p, post) == 1.0
    _price_parts(p)
