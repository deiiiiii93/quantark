"""KO-reset theta ages every schedule (user request 2026-10-08): ``time_shift`` moves an ABSOLUTE schedule 2 a day
closer too, so the one-day theta equals the correctly aged contract's, its cash unchanged.

Before the fix (main and branch alike) schedule 2 stayed where it was: alive contracts' theta came out about 10%
low and knocked-in ones of the wrong sign (-0.0256 against +0.0489).
"""
from dataclasses import replace
from datetime import timedelta

import pytest

from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.accrual import ko_accrual_factor
from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.util.enum import PostKOScheduleMode as M
from term_structure_benchmarks import make_term_env

from .conftest import T0, env, ko_reset

BOOK = dict(initial_price=100.0, strike=100.0, maturity_pre=0.25, maturity_post=0.5, pre_ko_barrier=103.0,
            pre_ko_rate=0.15, post_ko_barrier=95.0, post_ko_rate=0.03, ki_barrier=80.0, ki_continuous=True,
            contract_multiplier=1.0)
CASES = {
    "cont|alive": (dict(BOOK), False),
    "disc|alive": (dict(BOOK, ki_continuous=False, ki_monitoring="discrete"), False),
    "cont|knocked_in": (dict(BOOK), True),
    "disc|knocked_in": (dict(BOOK, ki_continuous=False, ki_monitoring="discrete"), True),
}


def _hand_aged(self, time_bump, bumped_date, pricing_env, *, keep_landing=False):
    """The contract aged by hand: the snowball's schedules by its own time_shift, schedule 2 (records and the legacy
    times the builder also states it as) by the same day, its records keeping their accrual periods. A record landing
    on the bumped instant is kept at time 0 when theta asks (``keep_landing``, spec 2026-10-07)."""
    SnowballOption.time_shift(self, time_bump, bumped_date, pricing_env, keep_landing=keep_landing)
    post = self.post_barrier_config
    legacy = post.ko_observation_dates
    shifted = post.ko_observation_schedule.time_shift(time_bump, bumped_date, keep_landing=keep_landing)
    self.post_barrier_config = replace(
        post, ko_observation_schedule=shifted,
        ko_observation_dates=None if legacy is None else [r.observation_time for r in shifted.records])


def _reference_theta(product, market, engine, monkeypatch):
    """The calculator's own theta (same bump context, same frozen grid) on the hand-aged contract."""
    with monkeypatch.context() as patch:
        patch.setattr(type(product), "time_shift", _hand_aged)
        return GreeksCalculator().calculate_numerical_theta(product, market, engine)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_library_theta_is_the_correctly_aged_contracts(name, monkeypatch):
    spec, knocked_in = CASES[name]
    product = make_ko_reset(spec)
    setattr(product, "_otc_lifecycle_knocked_in", knocked_in)
    market, engine = make_term_env("flat"), KOResetSnowballPDESolver(PDEParams())
    theta = GreeksCalculator().calculate_numerical_theta(product, market, engine)
    assert theta == _reference_theta(product, market, engine, monkeypatch)
    assert (theta > 0.0) == True   # a knocked-in contract's theta is no longer of the wrong sign


def test_ageing_moves_schedule_two_and_keeps_its_cash():
    p = ko_reset(M.ABSOLUTE, accrual=AccrualConfig(post_accrual_factors=None))
    post = p.post_barrier_config.ko_observation_schedule.records
    cash = [ko_accrual_factor(p, r) for r in post]
    p.time_shift(30 / 365, T0 + timedelta(days=30), env())
    moved = p.post_barrier_config.ko_observation_schedule.records
    assert [r.observation_time for r in moved] == pytest.approx([0.25 - 30 / 365, 0.5 - 30 / 365, 0.75 - 30 / 365])
    assert [ko_accrual_factor(p, r) for r in moved] == cash


def test_a_rebased_schedule_two_keeps_its_offsets():
    p = ko_reset(M.REBASED)
    before = [r.observation_time for r in p.post_barrier_config.ko_observation_schedule.records]
    p.time_shift(30 / 365, T0 + timedelta(days=30), env())
    assert [r.observation_time for r in p.post_barrier_config.ko_observation_schedule.records] == before


def test_a_knocked_in_contract_past_its_first_schedule_still_ages():
    """Schedule 1 and the knock-ins are behind it; schedule 2 is not, so nothing says 'every observation passed'."""
    p = ko_reset(M.ABSOLUTE)
    assert p.time_shift(0.3, T0 + timedelta(days=110), env()) is False
    assert len(p.post_barrier_config.ko_observation_schedule.records) == 2
    assert p.time_shift(0.5, T0 + timedelta(days=300), env()) is True


def test_post_factors_follow_the_dropped_records():
    p = ko_reset(M.ABSOLUTE, accrual=AccrualConfig(post_accrual_factors=[0.1, 0.2, 0.3]))
    p.time_shift(0.3, T0 + timedelta(days=110), env())
    assert p.accrual_config.post_accrual_factors == [0.2, 0.3]
    p.validate()
    assert [ko_accrual_factor(p, r) for r in p.post_barrier_config.ko_observation_schedule.records] == [0.2, 0.3]
