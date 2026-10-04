"""Contractual coupon periods (spec 3B decision 13, §4.7)."""
import dataclasses
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.v2.engine import PhoenixQuadEngineV2
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.exceptions import ValidationError
from intraday.conftest import dated_phoenix, flat_env
from pde_3a.products import PHOENIX, env

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0 = datetime(2026, 3, 16)


def _obs(p, i):
    return p.barrier_config.ko_observation_schedule.records[i].observation_date


def test_first_period_on_its_observation_date_is_contractual():
    p = dated_phoenix(CAL, T0, memory=False)
    env_ = flat_env(_obs(p, 0), spot=95.0)
    f = p.get_coupon_period_year_fractions(env_)
    assert p.get_coupon_payoff(0, year_fraction=f[0]) == pytest.approx(1.019178, abs=1e-6)


def test_coupon_day_count_is_the_coupons():
    p = dated_phoenix(CAL, T0, memory=False)
    p.coupon_config = dataclasses.replace(p.coupon_config, day_count_convention=DayCountConvention.ACT_360)
    f = p.get_coupon_period_year_fractions(flat_env(_obs(p, 0), spot=95.0))
    assert p.get_coupon_payoff(0, year_fraction=f[0]) == pytest.approx(1.033333, abs=1e-6)


def test_one_business_day_after_the_first_observation_keeps_the_next_period_whole():
    p = dated_phoenix(CAL, T0, memory=False)
    day_after = CAL.add_business_days(_obs(p, 0), 1)
    f = p.get_coupon_period_year_fractions(flat_env(day_after, spot=95.0))
    assert f[0] == pytest.approx(p.get_coupon_year_fraction(_obs(p, 0), _obs(p, 1)), abs=1e-15)


def test_a_list_of_times_is_refused():
    p = dated_phoenix(CAL, T0, memory=False)
    with pytest.raises(ValidationError, match="pricing environment"):
        p.get_coupon_period_year_fractions([0.1, 0.2])


def test_deterministic_control_is_the_contractual_value():
    # review 2 finding 1 / evidence p13 B: vol 1 %, knock-out unreachable, spot above the 80 coupon barrier
    p = dated_phoenix(CAL, T0, memory=False, ko=10000.0)
    env_ = flat_env(_obs(p, 0), spot=95.0, vol=0.01)
    pde = PhoenixPDESolver(PDEParams()).price(p, env_)
    v2 = PhoenixQuadEngineV2().price(p, env_)
    assert pde == pytest.approx(109.129238, abs=1e-6) and v2 == pytest.approx(109.129238, abs=1e-6)


def _tenor():
    p = make_phoenix(PHOENIX)                                          # tenor schedule, no initial_date
    assert p.initial_date is None and p.accrual_config.accrual_factors is None
    assert p.coupon_config.fixed_coupon_year_fraction is None
    return p


def test_aged_tenor_product_past_an_observation_refuses():
    p, e = _tenor(), env()
    t1 = float(p.resolve_ko_observations(e)[0].observation_time)
    bump = t1 + 1.0 / 365.0
    p.time_shift(bump, e.valuation_date + timedelta(days=int(np.ceil(bump * 365.0))), e)
    assert p.accrual_config.dropped_observations == 1
    with pytest.raises(ValidationError, match="fixed_coupon_year_fraction.*accrual_factors"):
        p.get_coupon_period_year_fractions(e)


def test_theta_bump_inside_the_first_period_keeps_the_contractual_fraction():
    p, e = _tenor(), env()
    before = p.get_coupon_period_year_fractions(e)
    p.time_shift(1.0 / 365.0, e.valuation_date + timedelta(days=1), e)
    assert p.accrual_config.dropped_observations == 0
    assert p.get_coupon_period_year_fractions(e) == pytest.approx(before, abs=1e-15)


@pytest.mark.slow
@pytest.mark.parametrize("when", ["observation_date", "ten_days_in"])
def test_three_engines_agree(when):
    """PDE and V2 within 2e-4; Monte Carlo within three batch standard errors of eight 25,000-path runs."""
    p = dated_phoenix(CAL, T0, memory=False)
    day = _obs(p, 0) if when == "observation_date" else CAL.add_business_days(T0, 10)
    env_ = flat_env(day, spot=95.0)
    pde = PhoenixPDESolver(PDEParams()).price(p, env_)
    v2 = PhoenixQuadEngineV2().price(p, env_)
    assert abs(pde - v2) <= 2e-4
    runs = np.array([PhoenixMCEngine(params=MCParams(num_paths=25_000, seed=seed)).price(p, env_)
                     for seed in range(1, 9)])
    se = runs.std(ddof=1) / np.sqrt(runs.size)
    assert abs(pde - runs.mean()) <= 3.0 * se, (pde, runs.mean(), se)
