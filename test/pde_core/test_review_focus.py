"""Inputs the spec implies and no spec test exercises (plan "Review Focus")."""
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from term_structure_benchmarks import make_term_env

SB = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0, ko_rate=0.15,
          rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0)


def test_cache_disabled_agrees_with_cache_enabled_to_golden_tolerance():
    # Review Focus 5.
    from golden_compare import assert_close
    p = make_snowball(SB); env = make_term_env("kinked")
    on = SnowballPDESolver(PDEParams()).price(p, env)
    off = SnowballPDESolver(PDEParams(cache_enabled=False)).price(p, env)
    assert_close(off, on)   # not ==: the rounded dt key shares ULP-different intervals (§5.7)


# --- Review Focus 1 and 2 (Task 5) ---------------------------------------------
import copy
import pickle
from datetime import datetime

import pytest

from quantark.asset.equity.engine.pde import AmericanPDESolver, BarrierPDESolver, EuropeanPDESolver
from quantark.asset.equity.engine.settlement_support import american_exercise_requires_dates
from quantark.asset.equity.product.option import AmericanOption
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.calendar import Calendar
from quantark.util.enum import ObservationType, OptionType

from intraday.conftest import flat_env
from pde_bitwise.capture_book import american_tenor, barrier, env_flat, european


@pytest.mark.parametrize("cls,make", [(EuropeanPDESolver, european), (AmericanPDESolver, american_tenor),
                                      (BarrierPDESolver, lambda: barrier(ObservationType.DISCRETE))])
def test_solver_deep_copies_and_pickles_after_pricing(cls, make):
    # Review Focus 1.
    s = cls(PDEParams()); p = make(); env = env_flat()
    a = s.price(p, env)
    assert copy.deepcopy(s).price(p, env) == a
    assert pickle.loads(pickle.dumps(s)).price(p, env) == a


def test_american_greeks_on_a_fresh_solver_equal_greeks_after_price():
    # Review Focus 2. Mirrors test_american_option_settlement._business_day_lag.
    cal = Calendar()
    conv = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.BUSINESS_DAYS, calendar=cal)
    p = AmericanOption(strike=100.0, option_type=OptionType.PUT, exercise_date=datetime(2027, 3, 15),
                       settlement_convention=conv)
    env = flat_env(datetime(2026, 3, 16)); env.calendar = cal
    assert american_exercise_requires_dates(p)
    fresh = AmericanPDESolver(PDEParams()).calculate_greeks(p, env)
    s = AmericanPDESolver(PDEParams()); s.price(p, env)
    after = s.calculate_greeks(p, env)
    assert fresh == after


# --- Review Focus 1 for the snowball family (Task 6) -----------------------------
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver

from pde_bitwise.capture_book import env_grid_vol, snowball


@pytest.mark.parametrize("cls,make,env", [(SnowballPDESolver, lambda: snowball("continuous"), env_flat),
                                          (LocalVolSnowballPDESolver, lambda: snowball("continuous"), env_grid_vol)])
def test_snowball_solver_deep_copies_and_pickles_after_pricing(cls, make, env):
    s = cls(PDEParams()); p = make(); e = env()
    a = s.price(p, e)
    assert copy.deepcopy(s).price(p, e) == a
    assert pickle.loads(pickle.dumps(s)).price(p, e) == a
