"""Untracked schedule-free float-maturity products roll daily (patch spec 2026-09-03 §4)."""
import warnings
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine, DeltaOneEngine
from quantark.asset.equity.lifecycle import PortfolioLifecycleManager
from quantark.asset.equity.lifecycle.float_roll import (
    FLOAT_ROLLABLE_PRODUCTS, MATURITY_FLOOR, FloatMaturityRoller, has_unrolled_float_maturity, is_float_rollable,
)
from quantark.asset.equity.product.deltaone import Futures
from quantark.asset.equity.product.option import AsianOption, EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType

U = "TEST"
START = datetime(2026, 1, 5)


def _env(date=START):
    return PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name=U), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=date)


def _book(*products):
    env = _env()
    portfolio = Portfolio(portfolio_name="t", pricing_environments={U: env}, creation_date=START)
    ids = []
    for p in products:
        engine = DeltaOneEngine() if isinstance(p, Futures) else BlackScholesEngine()
        ids.append(portfolio.add_position(product=p, quantity=1.0, entry_price=1.0, underlying=U, engine=engine,
                                          entry_timestamp=START).position_id)
    return portfolio, env, ids


def test_whitelist_and_unrolled_detection():
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    dated = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=datetime(2027, 1, 5))
    fut = Futures(underlying=U, maturity=0.25)
    asian = AsianOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    assert EuropeanVanillaOption in FLOAT_ROLLABLE_PRODUCTS and Futures not in FLOAT_ROLLABLE_PRODUCTS
    assert is_float_rollable(call) and not is_float_rollable(dated) and not is_float_rollable(fut)
    assert not is_float_rollable(asian) and has_unrolled_float_maturity(asian)
    assert not has_unrolled_float_maturity(fut)          # static-maturity futures hedge is the documented design
    assert not has_unrolled_float_maturity(call)


def test_roller_rolls_from_first_sight_and_floors():
    r = FloatMaturityRoller()
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=10 / 365)
    r.register("p", call, START)
    r.register("p", EuropeanVanillaOption(strike=1.0, option_type=OptionType.CALL, maturity=5.0), START)  # ignored
    day3 = r.rolled("p", call, datetime(2026, 1, 8))
    assert day3 is not call and day3.maturity == pytest.approx(7 / 365, abs=1e-15)
    assert call.maturity == 10 / 365                                        # input not mutated
    assert r.rolled("p", call, datetime(2026, 1, 15)).maturity == MATURITY_FLOOR
    assert r.rolled("p", call, datetime(2026, 2, 15)).maturity == MATURITY_FLOOR   # held at the floor
    r.retain(set())
    with pytest.raises(KeyError):
        r.rolled("p", call, START)


def test_manager_rolls_untracked_vanilla_but_not_futures_and_warns_once_for_schedule_products():
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    fut = Futures(underlying=U, maturity=0.25)
    asian = AsianOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    portfolio, env, (pid_call, pid_fut, pid_asian) = _book(call, fut, asian)
    manager = PortfolioLifecycleManager(base_date=START)
    manager.register_positions(portfolio)                                   # nothing trackable here
    assert manager.num_tracked == 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        manager.process_day(portfolio, day_index=0, day_date=START)
        day10 = datetime(2026, 1, 15)
        env.valuation_date = day10
        before = manager.pricing_products(portfolio, day10)
        manager.process_day(portfolio, day_index=10, day_date=day10)
    assert portfolio.positions[pid_call].product.maturity == pytest.approx(1.0 - 10 / 365, abs=1e-15)
    assert before[pid_call].maturity == portfolio.positions[pid_call].product.maturity
    assert before[pid_call] is not portfolio.positions[pid_call].product   # fresh object per call
    assert portfolio.positions[pid_fut].product is fut and fut.maturity == 0.25
    assert portfolio.positions[pid_asian].product is asian and asian.maturity == 1.0
    msgs = [str(x.message) for x in w if "no roll rule" in str(x.message)]
    assert len(msgs) == 1 and "AsianOption" in msgs[0]


def test_ids_that_leave_the_book_are_forgotten():
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    portfolio, env, (pid,) = _book(call)
    manager = PortfolioLifecycleManager(base_date=START)
    manager.process_day(portfolio, 0, START)
    portfolio.remove_position(pid)
    manager.process_day(portfolio, 1, datetime(2026, 1, 6))
    assert pid not in manager._float_roller._base


def test_the_roll_anchor_does_not_depend_on_the_lifecycle_flag():
    """Both backtest modes anchor initial positions at start_date (Kimi review 2026-09-03).

    register_positions runs only when the engine handles lifecycle events; without the
    unconditional register_float_rolls the same book would otherwise anchor at the first
    market date instead, so the flag would silently change every day's theta.
    """
    late = datetime(2026, 1, 15)                       # first bar, 10 days after start_date
    maturities = []
    for handle_lifecycle in (True, False):
        call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
        portfolio, env, (pid,) = _book(call)
        manager = PortfolioLifecycleManager(base_date=START)
        manager.register_float_rolls(portfolio)        # the engine does this in BOTH modes
        if handle_lifecycle:
            manager.register_positions(portfolio)
        env.valuation_date = late
        manager.process_day(portfolio, day_index=10, day_date=late)
        maturities.append(portfolio.positions[pid].product.maturity)
    assert maturities[0] == pytest.approx(1.0 - 10 / 365, abs=1e-15)
    assert maturities[0] == maturities[1]


def test_an_untracked_snowball_warns_even_when_registration_never_ran():
    """The KO-reset snowball's specific warning comes from registration, which the flag can skip.

    With handle_lifecycle_events off, register_positions never runs, so excluding the snowball
    from the generic "no roll rule" warning left it silently repriced at a constant maturity
    (Kimi review 2026-09-03).
    """
    from quantark.asset.equity.lifecycle.float_roll import has_unrolled_float_maturity
    from quantark.asset.equity.product.option.ko_reset_snowball_option import (
        KnockOutResetSnowballOption,
        PostKOScheduleMode,
    )
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.util.enum import ObservationType

    def _cfg(ko):
        return BarrierConfig(ko_barrier=ko, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                             ko_observation_dates=[0.25, 0.5, 0.75, 1.0])

    snowball = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, barrier_config=_cfg(105.0), post_barrier_config=_cfg(98.0),
        contract_multiplier=1.0, maturity=1.0, is_reverse=False,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE)

    assert has_unrolled_float_maturity(snowball)      # no longer excluded from the generic warning
    portfolio, env, (pid,) = _book(snowball)
    manager = PortfolioLifecycleManager(base_date=START)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        manager.process_day(portfolio, day_index=0, day_date=START)
        manager.process_day(portfolio, day_index=1, day_date=datetime(2026, 1, 6))
    msgs = [str(x.message) for x in w if "no roll rule" in str(x.message)]
    assert len(msgs) == 1                             # warned once, from the roll path
    assert portfolio.positions[pid].product.maturity == 1.0    # and still repriced unrolled
