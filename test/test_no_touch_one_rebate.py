"""A no-touch has one rebate, on every route (user decision 2026-10-15).

A no-touch (``OneTouchOption`` NO_TOUCH, ``DoubleOneTouchOption`` DOUBLE_NO_TOUCH) is paid only when nothing touches,
so, like a knock-in (``refuse_knock_in_record_payoff``, user ruling 2026-10-10), no record's payoff can apply. A
schedule record paying anything other than ``product.rebate`` is refused (``ValidationError``) on the analytical,
QUAD and PDE routes; a record without a payoff, or one paying the rebate, is the same contract. One-touches (a
record's payoff is the touch cash) are not concerned.

Before, QUAD's no-touch (rebate x DF less a one-touch paying each RECORD's payoff, floored at 0) disagreed with the
analytical and PDE routes on such a schedule: records paying 3 against a rebate of 1 gave 0.14 on QUAD where the
analytical engine and the PDE gave 0.69. The single sharkfin's no-touch leg (the no-hit cash) now carries records
paying that cash, as its knock-out leg carries records paying 0; its value is unchanged to the bit.
"""
from __future__ import annotations

import warnings

import pytest

import test_barrier_lifecycle_discrete_observation as LC
import test_discrete_barrier_between_observations as T
from quantark.asset.equity.lifecycle import BarrierLifecycleTracker, LifecycleEventType
from quantark.asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine
from quantark.asset.equity.engine.analytical.single_sharkfin_option_analytical_engine import (
    SingleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.barrier_schedule_rules import refuse_no_touch_record_payoff
from quantark.asset.equity.engine.pde.double_one_touch_pde_solver import DoubleOneTouchPDESolver
from quantark.asset.equity.engine.pde.one_touch_pde_solver import OneTouchPDESolver
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.engine.quad import DiscreteQuadEngine
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product.option import DoubleOneTouchOption, OneTouchOption, SingleSharkfinOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.enum import BarrierDirection, ObservationType, OptionType, TouchType
from quantark.util.exceptions import ValidationError

D = dict(maturity=1.0, observation_type=ObservationType.DISCRETE)
SPOT = 100.0


def _single(payoff, level=120.0):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level, payoff=payoff)
                                        for t in T.TIMES])


def _double(payoff):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, upper_barrier=130.0, lower_barrier=75.0,
                                                          payoff=payoff) for t in T.TIMES])


def no_touch(payoff, rebate=1.0, touch_type=TouchType.NO_TOUCH):
    return OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, rebate=rebate, touch_type=touch_type,
                          observation_schedule=_single(payoff), **D)


def double_no_touch(payoff, rebate=1.0, touch_type=TouchType.DOUBLE_NO_TOUCH):
    return DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=75.0, rebate=rebate, payment_at_hit=False,
                                touch_type=touch_type, observation_schedule=_double(payoff), **D)


ROUTES = {
    "analytical": lambda p: OneTouchAnalyticalEngine().price(p, T.env(SPOT)),
    "quad": lambda p: DiscreteQuadEngine(params=QuadParams(grid_points=401)).price(p, T.env(SPOT)),
    "pde_solver": lambda p: OneTouchPDESolver(PDEParams()).price(p, T.env(SPOT)),
    "pde_engine": lambda p: PDEEngine(PDEParams()).price(p, T.env(SPOT)),
}
DOUBLE_ROUTES = {
    "quad": ROUTES["quad"],
    "pde_solver": lambda p: DoubleOneTouchPDESolver(PDEParams()).price(p, T.env(SPOT)),
    "pde_engine": ROUTES["pde_engine"],
}


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(fn())


# ======================================================================================================================
# the rule
# ======================================================================================================================

def test_the_rule_refuses_a_record_paying_other_than_the_rebate():
    with pytest.raises(ValidationError, match="no-touch has one rebate"):
        refuse_no_touch_record_payoff(no_touch(3.0))
    with pytest.raises(ValidationError, match="no-touch has one rebate"):
        refuse_no_touch_record_payoff(double_no_touch(3.0))
    for product in (no_touch(None), no_touch(1.0), double_no_touch(None), double_no_touch(1.0),
                    no_touch(3.0, touch_type=TouchType.ONE_TOUCH),
                    double_no_touch(3.0, touch_type=TouchType.DOUBLE_ONE_TOUCH)):
        refuse_no_touch_record_payoff(product)       # the same contract, or not a no-touch


# ======================================================================================================================
# every route
# ======================================================================================================================

@pytest.mark.parametrize("route", list(ROUTES))
def test_a_no_touch_with_records_paying_other_cash_is_refused(route):
    with pytest.raises(ValidationError, match="no-touch has one rebate"):
        _quiet(lambda: ROUTES[route](no_touch(3.0)))


@pytest.mark.parametrize("route", list(DOUBLE_ROUTES))
def test_a_double_no_touch_with_records_paying_other_cash_is_refused(route):
    with pytest.raises(ValidationError, match="no-touch has one rebate"):
        _quiet(lambda: DOUBLE_ROUTES[route](double_no_touch(3.0)))


@pytest.mark.parametrize("route", list(ROUTES))
def test_records_paying_the_rebate_are_the_same_contract(route):
    assert _quiet(lambda: ROUTES[route](no_touch(1.0))) == _quiet(lambda: ROUTES[route](no_touch(None)))


@pytest.mark.parametrize("route", list(DOUBLE_ROUTES))
def test_double_records_paying_the_rebate_are_the_same_contract(route):
    assert (_quiet(lambda: DOUBLE_ROUTES[route](double_no_touch(1.0)))
            == _quiet(lambda: DOUBLE_ROUTES[route](double_no_touch(None))))


@pytest.mark.parametrize("route", list(ROUTES))
def test_a_one_touch_still_pays_its_records(route):
    assert _quiet(lambda: ROUTES[route](no_touch(3.0, touch_type=TouchType.ONE_TOUCH))) > 0.0


# ======================================================================================================================
# the single sharkfin's no-hit leg
# ======================================================================================================================

def _sharkfin(payoff):
    return SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, knock_out_rebate=0.5,
                                no_hit_rebate=3.0, pay_at_hit=True, observation_schedule=_single(payoff), **D)


@pytest.mark.parametrize("spot", [100.0, 119.0, 121.0])
def test_a_sharkfin_whose_records_pay_the_knock_out_cash_prices_as_before(spot):
    """Its records pay the knock-out cash (0.5), not the no-hit cash (3) the no-touch leg pays: the leg carries
    records paying 3, the same contract as records without a payoff."""
    explicit = _quiet(lambda: SingleSharkfinOptionAnalyticalEngine().price(_sharkfin(0.5), T.env(spot)))
    implicit = _quiet(lambda: SingleSharkfinOptionAnalyticalEngine().price(_sharkfin(None), T.env(spot)))
    assert explicit == implicit


# ======================================================================================================================
# the run lifecycle (BarrierLifecycleTracker: the equity backtest and the dynamic scenario)
# ======================================================================================================================

def _lifecycle_no_touch(payoff, double=False):
    if double:
        return DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=75.0, rebate=1.0, exercise_date=LC.EXERCISE,
                                    payment_at_hit=False, touch_type=TouchType.DOUBLE_NO_TOUCH,
                                    observation_type=ObservationType.DISCRETE,
                                    observation_schedule=LC._records(upper=130.0, lower=75.0, payoff=payoff))
    return OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, rebate=1.0, exercise_date=LC.EXERCISE,
                          payment_at_hit=False, touch_type=TouchType.NO_TOUCH, observation_type=ObservationType.DISCRETE,
                          observation_schedule=LC._records(120.0, payoff=payoff))


@pytest.mark.parametrize("double", [False, True], ids=["no_touch", "double_no_touch"])
def test_a_run_refuses_a_no_touch_whose_records_pay_other_cash(double):
    """The tracker is where a run takes the contract (``PortfolioLifecycleManager.register_positions``): refused
    there, whatever engine the position prices with."""
    with pytest.raises(ValidationError, match="no-touch has one rebate"):
        BarrierLifecycleTracker(product=_lifecycle_no_touch(3.0, double), quantity=2.0, start_date=LC.START)


@pytest.mark.parametrize("payoff", [None, 1.0], ids=["records_unset", "records_pay_the_rebate"])
@pytest.mark.parametrize("double", [False, True], ids=["no_touch", "double_no_touch"])
def test_a_touched_no_touch_books_no_cash(double, payoff):
    tracker = BarrierLifecycleTracker(product=_lifecycle_no_touch(payoff, double), quantity=2.0, start_date=LC.START)
    events = LC._run(tracker, [(0, 100.0), (30, 131.0)])
    assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT]
    assert events[0].payoff == 0.0 and events[0].cashflow == 0.0


@pytest.mark.parametrize("payoff", [None, 1.0], ids=["records_unset", "records_pay_the_rebate"])
@pytest.mark.parametrize("double", [False, True], ids=["no_touch", "double_no_touch"])
def test_a_no_touch_never_touched_books_its_rebate(double, payoff):
    tracker = BarrierLifecycleTracker(product=_lifecycle_no_touch(payoff, double), quantity=2.0, start_date=LC.START)
    events = LC._run(tracker, [(day, 100.0) for day in (0, 30, 60, 90, 180, 270, 359, 360)])
    assert [e.event_type for e in events] == [LifecycleEventType.EXPIRY]
    assert events[0].payoff == 1.0 and events[0].cashflow == 2.0
