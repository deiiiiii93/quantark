"""The record-level helpers reproduce the march's own builders (spec 3C decision 25: one object, wherever applied)."""
import numpy as np
import pytest

from pde_routes.cases import env_const, schedule
from quantark.asset.equity.engine.pde import (BarrierPDESolver, DoubleBarrierPDESolver, DoubleOneTouchPDESolver,
                                              EuropeanPDESolver, OneTouchPDESolver)
from quantark.asset.equity.product.option import BarrierOption, DoubleBarrierOption, EuropeanVanillaOption
from quantark.asset.equity.product.option.double_one_touch_option import DoubleOneTouchOption
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.util.enum import (BarrierDirection, BarrierType, DoubleBarrierType, ObservationAggregation,
                                ObservationType, OptionType, TouchType)

S = np.array([70.0, 99.0, 100.0, 101.0, 135.0])


@pytest.mark.parametrize("call", [True, False])
def test_barrier_terminal_payoff_and_slope(call):
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL if call else OptionType.PUT, barrier=130.0,
                      barrier_type=BarrierType.UP_OUT, maturity=1.0, participation_rate=0.5,
                      observation_type=ObservationType.CONTINUOUS)
    v, d = BarrierPDESolver()._terminal_payoff(p, env_const(), S)
    want = 0.5 * (np.maximum(S - 100.0, 0.0) if call else np.maximum(100.0 - S, 0.0))
    np.testing.assert_allclose(v, want, rtol=0, atol=1e-15)
    slope = np.where(S > 100.0, 0.5, 0.0) if call else np.where(S < 100.0, -0.5, 0.0)
    np.testing.assert_array_equal(d, slope)                      # zero at the strike, as the expired path has it


def test_step_event_is_event_for_records():
    for mode in (ObservationAggregation.STOP_FIRST_HIT, ObservationAggregation.ACCUMULATE):
        p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_OUT,
                          maturity=1.0, rebate=5.0, observation_type=ObservationType.DISCRETE,
                          observation_schedule=schedule([0.25, 0.5, 1.0], mode))
        solver, env = BarrierPDESolver(), env_const()
        t = np.array([0.0, 0.25, 0.5, 1.0])
        solver._setup_observation_indices(p, env, 1.0, t, solver._observation_resolve_kwargs(p))
        for idx in (1, 2):
            a = solver._step_event(p, env, idx, float(t[idx]))
            b = solver._event_for_records(p, env, solver._schedule_records.get(idx), float(t[idx]))
            assert a == b


@pytest.mark.parametrize("cls,product", [
    (DoubleBarrierPDESolver, lambda: DoubleBarrierOption(100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_OUT,
                                                         maturity=1.0, rebate=5.0)),
    (OneTouchPDESolver, lambda: OneTouchOption(barrier=130.0, barrier_direction=BarrierDirection.UP, maturity=1.0,
                                               rebate=5.0, payment_at_hit=False, touch_type=TouchType.NO_TOUCH)),
    (DoubleOneTouchPDESolver, lambda: DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=80.0, maturity=1.0,
                                                           rebate=5.0, payment_at_hit=False,
                                                           touch_type=TouchType.DOUBLE_NO_TOUCH)),
])
def test_default_record_from_empty_records(cls, product):
    solver, env = cls(), env_const()
    assert solver._event_for_records(product(), env, [], 0.5) == solver._event_for_records(product(), env, None, 0.5)


def test_european_terminal_payoff():
    p = EuropeanVanillaOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)
    v, d = EuropeanPDESolver()._terminal_payoff(p, env_const(), S)
    np.testing.assert_allclose(v, np.maximum(100.0 - S, 0.0), atol=1e-15, rtol=0)
    np.testing.assert_array_equal(d, np.where(S < 100.0, -1.0, 0.0))
