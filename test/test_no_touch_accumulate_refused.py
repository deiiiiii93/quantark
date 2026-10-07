"""ACCUMULATE has no meaning for a no-touch (user ruling 2026-10-07, 3C handoff item 8).

A no-touch pays only if its barrier is never hit up to and including expiry: nothing accumulates. The same
holds for a sharkfin's no-hit leg, so both sharkfins refuse the mode as well. The refusal is on the product,
so every engine refuses alike; one-touches and barrier knock-outs keep the mode.
"""
import pytest

from quantark.asset.equity.product.option import (
    DoubleOneTouchOption,
    DoubleSharkfinOption,
    OneTouchOption,
    SingleSharkfinOption,
)
from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
from quantark.util.enum import BarrierDirection, ObservationAggregation, ObservationType, OptionType, TouchType
from quantark.util.exceptions import ValidationError

TIMES = [0.25, 0.5, 0.75, 1.0]
ACC, FIRST = ObservationAggregation.ACCUMULATE, ObservationAggregation.STOP_FIRST_HIT


def single(mode, barrier=130.0, payoff=5.0):
    return ObservationSchedule.from_legacy(observation_dates=TIMES, default_barrier=barrier,
                                           default_payoff=payoff, aggregation_mode=mode)


def double(mode, payoff=5.0):
    return ObservationSchedule.from_legacy(observation_dates=TIMES, default_barrier=None, default_payoff=payoff,
                                           aggregation_mode=mode, upper_barrier=130.0, lower_barrier=80.0)


def touch(touch_type, mode):
    return OneTouchOption(barrier=130.0, barrier_direction=BarrierDirection.UP, maturity=1.0, rebate=5.0,
                          touch_type=touch_type, observation_type=ObservationType.DISCRETE,
                          observation_schedule=single(mode))


def double_touch(touch_type, mode):
    return DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=80.0, maturity=1.0, rebate=5.0,
                                touch_type=touch_type, observation_type=ObservationType.DISCRETE,
                                observation_schedule=double(mode))


def single_sharkfin(mode):
    return SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, maturity=1.0,
                                participation_rate=0.8, knock_out_rebate=5.0, no_hit_rebate=3.0,
                                observation_type=ObservationType.DISCRETE, observation_schedule=single(mode))


def double_sharkfin(mode):
    return DoubleSharkfinOption(strike=100.0, option_type=OptionType.CALL, lower_barrier=80.0,
                                upper_barrier=130.0, maturity=1.0, participation_rate=0.8, knock_out_rebate=5.0,
                                no_hit_rebate=3.0, observation_type=ObservationType.DISCRETE,
                                observation_schedule=double(mode))


@pytest.mark.parametrize("make", [
    lambda mode: touch(TouchType.NO_TOUCH, mode),
    lambda mode: double_touch(TouchType.DOUBLE_NO_TOUCH, mode),
    single_sharkfin,
    double_sharkfin,
], ids=["no_touch", "double_no_touch", "single_sharkfin", "double_sharkfin"])
def test_accumulate_is_refused(make):
    with pytest.raises(ValidationError, match="never hit up to and including expiry"):
        make(ACC)
    make(FIRST)                                   # first-hit monitoring is unchanged


@pytest.mark.parametrize("make", [
    lambda mode: touch(TouchType.ONE_TOUCH, mode),
    lambda mode: double_touch(TouchType.DOUBLE_ONE_TOUCH, mode),
], ids=["one_touch", "double_one_touch"])
def test_one_touches_keep_accumulate(make):
    assert make(ACC).observation_schedule.aggregation_mode is ACC
