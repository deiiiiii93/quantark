"""Products' time_shift (spec 3.1, 3.3): always aged, landing kept by flag, an ended schedule accepted."""
import pytest

from pde_routes.cases import env_const
from quantark.asset.equity.product.option import (BarrierOption, DoubleBarrierOption, DoubleOneTouchOption,
                                                  DoubleSharkfinOption, OneTouchOption, SingleSharkfinOption)
from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
from quantark.util.enum import (BarrierDirection, BarrierType, DoubleBarrierType, ObservationType, OptionType,
                                TouchType)
from quantark.util.exceptions import ValidationError

D = 1 / 252
DISC = ObservationType.DISCRETE


def barrier(dates, T=1.0):
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                         maturity=T, rebate=5.0, observation_type=DISC, observation_dates=dates)


PRODUCTS = {
    "barrier": lambda d: barrier(d),
    "double_barrier": lambda d: DoubleBarrierOption(100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_OUT,
                                                    maturity=1.0, rebate=5.0, observation_type=DISC,
                                                    observation_dates=d),
    "one_touch": lambda d: OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, maturity=1.0,
                                          rebate=5.0, touch_type=TouchType.ONE_TOUCH, observation_type=DISC,
                                          observation_dates=d),
    "double_one_touch": lambda d: DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=80.0, maturity=1.0,
                                                       rebate=5.0, touch_type=TouchType.DOUBLE_NO_TOUCH,
                                                       observation_type=DISC, observation_dates=d),
    "single_sharkfin": lambda d: SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                                                      maturity=1.0, participation_rate=0.8, knock_out_rebate=5.0,
                                                      no_hit_rebate=3.0, observation_type=DISC, observation_dates=d),
    "double_sharkfin": lambda d: DoubleSharkfinOption(strike=100.0, option_type=OptionType.CALL, lower_barrier=80.0,
                                                      upper_barrier=130.0, maturity=1.0, participation_rate=0.8,
                                                      knock_out_rebate=5.0, no_hit_rebate=3.0, observation_type=DISC,
                                                      observation_dates=d),
}


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_shift_past_last_observation_ages_and_ends_monitoring(name):
    p = PRODUCTS[name]([D])
    assert p.time_shift(2 * D, None, env_const(), keep_landing=True) is None
    assert p.maturity == pytest.approx(1.0 - 2 * D)
    assert p.observation_schedule.records == [] and p.observation_schedule.monitoring_ended
    assert p.observation_dates == []
    p.validate()                                            # an aged product with ended monitoring is legal


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_landing_kept_for_theta_dropped_for_aging(name):
    p, q = PRODUCTS[name]([D, 2 * D]), PRODUCTS[name]([D, 2 * D])
    p.time_shift(D, None, env_const(), keep_landing=True)
    q.time_shift(D, None, env_const())
    assert p.observation_dates == [0.0, pytest.approx(D)]
    assert q.observation_dates == [pytest.approx(D)]


def test_hand_emptied_schedule_is_refused():
    with pytest.raises(ValidationError):
        BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                      maturity=1.0, observation_type=DISC, observation_schedule=ObservationSchedule(records=[]))


def test_autocallable_config_keeps_landing_by_flag():
    from quantark.asset.equity.product.option import SnowballOption
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig

    def snowball():
        return SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0,
                              barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                           ko_observation_dates=[D, 0.5, 1.0],
                                                           ki_barrier=75.0, ki_continuous=True))
    kept, aged = snowball(), snowball()
    assert kept.time_shift(D, None, env_const(), keep_landing=True) is None
    aged.time_shift(D, None, env_const())
    assert kept.barrier_config.ko_observation_dates[0] == 0.0
    assert len(kept.barrier_config.ko_observation_dates) == 3
    assert len(aged.barrier_config.ko_observation_dates) == 2
