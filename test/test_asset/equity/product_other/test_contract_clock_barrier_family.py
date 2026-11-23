"""Contract clock wired into the barrier family (T4): barrier, double barrier, one-touch, double one-touch, single and
double sharkfin, accumulator, range accrual. Per product: a 1-ulp cross-source pair is canonical after construction,
a collision-free booking keeps its schedule object, and time_shift keeps the contract canonical (spec
.ship/2026-11-23-contract-clock AC2, AC2f).
"""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.product.option.accumulator_option import AccumulatorOption
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.contract_clock import canonicalize_contract_clock
from quantark.asset.equity.product.option.double_barrier_option import DoubleBarrierOption
from quantark.asset.equity.product.option.double_one_touch_option import DoubleOneTouchOption
from quantark.asset.equity.product.option.double_sharkfin_option import DoubleSharkfinOption
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.asset.equity.product.option.range_accrual_config import RangeAccrualConfig, RangeAccrualObservationRecord
from quantark.asset.equity.product.option.range_accrual_option import RangeAccrualOption
from quantark.asset.equity.product.option.single_sharkfin_option import SingleSharkfinOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierDirection, BarrierType, ObservationType, OptionType
from quantark.util.enum.option_enums import DoubleBarrierType

DOWN = lambda t: float(np.nextafter(t, -np.inf))   # noqa: E731
OBS = [0.25, 0.5, 0.75, 1.0]
OBS_ULP = [0.25, 0.5, 0.75, DOWN(1.0)]             # the last fixing one ulp before the maturity


def env():
    return PricingEnvironment(
        spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2), rate_curve=FlatRateCurve(0.03),
        div_yield=ContinuousDividendYield(0.01), valuation_date=datetime(2026, 9, 11),
    )


def barrier(obs):
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                         maturity=1.0, observation_type=ObservationType.DISCRETE, observation_dates=list(obs))


def double_barrier(obs):
    return DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=120.0, lower_barrier=80.0,
                               barrier_type=DoubleBarrierType.KNOCK_OUT, maturity=1.0,
                               observation_type=ObservationType.DISCRETE, observation_dates=list(obs))


def one_touch(obs):
    return OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, maturity=1.0,
                          observation_type=ObservationType.DISCRETE, observation_dates=list(obs))


def double_one_touch(obs):
    return DoubleOneTouchOption(upper_barrier=120.0, lower_barrier=80.0, maturity=1.0,
                                observation_type=ObservationType.DISCRETE, observation_dates=list(obs))


def single_sharkfin(obs):
    return SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, maturity=1.0,
                                observation_type=ObservationType.DISCRETE, observation_dates=list(obs))


def double_sharkfin(obs):
    return DoubleSharkfinOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=120.0, lower_barrier=80.0,
                                maturity=1.0, observation_type=ObservationType.DISCRETE,
                                observation_dates=list(obs))


def accumulator(obs):
    return AccumulatorOption(strike=100.0, knock_out_barrier=120.0, maturity=1.0, initial_price=100.0,
                             daily_share_accumulation=1.0, observation_dates=list(obs))


def range_config():
    return RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0)


def range_accrual_records(obs):
    return RangeAccrualOption(initial_price=100.0, range_config=range_config(), maturity=1.0,
                              observation_records=[RangeAccrualObservationRecord(observation_time=t) for t in obs])


def range_accrual_times(obs):
    return RangeAccrualOption(initial_price=100.0, range_config=range_config(), maturity=1.0,
                              observation_times=list(obs))


ALL = {
    "barrier": barrier, "double_barrier": double_barrier, "one_touch": one_touch,
    "double_one_touch": double_one_touch, "single_sharkfin": single_sharkfin, "double_sharkfin": double_sharkfin,
    "accumulator": accumulator, "range_accrual_records": range_accrual_records,
    "range_accrual_times": range_accrual_times,
}


def held_schedule(p):
    """The object that carries the product's observation times."""
    schedule = getattr(p, "observation_schedule", None)
    if schedule is not None:
        return schedule
    return p.observation_records if p.observation_records is not None else p.observation_times


def schedule_times(p):
    held = held_schedule(p)
    if hasattr(held, "records"):
        return [r.observation_time for r in held.records]
    return [r.observation_time if hasattr(r, "observation_time") else r for r in held]


@pytest.mark.parametrize("name", sorted(ALL))
class TestBarrierFamily:
    def test_ulp_twin_is_canonical_after_construction(self, name):
        p = ALL[name](OBS_ULP)
        assert schedule_times(p)[-1] == 1.0 == p.maturity
        assert canonicalize_contract_clock(p) is False

    def test_ulp_twin_equals_the_clean_booking(self, name):
        assert schedule_times(ALL[name](OBS_ULP)) == schedule_times(ALL[name](OBS))

    def test_collision_free_booking_keeps_its_schedule_object(self, name):
        p = ALL[name](OBS)
        held = held_schedule(p)
        assert canonicalize_contract_clock(p) is False
        assert held_schedule(p) is held

    def test_time_shift_keeps_it_canonical(self, name):
        p = ALL[name](OBS)
        p.time_shift(0.1, datetime(2026, 9, 11), env())
        assert canonicalize_contract_clock(p) is False

    def test_time_shift_is_a_canonical_clock_operation(self, name):
        assert hasattr(type(ALL[name](OBS)).time_shift, "__wrapped__")


class TestRangeAccrualRecords:
    def test_two_distinct_records_of_one_instant_are_refused(self):
        from quantark.util.exceptions import ValidationError

        with pytest.raises(ValidationError, match="observation_records"):
            range_accrual_records([0.25, 0.5, DOWN(1.0), 1.0])

    def test_a_record_ulp_before_a_later_record_time_is_the_input_not_mutated(self):
        records = [RangeAccrualObservationRecord(observation_time=t) for t in OBS_ULP]
        held = records[-1]
        p = RangeAccrualOption(initial_price=100.0, range_config=range_config(), maturity=1.0,
                               observation_records=records)
        assert p.observation_records[-1].observation_time == 1.0
        assert held.observation_time == OBS_ULP[-1]
