import pytest
from datetime import datetime

from asset.equity.engine.pde_engine import PDEEngine
from asset.equity.product.option import BarrierOption
from asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from param.rrf.rate_curve import FlatRateCurve
from param.quote.spot_quote import SpotQuote
from param.vol.vol_surface import FlatVolSurface
from priceenv import PricingEnvironment
from util.enum import (
    BarrierType,
    ObservationAggregation,
    ObservationType,
    OptionType,
)
from util.exceptions import PricingError, ValidationError


def _pricing_env(spot: float = 100.0) -> PricingEnvironment:
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.01),
        valuation_date=datetime(2020, 1, 1),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(0.2),
    )


def test_observation_schedule_frequency_inference():
    schedule = ObservationSchedule.from_legacy(
        observation_dates=[0.25, 0.50, 0.75],
        default_barrier=100.0,
        default_payoff=1.0,
        frequency=None,
    )
    freq = schedule.ensure_regular_frequency(schedule.times)
    assert pytest.approx(0.25) == freq


def test_observation_schedule_analytical_ready_requires_fixed_payoff():
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(observation_time=0.25, barrier=100.0, payoff=1.0),
            ObservationRecord(observation_time=0.50, barrier=100.0, payoff=2.0),
        ],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
    )
    with pytest.raises(ValidationError):
        schedule.assert_analytical_ready(default_payoff=0.0)


def test_barrier_option_normalizes_schedule_defaults():
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_time=0.20,
                barrier=None,
                payoff=None,
            )
        ],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        frequency=0.20,
    )
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=105.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=2.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )
    option.validate()
    assert option.observation_type == ObservationType.DISCRETE
    assert option.observation_dates == [0.20]
    assert option.observation_schedule.records[0].barrier == pytest.approx(105.0)
    assert option.observation_schedule.records[0].payoff == pytest.approx(2.0)


def test_barrier_pde_engine_rejects_best_aggregation():
    schedule = ObservationSchedule(
        records=[ObservationRecord(observation_time=0.50, barrier=110.0, payoff=1.0)],
        aggregation_mode=ObservationAggregation.BEST,
        frequency=0.50,
    )
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=1.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )
    engine = PDEEngine()
    with pytest.raises(PricingError):
        engine.price(option, _pricing_env())


def test_barrier_pde_engine_supports_accumulate_schedule():
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(observation_time=0.25, barrier=110.0, payoff=0.5),
            ObservationRecord(observation_time=0.50, barrier=110.0, payoff=0.5),
        ],
        aggregation_mode=ObservationAggregation.ACCUMULATE,
        frequency=0.25,
    )
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=1.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )
    engine = PDEEngine()
    price = engine.price(option, _pricing_env())
    assert isinstance(price, float)

