import pytest
from datetime import datetime, timedelta
from copy import deepcopy

from asset.equity.product.option import BarrierOption
from asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from param.quote.spot_quote import SpotQuote
from param.rrf.rate_curve import FlatRateCurve
from param.vol.vol_surface import FlatVolSurface
from priceenv import PricingEnvironment
from util.enum import (
    BarrierType,
    ObservationAggregation,
    ObservationType,
    OptionType,
)


def _pricing_env():
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.01),
        valuation_date=datetime(2024, 1, 1),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(0.2),
    )


def test_theta_bump_drops_past_time_records():
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(observation_time=0.001, barrier=110.0, payoff=1.0),
            ObservationRecord(observation_time=0.25, barrier=110.0, payoff=1.0),
        ],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        frequency=None,
    )
    bumped = schedule.time_shift(time_bump=1 / 365, bumped_date=None)

    assert bumped is not None
    assert len(bumped.records) == 1
    assert bumped.records[0].observation_time == pytest.approx(0.25 - 1 / 365)


def test_theta_bump_drops_past_date_records():
    valuation = datetime(2024, 1, 1)
    rec1 = ObservationRecord(
        observation_date=valuation + timedelta(days=1), barrier=110.0, payoff=1.0
    )
    rec2 = ObservationRecord(
        observation_date=valuation + timedelta(days=5), barrier=110.0, payoff=1.0
    )
    schedule = ObservationSchedule(
        records=[rec1, rec2],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
    )

    bumped = schedule.time_shift(
        time_bump=1 / 365, bumped_date=valuation + timedelta(days=1)
    )

    assert bumped is not None
    assert len(bumped.records) == 1
    assert bumped.records[0].observation_date == rec2.observation_date


def test_theta_uses_filtered_schedule_in_pricing():
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(observation_time=0.001, barrier=110.0, payoff=1.0),
            ObservationRecord(observation_time=0.25, barrier=110.0, payoff=1.0),
        ],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        frequency=None,
    )

    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=deepcopy(schedule),
    )

    env = _pricing_env()
    calc = GreeksCalculator()

    class DummyEngine:
        def __init__(self):
            self.calls = []

        def price(self, product, pricing_env):
            self.calls.append(
                len(product.observation_schedule.records)
                if getattr(product, "observation_schedule", None)
                else 0
            )
            return float(self.calls[-1])

    engine = DummyEngine()
    greeks = calc.calculate_numerical_greeks(option, env, engine)

    assert greeks["price"] == pytest.approx(2.0)
    assert greeks["theta"] == pytest.approx(-1.0)
    # Base price uses full schedule; theta bump uses filtered schedule
    assert engine.calls[0] == 2
    assert 1 in engine.calls


def test_theta_all_records_dropped_still_returns_rho():
    schedule = ObservationSchedule(
        records=[ObservationRecord(observation_time=0.0001, barrier=110.0, payoff=1.0)],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        frequency=None,
    )

    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=deepcopy(schedule),
    )

    env = _pricing_env()
    calc = GreeksCalculator()

    class DummyEngine:
        def __init__(self):
            self.calls = []

        def price(self, product, pricing_env):
            self.calls.append(
                len(product.observation_schedule.records)
                if getattr(product, "observation_schedule", None)
                else 0
            )
            return float(len(self.calls))

    engine = DummyEngine()
    greeks = calc.calculate_numerical_greeks(option, env, engine)

    assert greeks["theta"] == pytest.approx(0.0)
    assert "rho" in greeks
    assert greeks["rho"] != 0.0
    # Theta path did not short-circuit rho computation
    assert len(engine.calls) >= 5


def test_theta_date_schedule_preserves_legacy_observation_dates():
    valuation = datetime(2024, 1, 1)
    rec = ObservationRecord(
        observation_date=valuation + timedelta(days=5), barrier=110.0, payoff=1.0
    )
    schedule = ObservationSchedule(
        records=[rec],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
    )

    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=deepcopy(schedule),
    )
    option.observation_dates = [0.5]  # Legacy field should remain unchanged

    env = _pricing_env()
    calc = GreeksCalculator()

    class DummyEngine:
        def __init__(self):
            self.observation_dates_seen = []

        def price(self, product, pricing_env):
            self.observation_dates_seen.append(getattr(product, "observation_dates", None))
            return 1.0

    engine = DummyEngine()
    theta = calc.calculate_numerical_theta(option, env, engine, base_price=1.0)

    assert theta == pytest.approx(0.0)
    assert engine.observation_dates_seen[-1] == [0.5]
