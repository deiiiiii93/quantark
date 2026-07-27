import math
import pytest
from datetime import datetime, timedelta
from copy import deepcopy

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import BumpConfig, EngineParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import (
    BarrierConfig,
    BarrierOption,
    CouponBarrierConfig,
    EuropeanVanillaOption,
    PhoenixOption,
    SnowballOption,
)
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param.div import ContinuousDividendYield
from quantark.param.quote.spot_quote import SpotQuote
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import (
    CalendarType,
    DayCountConvention,
    create_calendar,
)
from quantark.util.enum import (
    BarrierType,
    CouponPayType,
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
            self.observation_dates_seen.append(
                getattr(product, "observation_dates", None)
            )
            return 1.0

    engine = DummyEngine()
    theta = calc.calculate_numerical_theta(option, env, engine, base_price=1.0)

    assert theta == pytest.approx(0.0)
    assert engine.observation_dates_seen[-1] == [0.5]


def _business_day_pricing_env():
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 6, 26),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(0.2),
        div_yield=ContinuousDividendYield(0.01),
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        bus_days_in_year=244,
        calendar=create_calendar(CalendarType.CHINA_SSE),
    )


def test_theta_auto_business_day_bump_advances_friday_to_monday():
    env = _business_day_pricing_env()
    calc = GreeksCalculator()

    bumped_date, time_bump, mode = calc._advance_theta_bump(env, 1, "auto")

    assert mode == "business_days"
    assert bumped_date == datetime(2026, 6, 29)
    assert time_bump == pytest.approx(1 / 244)


def test_vanilla_theta_auto_business_day_mode_avoids_weekend_zero():
    env = _business_day_pricing_env()
    option = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    engine = BlackScholesEngine()

    business_calc = GreeksCalculator()
    business_theta = business_calc.calculate_numerical_theta(option, env, engine)

    calendar_calc = GreeksCalculator(
        params=EngineParams(bump_config=BumpConfig(time_bump_mode="calendar_days"))
    )
    calendar_theta = calendar_calc.calculate_numerical_theta(option, env, engine)

    assert business_theta != pytest.approx(0.0)
    assert calendar_theta == pytest.approx(0.0)


class _CapturingMaturityEngine:
    def __init__(self):
        self.calls = []

    def price(self, product, pricing_env):
        self.calls.append((deepcopy(product), deepcopy(pricing_env)))
        return product.get_maturity(pricing_env)


def _dated_schedule(first_date, second_date, first_time, second_time):
    return ObservationSchedule(
        records=[
            ObservationRecord(
                observation_time=first_time,
                observation_date=first_date,
                barrier=110.0,
                payoff=0.01,
            ),
            ObservationRecord(
                observation_time=second_time,
                observation_date=second_date,
                barrier=110.0,
                payoff=0.01,
            ),
        ],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        frequency=None,
    )


def _scheduled_barrier_config():
    monday = datetime(2026, 6, 29)
    wednesday = datetime(2026, 7, 1)
    return BarrierConfig(
        ko_barrier=110.0,
        ko_rate=0.01,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_schedule=_dated_schedule(monday, wednesday, 1 / 244, 3 / 244),
        ki_barrier=80.0,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_schedule=_dated_schedule(monday, wednesday, 1 / 244, 3 / 244),
    )


def test_snowball_theta_business_day_shift_updates_state_and_drops_past_records():
    env = _business_day_pricing_env()
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=_scheduled_barrier_config(),
        maturity=1.0,
    )
    engine = _CapturingMaturityEngine()
    calc = GreeksCalculator()

    theta = calc.calculate_numerical_theta(product, env, engine)

    shifted_product, shifted_env = engine.calls[-1]
    assert theta == pytest.approx(-1 / 244)
    assert shifted_env.valuation_date == datetime(2026, 6, 29)
    assert shifted_product.maturity == pytest.approx(1.0 - 1 / 244)
    assert len(shifted_product.barrier_config.ko_observation_schedule.records) == 1
    assert len(shifted_product.barrier_config.ki_observation_schedule.records) == 1
    assert shifted_product.barrier_config.ko_observation_schedule.records[
        0
    ].observation_time == pytest.approx(2 / 244)
    assert shifted_product.barrier_config.ki_observation_schedule.records[
        0
    ].observation_time == pytest.approx(2 / 244)


def test_phoenix_theta_business_day_shift_updates_state_and_drops_past_records():
    env = _business_day_pricing_env()
    product = PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=_scheduled_barrier_config(),
        coupon_config=CouponBarrierConfig(coupon_barrier=85.0, coupon_rate=0.01),
        maturity=1.0,
    )
    engine = _CapturingMaturityEngine()
    calc = GreeksCalculator()

    theta = calc.calculate_numerical_theta(product, env, engine)

    shifted_product, shifted_env = engine.calls[-1]
    assert theta == pytest.approx(-1 / 244)
    assert shifted_env.valuation_date == datetime(2026, 6, 29)
    assert shifted_product.maturity == pytest.approx(1.0 - 1 / 244)
    assert len(shifted_product.barrier_config.ko_observation_schedule.records) == 1
    assert len(shifted_product.barrier_config.ki_observation_schedule.records) == 1
    assert shifted_product.barrier_config.ko_observation_schedule.records[
        0
    ].observation_time == pytest.approx(2 / 244)
    assert shifted_product.barrier_config.ki_observation_schedule.records[
        0
    ].observation_time == pytest.approx(2 / 244)


def _daily_ki_times():
    return [day / 244 for day in range(1, 245)]


def _engine_snowball_product():
    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=75.0,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=_daily_ki_times(),
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def _engine_phoenix_pde_product():
    ko_times = [0.25, 0.5, 0.75, 1.0]
    ki_times = _daily_ki_times()
    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=ko_times,
        ko_observation_schedule=ObservationSchedule(
            records=[
                ObservationRecord(observation_time=t, barrier=103.0, payoff=0.01)
                for t in ko_times
            ],
            aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        ),
        ki_barrier=75.0,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=ki_times,
        ki_observation_schedule=ObservationSchedule(
            records=[
                ObservationRecord(observation_time=t, barrier=75.0, payoff=0.0)
                for t in ki_times
            ],
            aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        ),
    )
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        coupon_config=CouponBarrierConfig(coupon_barrier=85.0, coupon_rate=0.01),
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def _engine_phoenix_quad_product():
    barrier_config = BarrierConfig(
        ko_barrier=1.0e9,
        ko_rate=0.0,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.5, 1.0],
        ki_barrier=None,
    )
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        coupon_config=CouponBarrierConfig(
            coupon_barrier=80.0,
            coupon_rate=0.02,
            coupon_pay_type=CouponPayType.INSTANT,
        ),
        contract_multiplier=1.0,
        maturity=1.0,
    )


def _assert_business_day_theta_reprices_engine(product, engine):
    env = _business_day_pricing_env()
    base_price = engine.price(product, env)
    theta = GreeksCalculator().calculate_numerical_theta(
        product, env, engine, base_price=base_price
    )

    assert math.isfinite(base_price)
    assert math.isfinite(theta)
    assert theta != pytest.approx(0.0)


def test_snowball_pde_theta_reprices_after_business_day_shift():
    _assert_business_day_theta_reprices_engine(
        _engine_snowball_product(),
        PDEEngine(params=PDEParams()),
    )


def test_snowball_quad_theta_reprices_after_business_day_shift():
    _assert_business_day_theta_reprices_engine(
        _engine_snowball_product(),
        SnowballQuadEngine(params=QuadParams(grid_points=151)),
    )


def test_phoenix_pde_theta_reprices_after_business_day_shift():
    _assert_business_day_theta_reprices_engine(
        _engine_phoenix_pde_product(),
        PDEEngine(params=PDEParams()),
    )


def test_phoenix_quad_theta_reprices_after_business_day_shift():
    _assert_business_day_theta_reprices_engine(
        _engine_phoenix_quad_product(),
        PhoenixQuadEngine(params=QuadParams(grid_points=201)),
    )
