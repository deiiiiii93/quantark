"""Settlement timing for terminal payoffs and scheduled accumulator legs."""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical import (
    AccumulatorAnalyticalEngine,
    AsianOptionAnalyticalEngine,
    DigitalOptionAnalyticalEngine,
    RangeAccrualAnalyticalEngine,
)
from quantark.asset.equity.engine.mc import (
    AccumulatorMCEngine,
    AsianOptionMCEngine,
    DigitalOptionMCEngine,
    RangeAccrualMCEngine,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import (
    AccumulatorOption,
    AsianOption,
    CashOrNothingDigitalOption,
    ObservationRecord,
    ObservationSchedule,
    RangeAccrualConfig,
    RangeAccrualOption,
)
from quantark.asset.equity.settlement import (
    SettlementConvention,
    SettlementLagUnit,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    AccumulatorKnockOutType,
    AveragingType,
    ObservationAggregation,
    OptionType,
)


MATURITY = 0.5
LAG = 3.0 / 365.0


@pytest.fixture
def env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=101.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.04),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 1, 1),
    )


def _delayed():
    return SettlementConvention(
        lag=LAG,
        lag_unit=SettlementLagUnit.YEAR_FRACTION,
    )


@pytest.mark.parametrize(
    ("engine", "make_product"),
    [
        (
            DigitalOptionAnalyticalEngine(),
            lambda convention: CashOrNothingDigitalOption(
                strike=100.0,
                payout=10.0,
                option_type=OptionType.CALL,
                maturity=MATURITY,
                settlement_convention=convention,
            ),
        ),
        (
            DigitalOptionMCEngine(
                params=MCParams(
                    num_paths=4096,
                    time_steps=8,
                    seed=31,
                    use_antithetic=True,
                )
            ),
            lambda convention: CashOrNothingDigitalOption(
                strike=100.0,
                payout=10.0,
                option_type=OptionType.CALL,
                maturity=MATURITY,
                settlement_convention=convention,
            ),
        ),
        (
            AsianOptionAnalyticalEngine(),
            lambda convention: AsianOption(
                strike=100.0,
                option_type=OptionType.CALL,
                averaging_type=AveragingType.GEOMETRIC,
                maturity=MATURITY,
                settlement_convention=convention,
            ),
        ),
        (
            AsianOptionMCEngine(
                params=MCParams(
                    num_paths=4096,
                    time_steps=8,
                    seed=37,
                    use_antithetic=True,
                )
            ),
            lambda convention: AsianOption(
                strike=100.0,
                option_type=OptionType.CALL,
                averaging_type=AveragingType.GEOMETRIC,
                maturity=MATURITY,
                observation_times=[0.125, 0.25, 0.375, 0.5],
                settlement_convention=convention,
            ),
        ),
    ],
)
def test_digital_and_asian_terminal_payoffs_use_payment_df(
    engine, make_product, env
):
    immediate = engine.price(make_product(None), env)
    delayed = engine.price(make_product(_delayed()), env)
    ratio = (
        env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(MATURITY)
    )

    assert delayed == pytest.approx(immediate * ratio, rel=2.0e-12)


@pytest.mark.parametrize(
    "engine",
    [
        RangeAccrualAnalyticalEngine(),
        RangeAccrualMCEngine(
            params=MCParams(
                num_paths=4096,
                time_steps=8,
                seed=41,
                use_antithetic=True,
            )
        ),
    ],
)
def test_terminal_range_accrual_amount_uses_terminal_payment(env, engine):
    config = RangeAccrualConfig(
        lower_barrier=1.0,
        upper_barrier=1_000.0,
        accrual_rate=0.08,
        is_rate_annualized=True,
    )

    def _product(convention):
        return RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[0.25, MATURITY],
            maturity=MATURITY,
            settlement_convention=convention,
        )

    immediate = engine.price(_product(None), env)
    delayed = engine.price(_product(_delayed()), env)
    ratio = (
        env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(MATURITY)
    )

    assert delayed == pytest.approx(immediate * ratio, rel=2.0e-12)


class _FlatPathGenerator:
    def __init__(self, num_times):
        self.num_times = num_times

    def generate_paths(self, **_kwargs):
        paths = np.full((32, self.num_times + 1), 101.0)
        return paths, None


def _numeric_accumulator():
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_time=0.25,
                settlement_time=0.30,
                barrier=1_000.0,
            ),
            ObservationRecord(
                observation_time=0.50,
                settlement_time=0.65,
                barrier=1_000.0,
            ),
        ],
        aggregation_mode=ObservationAggregation.ACCUMULATE,
    )
    return AccumulatorOption(
        strike=100.0,
        knock_out_barrier=1_000.0,
        maturity=0.50,
        daily_share_accumulation=1.0,
        gearing=2.0,
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        observation_schedule=schedule,
        settlement_convention=SettlementConvention(
            lag=0.40,
            lag_unit=SettlementLagUnit.YEAR_FRACTION,
        ),
    )


def test_accumulator_analytical_uses_each_record_payment_df(
    env, monkeypatch
):
    engine = AccumulatorAnalyticalEngine()
    monkeypatch.setattr(
        engine,
        "_price_leg",
        lambda _product, pricing_env, _dates, maturity_i, _daily: (
            pricing_env.get_discount_factor(maturity_i)
        ),
    )

    price = engine.price(_numeric_accumulator(), env)

    assert price == pytest.approx(
        env.get_discount_factor(0.30) + env.get_discount_factor(0.65)
    )


def test_accumulator_mc_uses_each_record_payment_df(env, monkeypatch):
    engine = AccumulatorMCEngine(
        params=MCParams(num_paths=32, time_steps=2, seed=43)
    )
    monkeypatch.setattr(
        engine,
        "_create_path_generator",
        lambda *_args: _FlatPathGenerator(num_times=2),
    )

    price = engine.price(_numeric_accumulator(), env)

    assert price == pytest.approx(
        env.get_discount_factor(0.30) + env.get_discount_factor(0.65)
    )


def test_terminal_settlement_date_does_not_replace_event_dates(
    env, monkeypatch
):
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_date=datetime(2026, 3, 1),
                settlement_date=datetime(2026, 3, 3),
                barrier=1_000.0,
            ),
            ObservationRecord(
                observation_date=datetime(2026, 6, 1),
                settlement_date=datetime(2026, 6, 10),
                barrier=1_000.0,
            ),
        ],
        aggregation_mode=ObservationAggregation.ACCUMULATE,
    )
    product = AccumulatorOption(
        strike=100.0,
        knock_out_barrier=1_000.0,
        exercise_date=datetime(2026, 12, 31),
        settlement_date=datetime(2027, 1, 15),
        daily_share_accumulation=1.0,
        gearing=2.0,
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        observation_schedule=schedule,
    )
    engine = AccumulatorMCEngine(
        params=MCParams(num_paths=32, time_steps=2, seed=47)
    )
    monkeypatch.setattr(
        engine,
        "_create_path_generator",
        lambda *_args: _FlatPathGenerator(num_times=3),
    )

    price = engine.price(product, env)
    payment_times = [
        record.settlement_time
        for record in schedule.resolve(env, product=product)
    ]

    assert price == pytest.approx(
        sum(env.get_discount_factor(t) for t in payment_times)
    )
