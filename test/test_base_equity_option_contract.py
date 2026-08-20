"""Regression tests for the BaseEquityOption constructor contract."""

from datetime import datetime
import inspect
from types import SimpleNamespace

import pytest

from quantark.asset.equity.product.option import (
    AccumulatorOption,
    AmericanOption,
    AsianOption,
    BarrierConfig,
    BarrierOption,
    BaseEquityOption,
    CashOrNothingDigitalOption,
    CouponBarrierConfig,
    DoubleBarrierOption,
    DoubleSharkfinOption,
    EuropeanVanillaOption,
    KnockOutResetSnowballOption,
    PhoenixOption,
    RangeAccrualConfig,
    RangeAccrualOption,
    SingleSharkfinOption,
    SnowballOption,
)
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import (
    BarrierType,
    DoubleBarrierType,
    ObservationType,
    OptionType,
    TenorEnd,
)


CONCRETE_OPTIONS = (
    EuropeanVanillaOption,
    AmericanOption,
    AsianOption,
    BarrierOption,
    CashOrNothingDigitalOption,
    DoubleBarrierOption,
    SingleSharkfinOption,
    DoubleSharkfinOption,
    AccumulatorOption,
    RangeAccrualOption,
    SnowballOption,
    PhoenixOption,
    KnockOutResetSnowballOption,
)

LIFECYCLE_PARAMETERS = {
    "maturity",
    "tenor",
    "initial_date",
    "exercise_date",
    "settlement_date",
    "maturity_date",
    "tenor_end",
    "annualization_day_count",
    "contract_multiplier",
}


def test_every_concrete_option_honors_the_base_lifecycle_signature():
    for option_class in CONCRETE_OPTIONS:
        parameters = set(inspect.signature(option_class).parameters)
        assert LIFECYCLE_PARAMETERS <= parameters, option_class.__name__


def test_initial_price_is_owned_only_by_reference_fixing_products():
    assert "initial_price" not in inspect.signature(BaseEquityOption).parameters

    absolute_strike_products = (
        EuropeanVanillaOption,
        AmericanOption,
        BarrierOption,
        CashOrNothingDigitalOption,
        DoubleBarrierOption,
        SingleSharkfinOption,
        DoubleSharkfinOption,
    )
    reference_fixing_products = (
        AsianOption,
        AccumulatorOption,
        RangeAccrualOption,
        SnowballOption,
        PhoenixOption,
        KnockOutResetSnowballOption,
    )

    for option_class in absolute_strike_products:
        assert "initial_price" not in inspect.signature(option_class).parameters
    for option_class in reference_fixing_products:
        assert "initial_price" in inspect.signature(option_class).parameters

    vanilla = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
    assert not hasattr(vanilla, "initial_price")


def test_maturity_date_and_tenor_are_real_construction_forms():
    valuation_date = datetime(2024, 1, 1)
    expiry_date = datetime(2025, 1, 1)
    pricing_env = SimpleNamespace(
        valuation_date=valuation_date,
        day_count_convention=DayCountConvention.ACT_365,
        bus_days_in_year=252,
        calendar=None,
    )

    dated = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        initial_date=valuation_date,
        maturity_date=expiry_date,
    )
    assert dated.exercise_date == expiry_date
    assert dated.get_maturity(pricing_env) == pytest.approx(366.0 / 365.0)
    assert dated.get_tenor() == pytest.approx(366.0 / 365.0)

    tenored = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        tenor=1.25,
    )
    assert tenored.maturity == pytest.approx(1.25)
    assert tenored.get_maturity() == pytest.approx(1.25)
    assert tenored.get_tenor() == pytest.approx(1.25)


def _shared_lifecycle_kwargs():
    return {
        "maturity": 1.0,
        "tenor": 1.25,
        "initial_date": datetime(2024, 1, 1),
        "settlement_date": datetime(2025, 1, 3),
        "maturity_date": datetime(2025, 1, 1),
        "tenor_end": TenorEnd.MATURITY,
        "annualization_day_count": DayCountConvention.ACT_365,
        "contract_multiplier": 2.0,
    }


def _barrier_config():
    return BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.10,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=75.0,
        ki_observation_type=ObservationType.CONTINUOUS,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda terms: EuropeanVanillaOption(100.0, OptionType.CALL, **terms),
        lambda terms: AmericanOption(100.0, OptionType.PUT, **terms),
        lambda terms: AsianOption(
            strike=100.0, option_type=OptionType.CALL, initial_price=100.0, **terms
        ),
        lambda terms: BarrierOption(
            100.0, OptionType.CALL, 120.0, BarrierType.UP_OUT, **terms
        ),
        lambda terms: CashOrNothingDigitalOption(
            100.0, 10.0, OptionType.CALL, **terms
        ),
        lambda terms: DoubleBarrierOption(
            100.0,
            OptionType.CALL,
            120.0,
            80.0,
            DoubleBarrierType.KNOCK_OUT,
            **terms,
        ),
        lambda terms: SingleSharkfinOption(
            100.0, OptionType.CALL, 120.0, **terms
        ),
        lambda terms: DoubleSharkfinOption(
            100.0, OptionType.CALL, 120.0, 80.0, **terms
        ),
        lambda terms: AccumulatorOption(
            strike=95.0,
            knock_out_barrier=105.0,
            initial_price=100.0,
            notional=1_000_000.0,
            observation_dates=[0.25, 0.5, 0.75, 1.0],
            **terms,
        ),
        lambda terms: RangeAccrualOption(
            initial_price=100.0,
            range_config=RangeAccrualConfig(
                upper_barrier=110.0,
                lower_barrier=90.0,
                accrual_rate=0.05,
            ),
            num_observations=12,
            **terms,
        ),
        lambda terms: SnowballOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=_barrier_config(),
            **terms,
        ),
        lambda terms: PhoenixOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=_barrier_config(),
            coupon_config=CouponBarrierConfig(
                coupon_barrier=85.0,
                coupon_rate=0.01,
            ),
            **terms,
        ),
    ],
)
def test_concrete_options_forward_lifecycle_values(factory):
    terms = _shared_lifecycle_kwargs()
    option = factory(terms)

    for name, expected in terms.items():
        assert getattr(option, name) == expected
