"""Product and observation-schedule settlement contract tests."""

from datetime import datetime
import inspect
from types import SimpleNamespace

import pytest

from quantark.asset.equity.product.option import (
    DCNDirection,
    DCNOption,
    DoubleOneTouchOption,
    EuropeanVanillaOption,
    OneTouchOption,
)
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.asset.equity.settlement import (
    SettlementConvention,
    SettlementLagUnit,
)
from quantark.param import FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import Calendar, DayCountConvention
from quantark.util.enum import (
    BarrierDirection,
    OptionType,
    TouchType,
)
from quantark.util.exceptions import ValidationError


@pytest.fixture
def pricing_env():
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 29),
        day_count_convention=DayCountConvention.ACT_365,
        calendar=Calendar(holidays={datetime(2026, 8, 3)}),
    )


@pytest.fixture
def t2_convention():
    return SettlementConvention(
        lag=2,
        lag_unit=SettlementLagUnit.BUSINESS_DAYS,
    )


def test_standalone_option_signatures_expose_settlement_convention():
    for product_class in (OneTouchOption, DoubleOneTouchOption, DCNOption):
        assert "settlement_convention" in inspect.signature(
            product_class
        ).parameters


def test_base_option_preserves_settlement_convention(t2_convention):
    product = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        settlement_convention=t2_convention,
    )

    assert product.settlement_convention is t2_convention


def test_one_touch_preserves_settlement_convention(t2_convention):
    product = OneTouchOption(
        barrier=110.0,
        barrier_direction=BarrierDirection.UP,
        maturity=1.0,
        settlement_convention=t2_convention,
    )

    assert product.settlement_convention is t2_convention


def test_double_one_touch_preserves_settlement_convention(t2_convention):
    product = DoubleOneTouchOption(
        upper_barrier=110.0,
        lower_barrier=90.0,
        maturity=1.0,
        touch_type=TouchType.DOUBLE_ONE_TOUCH,
        settlement_convention=t2_convention,
    )

    assert product.settlement_convention is t2_convention


def test_dcn_preserves_settlement_convention(t2_convention):
    final_observation = datetime(2026, 12, 30)
    schedule = SimpleNamespace(
        monthly=[SimpleNamespace(observation_date=final_observation)]
    )
    product = DCNOption(
        notional=1_000_000.0,
        initial_price=100.0,
        direction=DCNDirection.BUYER,
        coupon_barrier_ratio=0.8,
        ko_barrier_ratio=1.03,
        ki_barrier_ratio=0.7,
        ki_put_strike_ratio=1.0,
        coupon_rate=0.12,
        ko_coupon_rate=0.12,
        participation=1.0,
        coupon_counted_days=30,
        coupon_days_denom=360,
        schedule=schedule,
        settlement_date=datetime(2026, 12, 31),
        settlement_convention=t2_convention,
    )

    assert product.settlement_convention is t2_convention


def test_invalid_product_settlement_convention_rejected():
    with pytest.raises(ValidationError, match="settlement_convention"):
        EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
            settlement_convention="T+2",
        )


def test_record_explicit_settlement_time_survives_resolution(pricing_env):
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_time=0.5,
                settlement_time=0.55,
                barrier=110.0,
            )
        ]
    )

    [record] = schedule.resolve(pricing_env, require_single=True)

    assert record.observation_time == pytest.approx(0.5)
    assert record.settlement_time == pytest.approx(0.55)
    assert record.observation_date is None
    assert record.settlement_date is None


def test_invalid_record_settlement_date_fails_closed(pricing_env):
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_date=datetime(2026, 9, 1),
                settlement_date=datetime(2026, 8, 31),
                barrier=110.0,
            )
        ]
    )

    with pytest.raises(ValidationError, match="before determination"):
        schedule.resolve(pricing_env, require_single=True)


def test_record_explicit_payment_beats_product_convention(
    pricing_env, t2_convention
):
    product = SimpleNamespace(
        settlement_date=datetime(2026, 12, 31),
        settlement_convention=t2_convention,
    )
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_date=datetime(2026, 8, 4),
                settlement_date=datetime(2026, 8, 7),
                barrier=110.0,
            )
        ]
    )

    [record] = schedule.resolve(
        pricing_env,
        require_single=True,
        product=product,
    )

    assert record.observation_date == datetime(2026, 8, 4)
    assert record.settlement_date == datetime(2026, 8, 7)


def test_event_ignores_product_terminal_settlement_date(pricing_env):
    product = SimpleNamespace(
        settlement_date=datetime(2026, 12, 31),
        settlement_convention=None,
    )
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_date=datetime(2026, 8, 4),
                barrier=110.0,
            )
        ]
    )

    [record] = schedule.resolve(
        pricing_env,
        require_single=True,
        product=product,
    )

    assert record.settlement_date == record.observation_date
    assert record.settlement_time == pytest.approx(record.observation_time)


def test_event_uses_product_settlement_convention(
    pricing_env, t2_convention
):
    product = SimpleNamespace(
        settlement_date=datetime(2026, 12, 31),
        settlement_convention=t2_convention,
    )
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_date=datetime(2026, 7, 31),
                barrier=110.0,
            )
        ]
    )

    [record] = schedule.resolve(
        pricing_env,
        require_single=True,
        product=product,
    )

    assert record.settlement_date == datetime(2026, 8, 5)


def test_time_shift_moves_explicit_observation_and_settlement_times():
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_time=0.5,
                settlement_time=0.6,
                barrier=110.0,
            )
        ]
    )

    shifted = schedule.time_shift(0.1)

    assert shifted is not None
    assert shifted.records[0].observation_time == pytest.approx(0.4)
    assert shifted.records[0].settlement_time == pytest.approx(0.5)


def test_schedule_rejects_both_product_and_convention_context(
    pricing_env, t2_convention
):
    schedule = ObservationSchedule(
        records=[ObservationRecord(observation_time=0.5, barrier=110.0)]
    )
    product = SimpleNamespace(
        settlement_date=None,
        settlement_convention=t2_convention,
    )

    with pytest.raises(ValidationError, match="either product or"):
        schedule.resolve(
            pricing_env,
            require_single=True,
            product=product,
            settlement_convention=t2_convention,
        )
