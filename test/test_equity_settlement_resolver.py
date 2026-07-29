"""Contract tests for equity cashflow settlement timing resolution."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.settlement import (
    CashflowKind,
    SettlementConvention,
    SettlementLagUnit,
    SettlementRequest,
    SettlementResolver,
)
from quantark.param import FlatRateCurve
from quantark.param.rrf.rate_curve import RateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import (
    BusinessDayConvention,
    Calendar,
    DayCountConvention,
    calculate_year_fraction,
)
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError


@pytest.fixture
def settlement_calendar():
    return Calendar(
        holidays={datetime(2026, 8, 3)},
        name="Settlement Test Calendar",
    )


@pytest.fixture
def pricing_env(settlement_calendar):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 29),
        day_count_convention=DayCountConvention.ACT_365,
        calendar=settlement_calendar,
    )


@pytest.fixture
def date_option():
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        exercise_date=datetime(2026, 8, 7),
    )


@pytest.fixture
def time_option():
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )


def test_zero_lag_is_identity(date_option, pricing_env):
    timing = SettlementResolver.resolve_contingent(
        date_option,
        SettlementRequest(
            kind=CashflowKind.TERMINAL,
            determination_date=date_option.exercise_date,
        ),
        pricing_env,
    )

    assert timing.payment_date == timing.determination_date
    assert timing.payment_time == pytest.approx(timing.determination_time)
    assert timing.determination_df == pytest.approx(timing.payment_df)
    assert timing.delay_df == pytest.approx(1.0)


def test_zero_business_day_lag_needs_no_calendar(time_option):
    time_option.settlement_convention = SettlementConvention(
        lag=0,
        lag_unit=SettlementLagUnit.BUSINESS_DAYS,
    )
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 29),
    )

    timing = SettlementResolver.resolve_contingent(
        time_option,
        SettlementRequest(
            kind=CashflowKind.TERMINAL,
            determination_time=1.0,
        ),
        env,
    )

    assert timing.payment_time == pytest.approx(1.0)
    assert timing.payment_date is None


def test_explicit_event_payment_beats_product_convention(
    date_option, pricing_env
):
    date_option.settlement_convention = SettlementConvention(
        lag=2,
        lag_unit=SettlementLagUnit.BUSINESS_DAYS,
        calendar=pricing_env.calendar,
    )
    explicit = datetime(2026, 8, 7)

    timing = SettlementResolver.resolve_contingent(
        date_option,
        SettlementRequest(
            kind=CashflowKind.COUPON,
            determination_date=datetime(2026, 8, 4),
            explicit_payment_date=explicit,
        ),
        pricing_env,
    )

    assert timing.payment_date == explicit


def test_terminal_override_does_not_apply_to_event(date_option, pricing_env):
    terminal_payment = datetime(2026, 8, 11)
    date_option.settlement_date = terminal_payment

    event = SettlementResolver.resolve_contingent(
        date_option,
        SettlementRequest(
            kind=CashflowKind.COUPON,
            determination_date=datetime(2026, 8, 4),
        ),
        pricing_env,
    )
    terminal = SettlementResolver.resolve_contingent(
        date_option,
        SettlementRequest(
            kind=CashflowKind.TERMINAL,
            determination_date=date_option.exercise_date,
        ),
        pricing_env,
    )

    assert event.payment_date == event.determination_date
    assert terminal.payment_date == terminal_payment


def test_business_day_lag_skips_weekend_and_holiday(
    date_option, pricing_env
):
    date_option.settlement_convention = SettlementConvention(
        lag=2,
        lag_unit=SettlementLagUnit.BUSINESS_DAYS,
    )

    timing = SettlementResolver.resolve_contingent(
        date_option,
        SettlementRequest(
            kind=CashflowKind.COUPON,
            determination_date=datetime(2026, 7, 31),
        ),
        pricing_env,
    )

    assert timing.payment_date == datetime(2026, 8, 5)


def test_calendar_day_lag_applies_modified_following(pricing_env):
    product = SimpleNamespace(
        settlement_date=None,
        settlement_convention=SettlementConvention(
            lag=2,
            lag_unit=SettlementLagUnit.CALENDAR_DAYS,
            business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
        ),
    )

    timing = SettlementResolver.resolve_contingent(
        product,
        SettlementRequest(
            kind=CashflowKind.COUPON,
            determination_date=datetime(2026, 8, 28),
        ),
        pricing_env,
    )

    assert timing.payment_date == datetime(2026, 8, 31)


def test_year_fraction_lag_supports_time_only_product(
    time_option, pricing_env
):
    time_option.settlement_convention = SettlementConvention(
        lag=0.125,
        lag_unit=SettlementLagUnit.YEAR_FRACTION,
    )

    timing = SettlementResolver.resolve_contingent(
        time_option,
        SettlementRequest(
            kind=CashflowKind.TERMINAL,
            determination_time=1.0,
        ),
        pricing_env,
    )

    assert timing.payment_date is None
    assert timing.payment_time == pytest.approx(1.125)
    assert timing.delay_df == pytest.approx(
        pricing_env.get_discount_factor(1.125)
        / pricing_env.get_discount_factor(1.0)
    )


def test_time_only_determination_rejects_business_day_lag(
    time_option, pricing_env
):
    time_option.settlement_convention = SettlementConvention(
        lag=2,
        lag_unit=SettlementLagUnit.BUSINESS_DAYS,
        calendar=pricing_env.calendar,
    )

    with pytest.raises(
        ValidationError, match="authoritative determination date"
    ):
        SettlementResolver.resolve_contingent(
            time_option,
            SettlementRequest(
                kind=CashflowKind.TERMINAL,
                determination_time=1.0,
            ),
            pricing_env,
        )


def test_business_day_lag_requires_calendar(date_option):
    date_option.settlement_convention = SettlementConvention(
        lag=2,
        lag_unit=SettlementLagUnit.BUSINESS_DAYS,
    )
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 29),
        day_count_convention=DayCountConvention.ACT_365,
    )

    with pytest.raises(ValidationError, match="calendar"):
        SettlementResolver.resolve_contingent(
            date_option,
            SettlementRequest(
                kind=CashflowKind.TERMINAL,
                determination_date=date_option.exercise_date,
            ),
            env,
        )


def test_consistent_determination_date_and_time_are_accepted(
    date_option, pricing_env
):
    determination_time = calculate_year_fraction(
        pricing_env.valuation_date,
        date_option.exercise_date,
        pricing_env.day_count_convention,
        pricing_env.bus_days_in_year,
        calendar=pricing_env.calendar,
    )

    timing = SettlementResolver.resolve_contingent(
        date_option,
        SettlementRequest(
            kind=CashflowKind.TERMINAL,
            determination_date=date_option.exercise_date,
            determination_time=determination_time,
        ),
        pricing_env,
    )

    assert timing.determination_time == pytest.approx(determination_time)


def test_inconsistent_determination_date_and_time_are_rejected(
    date_option, pricing_env
):
    with pytest.raises(ValidationError, match="inconsistent"):
        SettlementResolver.resolve_contingent(
            date_option,
            SettlementRequest(
                kind=CashflowKind.TERMINAL,
                determination_date=date_option.exercise_date,
                determination_time=0.5,
            ),
            pricing_env,
        )


def test_consistent_explicit_payment_date_and_time_are_accepted(
    date_option, pricing_env
):
    payment_date = datetime(2026, 8, 11)
    payment_time = calculate_year_fraction(
        pricing_env.valuation_date,
        payment_date,
        pricing_env.day_count_convention,
        pricing_env.bus_days_in_year,
        calendar=pricing_env.calendar,
    )

    timing = SettlementResolver.resolve_contingent(
        date_option,
        SettlementRequest(
            kind=CashflowKind.TERMINAL,
            determination_date=date_option.exercise_date,
            explicit_payment_date=payment_date,
            explicit_payment_time=payment_time,
        ),
        pricing_env,
    )

    assert timing.payment_date == payment_date
    assert timing.payment_time == pytest.approx(payment_time)


def test_inconsistent_explicit_payment_date_and_time_are_rejected(
    date_option, pricing_env
):
    with pytest.raises(ValidationError, match="inconsistent"):
        SettlementResolver.resolve_contingent(
            date_option,
            SettlementRequest(
                kind=CashflowKind.TERMINAL,
                determination_date=date_option.exercise_date,
                explicit_payment_date=datetime(2026, 8, 11),
                explicit_payment_time=0.75,
            ),
            pricing_env,
        )


def test_explicit_payment_before_determination_is_rejected(
    date_option, pricing_env
):
    with pytest.raises(ValidationError, match="before determination"):
        SettlementResolver.resolve_contingent(
            date_option,
            SettlementRequest(
                kind=CashflowKind.TERMINAL,
                determination_date=date_option.exercise_date,
                explicit_payment_date=datetime(2026, 8, 6),
            ),
            pricing_env,
        )


def test_terminal_override_before_determination_is_rejected(
    date_option, pricing_env
):
    date_option.settlement_date = datetime(2026, 8, 6)

    with pytest.raises(ValidationError, match="before determination"):
        SettlementResolver.resolve_contingent(
            date_option,
            SettlementRequest(
                kind=CashflowKind.TERMINAL,
                determination_date=date_option.exercise_date,
            ),
            pricing_env,
        )


@pytest.mark.parametrize("bad_lag", [-1.0, float("nan"), float("inf")])
def test_invalid_lag_rejected(bad_lag):
    with pytest.raises(ValidationError):
        SettlementConvention(lag=bad_lag)


@pytest.mark.parametrize("bad_lag", [1.5, 2.25])
def test_fractional_day_lag_rejected(bad_lag):
    with pytest.raises(ValidationError, match="integral"):
        SettlementConvention(
            lag=bad_lag,
            lag_unit=SettlementLagUnit.CALENDAR_DAYS,
        )


@pytest.mark.parametrize("bad_df", [0.0, -0.5, float("nan"), float("inf")])
def test_non_positive_or_non_finite_discount_factor_rejected(
    date_option, pricing_env, bad_df
):
    class BadCurve(RateCurve):
        def get_rate(self, time_to_maturity):
            return 0.03

        def get_discount_factor(self, time_to_maturity):
            return bad_df

    pricing_env.rate_curve = BadCurve()

    with pytest.raises(ValidationError, match="discount factor"):
        SettlementResolver.resolve_contingent(
            date_option,
            SettlementRequest(
                kind=CashflowKind.TERMINAL,
                determination_date=date_option.exercise_date,
            ),
            pricing_env,
        )


def test_error_identifies_product_cashflow_and_request(pricing_env):
    product = SimpleNamespace(
        settlement_date=None,
        settlement_convention=SettlementConvention(
            lag=2,
            lag_unit=SettlementLagUnit.BUSINESS_DAYS,
        ),
    )
    request = SettlementRequest(
        kind=CashflowKind.COUPON,
        determination_time=0.5,
        cashflow_id="trade-1:coupon:3",
    )

    with pytest.raises(ValidationError) as exc_info:
        SettlementResolver.resolve_contingent(product, request, pricing_env)

    message = str(exc_info.value)
    assert "SimpleNamespace" in message
    assert "coupon" in message
    assert "trade-1:coupon:3" in message
    assert "BUSINESS_DAYS" in message
