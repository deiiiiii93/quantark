"""The date / numeric clock rule (user ruling 2026-10-26; ship spec 2026-10-26-date-numeric-clock).

An equity or FX option is on the DATE clock (every time a date) or the NUMERIC clock (every time a float), never
mixed. Both given completely: the dates win and the floats are ignored. Partial input is a construction error.

This module checks the detection (``quantark.asset.clock``): which clock a product is on and which of its fields
break the rule. Construction refuses a mixed product (``test_clock_enforcement``), so each case here builds a valid
product and then breaks it, the way a post-construction edit would.
"""
from dataclasses import replace
from datetime import datetime

import pytest

from quantark.asset.clock import ProductClock, clock_violations, product_clock
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.asian_option import AsianOption
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.asset.fx.product.option.fx_barrier_option import FxBarrierOption
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.util.enum.fx_enums import FxBarrierType
from quantark.util.enum.option_enums import BarrierType, ObservationType, OptionType

T0 = datetime(2026, 1, 5)
EXPIRY = datetime(2027, 1, 5)
OBS = [datetime(2026, 4, 6), datetime(2026, 7, 6), datetime(2026, 10, 5), EXPIRY]


def _fx_vanilla(**kw):
    return FxVanillaOption(notional_foreign=1e6, **kw)


def _fields(product):
    return {field for field, _ in clock_violations(product)}


def _snowball(dated=True, **kw):
    if dated:
        sched = ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=103.0) for d in OBS])
        config = BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                               ko_observation_schedule=sched, ki_barrier=75.0, ki_continuous=True)
        return SnowballOption(initial_price=100.0, strike=100.0, initial_date=T0, exercise_date=EXPIRY,
                              barrier_config=config, **kw)
    config = BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
                           ki_barrier=75.0, ki_continuous=True)
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0, barrier_config=config, **kw)


# --- the clock -----------------------------------------------------------------------------------------------------

def test_the_anchor_decides_the_clock():
    assert product_clock(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)) \
        is ProductClock.NUMERIC
    assert product_clock(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=EXPIRY)) \
        is ProductClock.DATE
    assert product_clock(_fx_vanilla(strike=1.1, option_type=OptionType.CALL, maturity=1.0)) \
        is ProductClock.NUMERIC
    assert product_clock(_fx_vanilla(strike=1.1, option_type=OptionType.CALL, expiry_date=EXPIRY)) \
        is ProductClock.DATE


# --- complete clocks have nothing to report ---------------------------------------------------------------------

@pytest.mark.parametrize("dated", [True, False])
def test_a_complete_clock_reports_nothing(dated):
    assert clock_violations(_snowball(dated)) == []


def test_derived_and_neutral_floats_on_a_dated_contract_are_not_violations():
    note = _snowball()
    note.accrual_config = replace(note.accrual_config, accrued_offset=0.1)
    note.tenor = 1.0
    assert clock_violations(note) == []


def test_a_shadowed_float_is_not_a_violation():
    vanilla = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=EXPIRY)
    vanilla.maturity = 1.0
    assert clock_violations(vanilla) == []
    fx = _fx_vanilla(strike=1.1, option_type=OptionType.CALL, expiry_date=EXPIRY, delivery_date=EXPIRY)
    fx.maturity, fx.delivery = 1.0, 1.0
    assert clock_violations(fx) == []


def test_a_float_list_beside_its_dates_is_shadowed():
    """Both forms of the same observations given: the dates win (CP1 review)."""
    asian = AsianOption(strike=100.0, exercise_date=EXPIRY, observation_times=[0.5, 1.0],
                        observation_dates=OBS[1::2])
    assert clock_violations(asian) == []
    records = [ObservationRecord(observation_date=d, barrier=90.0) for d in OBS]
    barrier = BarrierOption(strike=100.0, option_type=OptionType.PUT, barrier=90.0, barrier_type=BarrierType.DOWN_IN,
                            exercise_date=EXPIRY, observation_type=ObservationType.DISCRETE,
                            observation_schedule=ObservationSchedule(records=records))
    barrier.observation_dates = [0.25, 0.5, 0.75, 1.0]
    assert clock_violations(barrier) == []
    note = _snowball()
    note.barrier_config = replace(note.barrier_config, ko_observation_dates=[0.25, 0.5, 0.75, 1.0])
    assert clock_violations(note) == []


# --- a date field on the numeric clock ------------------------------------------------------------------------------

def test_a_numeric_note_with_an_initial_date_is_partial():
    note = _snowball(dated=False)
    note.initial_date = T0
    assert _fields(note) == {"initial_date"}


def _numeric_barrier():
    records = [ObservationRecord(observation_time=t, barrier=90.0) for t in (0.5, 1.0)]
    return BarrierOption(strike=100.0, option_type=OptionType.PUT, barrier=90.0, barrier_type=BarrierType.DOWN_IN,
                         maturity=1.0, observation_type=ObservationType.DISCRETE,
                         observation_schedule=ObservationSchedule(records=records))


def test_a_numeric_contract_with_a_dated_record_is_mixed():
    barrier = _numeric_barrier()
    barrier.observation_schedule = ObservationSchedule(records=[
        ObservationRecord(observation_time=0.5, barrier=90.0), ObservationRecord(observation_date=OBS[2], barrier=90.0)])
    assert any("observation_date" in field for field in _fields(barrier))


def test_a_numeric_fx_option_with_a_delivery_date_is_mixed():
    fx = _fx_vanilla(strike=1.1, option_type=OptionType.CALL, maturity=1.0)
    fx.delivery_date = EXPIRY
    assert _fields(fx) == {"delivery_date"}


def test_a_numeric_asian_with_dated_observations_is_mixed():
    asian = AsianOption(strike=100.0, maturity=1.0, observation_times=[0.5, 1.0])
    asian.observation_dates = OBS[1::2]
    assert "observation_dates" in _fields(asian)


# --- a numeric time on the date clock -------------------------------------------------------------------------------

def test_a_dated_contract_with_a_time_only_record_is_partial():
    records = [ObservationRecord(observation_date=d, barrier=90.0) for d in OBS]
    barrier = BarrierOption(strike=100.0, option_type=OptionType.PUT, barrier=90.0, barrier_type=BarrierType.DOWN_IN,
                            exercise_date=EXPIRY, observation_type=ObservationType.DISCRETE,
                            observation_schedule=ObservationSchedule(records=records))
    barrier.observation_schedule = ObservationSchedule(records=[
        ObservationRecord(observation_date=OBS[0], barrier=90.0), ObservationRecord(observation_time=0.5, barrier=90.0)])
    assert any("observation_time" in field for field in _fields(barrier))


def test_a_dated_contract_with_a_legacy_float_list_is_partial():
    note = _snowball()
    note.barrier_config = replace(note.barrier_config, ki_observation_dates=[0.5, 1.0])
    assert "barrier_config.ki_observation_dates" in _fields(note)


@pytest.mark.parametrize("unit, lag", [(SettlementLagUnit.YEAR_FRACTION, 2.0 / 365.0),
                                       (SettlementLagUnit.CALENDAR_DAYS, 2), (SettlementLagUnit.BUSINESS_DAYS, 2)])
def test_a_settlement_lag_is_a_product_term_on_either_clock(unit, lag):
    """User ruling 2026-10-26: a settlement lag is a product term (a settlement date, "1D", "2D"), not a clock time."""
    convention = SettlementConvention(lag, unit)
    dated = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=EXPIRY,
                                  settlement_convention=convention)
    numeric = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0,
                                    settlement_convention=convention)
    assert clock_violations(dated) == [] and clock_violations(numeric) == []


def test_a_dated_fx_option_needing_float_times_is_partial():
    fx = FxBarrierOption(strike=1.1, barrier=1.2, is_up=True, knock_type=FxBarrierType.KNOCK_OUT,
                         option_type=OptionType.CALL, monitoring=ObservationType.DISCRETE,
                         observation_dates=OBS, expiry_date=EXPIRY)
    fx.observation_dates, fx.observation_times = None, [0.5, 1.0]
    assert "observation_times" in _fields(fx)
    alone = _fx_vanilla(strike=1.1, option_type=OptionType.CALL, expiry_date=EXPIRY)
    alone.delivery = 1.01
    assert _fields(alone) == {"delivery"}
