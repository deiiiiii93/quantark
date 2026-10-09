"""An annualized range accrual's coupon accrues over the CONTRACT, not over what is left of it.

The coupon is ``rate x (in-range weight / total weight) x YF``. The ratio is the whole contract's (past fixings plus
future ones), so the YF must be the whole contract's too: ``tenor`` when set, else ``initial_date`` -> expiry in the
config's day count, else (numeric) ``maturity`` plus the time ``time_shift`` has banked. The remaining maturity -- what
these products used before -- shrinks the coupon every day the contract ages (findings 2026-10-08 item 9).

The control contract: one year, 10% annualized, 100 notional, twelve monthly fixings, a band so wide every fixing is
in range. Its coupon is 10.0 on every valuation date, so its price is 10 x DF(valuation -> expiry).
"""
import copy
import math
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical import RangeAccrualAnalyticalEngine
from quantark.asset.equity.product.option.range_accrual_config import (
    RangeAccrualConfig, RangeAccrualObservationRecord,
)
from quantark.asset.equity.product.option.range_accrual_option import RangeAccrualOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.fx.engine.analytical import FxRangeAccrualAnalyticalEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxForeignRangeAccrualOption, FxRangeAccrualConfig, FxRangeAccrualOption,
)
from quantark.asset.fx.product.option.fx_range_accrual_config import FxRangeAccrualObservationRecord
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment, PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention as DC
from quantark.util.exceptions import ValidationError

T0 = datetime(2024, 1, 2)
EXPIRY = T0 + timedelta(days=365)
FIX_DAYS = [round(365 * k / 12) for k in range(1, 13)]
MONTHLY = [k / 12 for k in range(1, 13)]
R, RATE = 0.03, 0.10


def _eq_config(**kw):
    return RangeAccrualConfig(upper_barrier=1e6, lower_barrier=1.0, accrual_rate=RATE, is_rate_annualized=True, **kw)


def _fx_config(**kw):
    return FxRangeAccrualConfig(upper_barrier=1e6, lower_barrier=1e-3, accrual_rate=RATE, is_rate_annualized=True,
                                **kw)


def _eq_env(valuation=T0, spot=100.0):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(R), valuation_date=valuation)


def _fx_env(valuation=T0, spot=1.10):
    return FxPricingEnvironment(valuation_date=valuation, spot_quote=SpotQuote(spot=spot),
                                domestic_curve=FlatRateCurve(rate=R), foreign_curve=FlatRateCurve(rate=0.01),
                                vol_surface=FlatVolSurface(volatility=0.08))


def _eq_dated(age_days, **kw):
    records = [RangeAccrualObservationRecord(observation_date=T0 + timedelta(days=d),
                                             observed_in_range=True if d <= age_days else None) for d in FIX_DAYS]
    return RangeAccrualOption(initial_price=100.0, range_config=kw.pop("range_config", _eq_config()),
                              exercise_date=EXPIRY, initial_date=kw.pop("initial_date", T0),
                              observation_records=records, **kw)


def _fx_dated(age_days, cls=FxRangeAccrualOption, **kw):
    records = [FxRangeAccrualObservationRecord(observation_date=T0 + timedelta(days=d),
                                               observed_in_range=True if d <= age_days else None) for d in FIX_DAYS]
    return cls(notional=100.0, range_config=kw.pop("range_config", _fx_config()),
               currency_pair=CurrencyPair("EUR", "USD"), expiry_date=EXPIRY,
               initial_date=kw.pop("initial_date", T0), observation_records=records, **kw)


# ---------------------------------------------------------------------------
# Dated contracts: the year fraction is initial_date -> expiry in the config's day count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("age", [0, 91, 182, 273, 364])
def test_an_aged_dated_equity_contract_is_priced_at_its_whole_coupon(age):
    env = _eq_env(T0 + timedelta(days=age))
    price = RangeAccrualAnalyticalEngine().price(_eq_dated(age), env)
    assert price == pytest.approx(10.0 * math.exp(-R * (365 - age) / 365), rel=1e-9)


@pytest.mark.parametrize("age", [0, 91, 182, 273, 364])
def test_an_aged_dated_fx_contract_is_priced_at_its_whole_coupon(age):
    env = _fx_env(T0 + timedelta(days=age))
    price = FxRangeAccrualAnalyticalEngine().price(_fx_dated(age), env)
    assert price == pytest.approx(10.0 * math.exp(-R * (365 - age) / 365), rel=1e-9)


def test_the_equity_year_fraction_reads_the_configs_day_count():
    product = _eq_dated(182, range_config=_eq_config(day_count_convention=DC.ACT_360))
    assert product.get_year_fraction(_eq_env(T0 + timedelta(days=182))) == pytest.approx(365 / 360, rel=1e-12)


def test_the_fx_year_fraction_reads_the_configs_day_count():
    product = _fx_dated(182, range_config=_fx_config(day_count_convention=DC.ACT_360))
    assert product.get_year_fraction(_fx_env(T0 + timedelta(days=182))) == pytest.approx(365 / 360, rel=1e-12)


def test_a_set_tenor_wins_on_both_asset_classes():
    assert _eq_dated(182, tenor=1.0).get_year_fraction(_eq_env(T0 + timedelta(days=182))) == 1.0
    assert _fx_dated(182, tenor=1.0).get_year_fraction(_fx_env(T0 + timedelta(days=182))) == 1.0


def test_an_fx_subclass_forwards_the_contract_start():
    product = _fx_dated(91, cls=FxForeignRangeAccrualOption)
    assert product.get_year_fraction(_fx_env(T0 + timedelta(days=91))) == pytest.approx(1.0, rel=1e-12)


def test_an_annualized_dated_equity_contract_without_a_start_is_refused():
    with pytest.raises(ValidationError, match="initial_date or tenor"):
        _eq_dated(0, initial_date=None)


def test_an_annualized_dated_fx_contract_without_a_start_is_refused():
    with pytest.raises(ValidationError, match="initial_date or tenor"):
        _fx_dated(0, initial_date=None)


def test_an_unannualized_dated_contract_needs_no_start():
    flat = RangeAccrualConfig(upper_barrier=1e6, lower_barrier=1.0, accrual_rate=RATE)
    assert _eq_dated(0, initial_date=None, range_config=flat).get_year_fraction(_eq_env()) == 1.0


# ---------------------------------------------------------------------------
# Numeric contracts: time_shift banks the elapsed time and moves the fixings with the clock
# ---------------------------------------------------------------------------

def _eq_numeric(**kw):
    return RangeAccrualOption(initial_price=100.0, range_config=kw.pop("range_config", _eq_config()), maturity=1.0,
                              observation_times=MONTHLY, **kw)


def _fx_numeric(**kw):
    return FxRangeAccrualOption(notional=100.0, range_config=kw.pop("range_config", _fx_config()),
                                currency_pair=CurrencyPair("EUR", "USD"), maturity=1.0, observation_times=MONTHLY,
                                **kw)


def test_a_shifted_numeric_equity_contract_keeps_its_whole_year_fraction():
    aged = _eq_numeric()
    env = _eq_env()
    aged.time_shift(0.5, T0 + timedelta(days=182), env)
    assert aged.maturity == pytest.approx(0.5)
    assert aged.get_year_fraction(env) == pytest.approx(1.0, rel=1e-12)


def test_a_shifted_numeric_fx_contract_keeps_its_whole_year_fraction():
    aged = _fx_numeric()
    aged.time_shift(0.5)
    assert aged.maturity == pytest.approx(0.5)
    assert aged.get_year_fraction(_fx_env()) == pytest.approx(1.0, rel=1e-12)


def test_time_shift_moves_the_equity_fixings_and_records_the_passed_ones_at_spot():
    aged = _eq_numeric(range_config=_eq_config())
    aged.range_config = RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=RATE,
                                           is_rate_annualized=True)
    aged.time_shift(0.25, T0 + timedelta(days=91), _eq_env(spot=120.0))
    records = aged.get_observation_records()
    assert [r.observation_time for r in records] == pytest.approx([t - 0.25 for t in MONTHLY])
    # the three fixings at or before the new valuation fixed at the held spot, 120: outside [90, 110]
    assert [r.observed_in_range for r in records[:3]] == [False, False, False]
    assert all(r.observed_in_range is None for r in records[3:])


def test_time_shift_moves_the_fx_fixings_and_records_the_passed_ones_at_spot():
    aged = _fx_numeric()
    aged.time_shift(0.25)
    aged.record_passed_fixings(_fx_env(spot=1.10))
    records = aged.get_observation_records()
    assert [r.observation_time for r in records] == pytest.approx([t - 0.25 for t in MONTHLY])
    assert [r.observed_in_range for r in records[:3]] == [True, True, True]
    assert all(r.observed_in_range is None for r in records[3:])


def test_a_uniform_schedule_keeps_its_fixings_when_shifted():
    """``num_observations`` builds the schedule from ``maturity``; shifting must not rebuild it from the shorter one."""
    aged = RangeAccrualOption(initial_price=100.0, range_config=_eq_config(), maturity=1.0, num_observations=4)
    aged.time_shift(0.1, T0 + timedelta(days=37), _eq_env())
    times = [r.observation_time for r in aged.get_observation_records()]
    assert times == pytest.approx([0.15, 0.4, 0.65, 0.9])


def test_the_original_contract_is_untouched_by_a_shift_of_its_copy():
    product = _eq_numeric()
    aged = copy.deepcopy(product)
    aged.time_shift(0.5, T0 + timedelta(days=182), _eq_env())
    assert [r.observation_time for r in product.get_observation_records()] == pytest.approx(MONTHLY)
    assert product.get_year_fraction(_eq_env()) == 1.0


# ---------------------------------------------------------------------------
# Theta: the coupon no longer bleeds a day's worth of accrual per day
# ---------------------------------------------------------------------------

def test_equity_theta_on_a_certain_coupon_is_the_discount_roll():
    product = _eq_numeric()
    env = _eq_env()
    engine = RangeAccrualAnalyticalEngine()
    price = engine.price(product, env)
    theta = GreeksCalculator().calculate_numerical_theta(product, env, engine)
    # a certain 10.0 at T=1: one day's roll is 10 x (DF(1 - dt) - DF(1)) > 0, about +8e-4; the bug read about -0.027
    assert price == pytest.approx(10.0 * math.exp(-R), rel=1e-9)
    assert 0.0 < theta < 2e-3


def test_fx_theta_on_a_certain_coupon_is_the_discount_roll():
    product = _fx_numeric()
    env = _fx_env()
    theta = FxRangeAccrualAnalyticalEngine().calculate_greeks(product, env)["theta"]
    assert 0.0 < theta < 2e-3


def test_fx_theta_on_a_dated_contract_crossing_a_fixing_records_it():
    """A dated fixing between the valuation date and the theta date is fixed at the held spot, not refused."""
    age = FIX_DAYS[0] - 1
    product = _fx_dated(age)
    theta = FxRangeAccrualAnalyticalEngine().calculate_greeks(product, _fx_env(T0 + timedelta(days=age)))["theta"]
    assert 0.0 < theta < 2e-3


# ---------------------------------------------------------------------------
# A numeric contract booked mid-life: its maturity is what is LEFT, so it has to say how long the contract is
# ---------------------------------------------------------------------------

_PAST = [-0.25, -0.1, 0.25, 0.5]


def test_a_numeric_equity_contract_booked_mid_life_without_a_tenor_is_refused():
    records = [RangeAccrualObservationRecord(observation_time=t, observed_in_range=True if t <= 0 else None)
               for t in _PAST]
    with pytest.raises(ValidationError, match="tenor"):
        RangeAccrualOption(initial_price=100.0, range_config=_eq_config(), maturity=0.5, observation_records=records)


def test_a_numeric_fx_contract_booked_mid_life_without_a_tenor_is_refused():
    with pytest.raises(ValidationError, match="tenor"):
        FxRangeAccrualOption(notional=100.0, range_config=_fx_config(), currency_pair=CurrencyPair("EUR", "USD"),
                             maturity=0.5, observation_records=[
                                 FxRangeAccrualObservationRecord(observation_time=t,
                                                                 observed_in_range=True if t <= 0 else None)
                                 for t in _PAST])


def test_a_numeric_contract_booked_mid_life_with_its_tenor_pays_over_it():
    records = [RangeAccrualObservationRecord(observation_time=t, observed_in_range=True if t <= 0 else None)
               for t in _PAST]
    product = RangeAccrualOption(initial_price=100.0, range_config=_eq_config(), maturity=0.5, tenor=1.0,
                                 observation_records=records)
    assert product.get_year_fraction(_eq_env()) == 1.0


def test_a_shifted_numeric_contract_is_not_refused_when_validated_again():
    """``time_shift`` makes fixings negative and banks the elapsed time: the contract still knows its length."""
    aged = _eq_numeric()
    aged.time_shift(0.25, T0 + timedelta(days=91), _eq_env())
    aged.validate()
    assert aged.get_year_fraction(_eq_env()) == pytest.approx(1.0, rel=1e-12)
