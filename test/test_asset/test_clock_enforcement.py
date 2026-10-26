"""The date / numeric clock rule enforced at construction (ship spec 2026-10-26: AC1, AC2, AC3).

AC1: a NUMERIC product given a date field is refused, naming the field (every concrete in-scope class).
AC2: a DATE product given a numeric time it needs, with no date for it, is refused, naming the field.
AC3: both forms complete: the date wins and the floats are dropped, so it prices bitwise like the dated product
alone; validation rebinds only the product's own attributes, never a shared record, schedule or config.
"""
import math
from copy import copy, deepcopy
from dataclasses import replace
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.product.option import (
    AccumulatorOption, AmericanOption, AsianOption, BarrierOption, DoubleBarrierOption, DoubleOneTouchOption,
    DoubleSharkfinOption, EuropeanVanillaOption, OneTouchOption, SingleSharkfinOption, create_ko_reset_snowball,
)
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord
from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.range_accrual_config import RangeAccrualConfig
from quantark.asset.equity.product.option.range_accrual_option import RangeAccrualOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxBarrierOption, FxDigitalOption, FxOneTouchOption, FxSharkfinOption, FxVanillaOption,
)
from quantark.asset.fx.product.option.fx_foreign_range_accrual_option import FxForeignRangeAccrualOption
from quantark.asset.fx.product.option.fx_quanto_digital_option import FxQuantoDigitalOption
from quantark.asset.fx.product.option.fx_quanto_range_accrual_option import FxQuantoRangeAccrualOption
from quantark.asset.fx.product.option.fx_quanto_vanilla_option import FxQuantoVanillaOption
from quantark.asset.fx.product.option.fx_range_accrual_config import FxRangeAccrualConfig
from quantark.asset.fx.product.option.fx_range_accrual_option import FxRangeAccrualOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment, PricingEnvironment
from quantark.util.enum import (
    AccumulatorKnockOutType, AsianStrikeType, AveragingType, BarrierDirection, BarrierType, DoubleBarrierType,
    FxBarrierType, ObservationType, OptionType, PostKOScheduleMode, TouchType,
)
from quantark.util.exceptions import ValidationError

T0 = datetime(2026, 1, 5)
EXPIRY = datetime(2027, 1, 5)
PAIR = CurrencyPair("EUR", "USD")
CALL, PUT = OptionType.CALL, OptionType.PUT


def _fx_ra_config():
    return FxRangeAccrualConfig(upper_barrier=1.25, lower_barrier=1.05, accrual_rate=0.05)


#: Every concrete in-scope class, numerically (``maturity``); ``kw`` adds or overrides fields.
NUMERIC = {
    "european": lambda **kw: EuropeanVanillaOption(strike=100.0, option_type=CALL, **{"maturity": 1.0, **kw}),
    "american": lambda **kw: AmericanOption(strike=100.0, option_type=PUT, **{"maturity": 1.0, **kw}),
    "digital": lambda **kw: CashOrNothingDigitalOption(strike=100.0, option_type=CALL, payout=10.0,
                                                       **{"maturity": 1.0, **kw}),
    "barrier": lambda **kw: BarrierOption(strike=100.0, option_type=PUT, barrier=80.0,
                                          barrier_type=BarrierType.DOWN_IN, **{"maturity": 1.0, **kw}),
    "double_barrier": lambda **kw: DoubleBarrierOption(strike=100.0, option_type=CALL, upper_barrier=120.0,
                                                       lower_barrier=80.0, barrier_type=DoubleBarrierType.KNOCK_OUT,
                                                       **{"maturity": 1.0, **kw}),
    "one_touch": lambda **kw: OneTouchOption(barrier=110.0, barrier_direction=BarrierDirection.UP,
                                             touch_type=TouchType.ONE_TOUCH, **{"maturity": 1.0, **kw}),
    "double_one_touch": lambda **kw: DoubleOneTouchOption(upper_barrier=110.0, lower_barrier=90.0,
                                                          **{"maturity": 1.0, **kw}),
    "single_sharkfin": lambda **kw: SingleSharkfinOption(
        strike=95.0, option_type=CALL, barrier=120.0,
        **{"maturity": 1.0, "observation_type": ObservationType.EXPIRY, **kw}),
    "double_sharkfin": lambda **kw: DoubleSharkfinOption(
        strike=100.0, option_type=CALL, lower_barrier=70.0, upper_barrier=130.0,
        **{"maturity": 1.0, "observation_type": ObservationType.EXPIRY, **kw}),
    "asian": lambda **kw: AsianOption(strike=100.0, option_type=CALL, asian_strike_type=AsianStrikeType.FIXED,
                                      averaging_type=AveragingType.ARITHMETIC, **{"maturity": 1.0, **kw}),
    "accumulator": lambda **kw: AccumulatorOption(strike=96.0, knock_out_barrier=110.0, daily_share_accumulation=1.0,
                                                  knock_out_type=AccumulatorKnockOutType.TERMINATION,
                                                  **{"maturity": 1.0, "observation_dates": [0.5, 1.0], **kw}),
    "range_accrual": lambda **kw: RangeAccrualOption(
        initial_price=100.0, range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0,
                                                             accrual_rate=0.05),
        **{"maturity": 1.0, "observation_times": [0.5, 1.0], **kw}),
    "snowball": lambda **kw: _with(create_standard_snowball(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=75.0),
                                   **kw),
    "phoenix": lambda **kw: _with(create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0,
                                                          coupon_barrier=80.0, coupon_rate=0.01,
                                                          num_observations=12), **kw),
    "ko_reset": lambda **kw: _with(create_ko_reset_snowball(initial_price=100.0, strike=100.0, maturity_pre=1.0,
                                                            maturity_post=2.0,
                                                            post_ko_mode=PostKOScheduleMode.ABSOLUTE,
                                                            ki_continuous=False), **kw),
    "fx_vanilla": lambda **kw: FxVanillaOption(strike=1.1, option_type=CALL, currency_pair=PAIR,
                                               notional_foreign=1.0, **{"maturity": 1.0, **kw}),
    "fx_quanto_vanilla": lambda **kw: FxQuantoVanillaOption(strike=1.1, option_type=CALL, quanto_fx_rate=1.0,
                                                            currency_pair=PAIR, notional_foreign=1.0,
                                                            **{"maturity": 1.0, **kw}),
    "fx_digital": lambda **kw: FxDigitalOption(strike=1.1, option_type=CALL, payout=1.0, currency_pair=PAIR,
                                               **{"maturity": 1.0, **kw}),
    "fx_quanto_digital": lambda **kw: FxQuantoDigitalOption(strike=1.1, option_type=CALL, payout=1.0,
                                                            quanto_fx_rate=1.0, currency_pair=PAIR,
                                                            **{"maturity": 1.0, **kw}),
    "fx_one_touch": lambda **kw: FxOneTouchOption(barrier=1.3, is_up=True, currency_pair=PAIR,
                                                  **{"maturity": 1.0, **kw}),
    "fx_barrier": lambda **kw: FxBarrierOption(strike=1.1, barrier=1.3, is_up=True, knock_type=FxBarrierType.KNOCK_OUT,
                                               option_type=CALL, currency_pair=PAIR, **{"maturity": 1.0, **kw}),
    "fx_sharkfin": lambda **kw: FxSharkfinOption(strike=1.1, barrier=1.3, is_up=True, option_type=CALL,
                                                 currency_pair=PAIR, **{"maturity": 1.0, **kw}),
    "fx_range_accrual": lambda **kw: FxRangeAccrualOption(notional=1.0, range_config=_fx_ra_config(),
                                                          currency_pair=PAIR,
                                                          **{"maturity": 1.0, "observation_times": [0.5, 1.0], **kw}),
    "fx_foreign_range_accrual": lambda **kw: FxForeignRangeAccrualOption(
        notional=1.0, range_config=_fx_ra_config(), currency_pair=PAIR,
        **{"maturity": 1.0, "observation_times": [0.5, 1.0], **kw}),
    "fx_quanto_range_accrual": lambda **kw: FxQuantoRangeAccrualOption(
        notional=1.0, range_config=_fx_ra_config(), quanto_fx_rate=1.0, currency_pair=PAIR,
        **{"maturity": 1.0, "observation_times": [0.5, 1.0], **kw}),
}


def _with(product, **kw):
    """A helper-built product with ``kw`` set, re-validated (the codebase's copy idiom)."""
    for name, value in kw.items():
        setattr(product, name, value)
    product.validate()
    return product


def _date_field(kind):
    """A date field the class takes: the FX classes' delivery_date, the touch options' settlement_date, else
    initial_date."""
    if kind.startswith("fx_"):
        return "delivery_date"
    if kind in ("one_touch", "double_one_touch"):
        return "settlement_date"
    return "initial_date"


@pytest.mark.parametrize("kind", NUMERIC)
def test_ac1_a_numeric_product_with_a_date_field_is_refused(kind):
    field = _date_field(kind)
    value = EXPIRY if field != "initial_date" else T0
    with pytest.raises(ValidationError, match=f"numeric clock .*{field} is a date field"):
        NUMERIC[kind](**{field: value})


@pytest.mark.parametrize("kind", NUMERIC)
def test_every_class_constructs_on_the_numeric_clock(kind):
    NUMERIC[kind]()


# --- AC2: a numeric time a dated product needs -----------------------------------------------------------------------

def _dated_snowball(ko_records):
    config = BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                           ko_observation_schedule=ObservationSchedule(records=ko_records), ki_barrier=75.0,
                           ki_continuous=True)
    return SnowballOption(initial_price=100.0, strike=100.0, initial_date=T0, exercise_date=EXPIRY,
                          barrier_config=config)


DATED_WITH_A_NEEDED_FLOAT = {
    "snowball_time_only_record": (
        lambda: _dated_snowball([ObservationRecord(observation_date=datetime(2026, 7, 6), barrier=103.0),
                                 ObservationRecord(observation_time=1.0, barrier=103.0)]),
        "observation_time"),
    "barrier_legacy_float_list": (
        lambda: BarrierOption(strike=100.0, option_type=PUT, barrier=80.0, barrier_type=BarrierType.DOWN_IN,
                              exercise_date=EXPIRY, observation_type=ObservationType.DISCRETE,
                              observation_dates=[0.5, 1.0]),
        "observation_dates"),
    "one_touch_legacy_float_list": (
        lambda: OneTouchOption(barrier=110.0, barrier_direction=BarrierDirection.UP, exercise_date=EXPIRY,
                               observation_type=ObservationType.DISCRETE, observation_dates=[0.5, 1.0]),
        "observation_dates"),
    "asian_float_times": (
        lambda: NUMERIC["asian"](maturity=None, exercise_date=EXPIRY, observation_times=[0.5, 1.0]),
        "observation_times"),
    "accumulator_float_list": (
        lambda: NUMERIC["accumulator"](maturity=None, exercise_date=EXPIRY),
        "observation_dates"),
    "accumulator_float_fixings": (
        lambda: NUMERIC["accumulator"](maturity=None, exercise_date=EXPIRY,
                                       observation_schedule=ObservationSchedule(records=[
                                           ObservationRecord(observation_date=datetime(2026, 7, 6))]),
                                       observation_dates=None, past_observations=[(-0.1, 100.0)]),
        "past_observations"),
    "range_accrual_float_times": (
        lambda: NUMERIC["range_accrual"](maturity=None, exercise_date=EXPIRY),
        "observation_times"),
    "fx_delivery_alone": (lambda: NUMERIC["fx_vanilla"](maturity=None, expiry_date=EXPIRY, delivery=1.01),
                          "delivery"),
    "fx_barrier_float_times": (
        lambda: NUMERIC["fx_barrier"](maturity=None, expiry_date=EXPIRY, monitoring=ObservationType.DISCRETE,
                                      observation_times=[0.5, 1.0]),
        "observation_times"),
    "fx_sharkfin_float_times": (
        lambda: NUMERIC["fx_sharkfin"](maturity=None, expiry_date=EXPIRY, monitoring=ObservationType.DISCRETE,
                                       observation_times=[0.5, 1.0]),
        "observation_times"),
    "fx_range_accrual_float_times": (
        lambda: NUMERIC["fx_range_accrual"](maturity=None, expiry_date=EXPIRY), "observation_times"),
}


def _dated(kind, **kw):
    """``kind`` booked on the date clock (an ``exercise_date`` / ``expiry_date`` in place of ``maturity``)."""
    anchor = {"expiry_date": EXPIRY} if kind.startswith("fx_") else {"exercise_date": EXPIRY}
    return NUMERIC[kind](maturity=None, **anchor, **kw)


def _dated_autocallable(kind, **kw):
    """A helper-built Snowball / Phoenix / KO-reset (numeric, legacy float schedules) re-dated: the float schedules
    are now times the dated contract needs with no date for them."""
    return NUMERIC[kind](maturity=None, initial_date=T0, exercise_date=EXPIRY, **kw)


def _time_only_records(kind):
    """A dated autocallable whose knock-out schedule is two dated records and one time-only record."""
    records = [ObservationRecord(observation_date=datetime(2026, 7, 6), barrier=103.0),
               ObservationRecord(observation_time=1.0, barrier=103.0)]
    product = NUMERIC[kind]()
    config = replace(product.barrier_config, ko_observation_dates=None, ko_barrier=103.0,
                     ko_observation_schedule=ObservationSchedule(records=records))
    return _dated_autocallable(kind, barrier_config=config)


def _fx_dated_delivery_alone(kind):
    if "range_accrual" in kind:
        from quantark.asset.fx.product.option.fx_range_accrual_config import FxRangeAccrualObservationRecord
        records = [FxRangeAccrualObservationRecord(observation_date=d) for d in (datetime(2026, 7, 6), EXPIRY)]
        return _dated(kind, observation_times=None, observation_records=records, delivery=1.01)
    return _dated(kind, delivery=1.01)


def _discrete(**kw):
    return dict(observation_type=ObservationType.DISCRETE, observation_dates=[0.5, 1.0], **kw)


DATED_WITH_A_NEEDED_FLOAT.update({
    # legacy float lists, on every class that books them
    "double_barrier_legacy_float_list": (lambda: _dated("double_barrier", **_discrete()), "observation_dates"),
    "single_sharkfin_legacy_float_list": (lambda: _dated("single_sharkfin", **_discrete()), "observation_dates"),
    "double_sharkfin_legacy_float_list": (lambda: _dated("double_sharkfin", **_discrete()), "observation_dates"),
    "double_one_touch_legacy_float_list": (lambda: _dated("double_one_touch", **_discrete()), "observation_dates"),
    "snowball_barrier_config_float_list": (lambda: _dated_autocallable("snowball"), "observation_"),
    "phoenix_barrier_config_float_list": (lambda: _dated_autocallable("phoenix"), "observation_"),
    "ko_reset_barrier_config_float_list": (lambda: _dated_autocallable("ko_reset"), "observation_"),
    # a record with only observation_time, on every class with records
    "phoenix_time_only_record": (lambda: _time_only_records("phoenix"), "observation_time"),
    "ko_reset_time_only_record": (lambda: _time_only_records("ko_reset"), "observation_time"),
    "asian_time_only_record": (
        lambda: _dated("asian", observation_records=[AsianObservationRecord(observation_date=datetime(2026, 7, 6)),
                                                     AsianObservationRecord(observation_time=1.0)]),
        "observation_time"),
    "range_accrual_time_only_record": (lambda: _range_accrual_time_only_record(), "observation_time"),
    # FX delivery without delivery_date, on every FX class
    **{f"{kind}_delivery_alone": ((lambda kind=kind: _fx_dated_delivery_alone(kind)), "delivery")
       for kind in NUMERIC if kind.startswith("fx_")},
})


def _range_accrual_time_only_record():
    from quantark.asset.equity.product.option.range_accrual_config import RangeAccrualObservationRecord
    records = [RangeAccrualObservationRecord(observation_date=datetime(2026, 7, 6)),
               RangeAccrualObservationRecord(observation_time=1.0)]
    return _dated("range_accrual", observation_times=None, observation_records=records)


@pytest.mark.parametrize("case", DATED_WITH_A_NEEDED_FLOAT)
def test_ac2_a_dated_product_needing_a_numeric_time_is_refused(case):
    build, field = DATED_WITH_A_NEEDED_FLOAT[case]
    with pytest.raises(ValidationError, match=f"date clock .*{field}"):
        build()


# --- AC3: both forms given, the date wins ---------------------------------------------------------------------------

def _eq_env():
    return PricingEnvironment(spot_quote=SpotQuote(spot=100.0), rate_curve=FlatRateCurve(rate=0.03),
                              vol_surface=FlatVolSurface(volatility=0.2),
                              div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=T0)


def _fx_env():
    return FxPricingEnvironment(valuation_date=T0, spot_quote=SpotQuote(spot=1.1),
                                domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
                                vol_surface=FlatVolSurface(volatility=0.1))


@pytest.mark.parametrize("dated, both", [
    (dict(exercise_date=EXPIRY), dict(exercise_date=EXPIRY, maturity=0.9)),
    (dict(maturity_date=EXPIRY), dict(maturity_date=EXPIRY, maturity=0.9)),
], ids=["exercise_date", "maturity_date"])
def test_ac3_an_equity_date_shadows_the_maturity(dated, both):
    alone = EuropeanVanillaOption(strike=100.0, option_type=CALL, **dated)
    shadowed = EuropeanVanillaOption(strike=100.0, option_type=CALL, **both)
    assert shadowed.maturity is None and shadowed.exercise_date == alone.exercise_date == EXPIRY
    assert BlackScholesEngine().price(shadowed, _eq_env()) == BlackScholesEngine().price(alone, _eq_env())


@pytest.mark.parametrize("dated, both", [
    (dict(expiry_date=EXPIRY), dict(expiry_date=EXPIRY, maturity=0.9)),
    (dict(expiry_date=EXPIRY, delivery_date=datetime(2027, 1, 7)),
     dict(expiry_date=EXPIRY, delivery_date=datetime(2027, 1, 7), maturity=0.9, delivery=0.95)),
], ids=["expiry_date", "delivery_date"])
def test_ac3_an_fx_date_shadows_the_year_fraction(dated, both):
    alone = FxVanillaOption(strike=1.1, option_type=CALL, currency_pair=PAIR, notional_foreign=1.0, **dated)
    shadowed = FxVanillaOption(strike=1.1, option_type=CALL, currency_pair=PAIR, notional_foreign=1.0, **both)
    assert shadowed.maturity is None and shadowed.delivery is None
    assert GarmanKohlhagenEngine().price(shadowed, _fx_env()) == GarmanKohlhagenEngine().price(alone, _fx_env())


def test_ac3_validation_never_mutates_a_shared_record_or_config():
    records = [ObservationRecord(observation_date=datetime(2026, 7, 6), barrier=103.0),
               ObservationRecord(observation_date=EXPIRY, barrier=103.0)]
    note = _dated_snowball(records)
    config, schedule = note.barrier_config, note.barrier_config.ko_observation_schedule
    before = [replace(r) for r in schedule.records]
    shallow = copy(note)                            # QUAD V2 validates a shallow copy
    shallow.maturity = 0.9                          # a shadowed float on the copy only
    shallow.validate()
    assert shallow.maturity is None and note.maturity is None
    assert note.barrier_config is config and config.ko_observation_schedule is schedule
    assert schedule.records == before


def test_ac3_a_dated_theta_landing_record_stays_put():
    """Theta's keep_landing writes observation_time = 0.0 beside the record's date: the pair is left in place (the
    record time is not a violation and not cleared), and QUAD V2 prices the aged copy."""
    records = [ObservationRecord(observation_date=datetime(2026, 7, 6), barrier=103.0),
               ObservationRecord(observation_date=EXPIRY, barrier=103.0)]
    note = _dated_snowball(records)
    env = _eq_env()
    env.valuation_date = datetime(2026, 7, 5)
    aged = copy(note)
    aged.time_shift(1 / 365, datetime(2026, 7, 6), env, keep_landing=True)
    aged.validate()
    landing = aged.barrier_config.ko_observation_schedule.records[0]
    assert landing.observation_date == datetime(2026, 7, 6) and landing.observation_time == 0.0
    assert math.isfinite(float(SnowballQuadEngineV2().price(aged, env)))


# --- AC4: dated theta is unchanged by enforcement (QUAD V2, MC, PDE) -----------------------------------------------

def _theta_engines():
    from quantark.asset.equity.engine.mc import SnowballMCEngine
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.param import MCParams, PDEParams
    return {
        "quad_v2": SnowballQuadEngineV2,
        "mc": lambda: SnowballMCEngine(params=MCParams(num_paths=2_000, seed=11)),
        "pde": lambda: SnowballPDESolver(PDEParams()),
    }


@pytest.mark.parametrize("engine", ["quad_v2", "mc", "pde"])
def test_ac4_dated_theta_is_bitwise_unchanged_by_enforcement(engine, monkeypatch):
    """Theta on a dated note valued the day before a knock-out date: its bumped copy lands a record on the bumped
    date (observation_time 0.0 beside the date) and is re-validated. Enforcement must not move it: theta with the
    clock rule wired equals theta with ``enforce_clock`` a no-op, bit for bit."""
    from quantark.asset.equity.product.option import base_equity_option, snowball_option
    from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator

    records = [ObservationRecord(observation_date=d, barrier=103.0)
               for d in (datetime(2026, 4, 6), datetime(2026, 7, 6), datetime(2026, 10, 5), EXPIRY)]
    env = _eq_env()
    env.valuation_date = datetime(2026, 4, 5)

    def theta():
        note = _dated_snowball([replace(r) for r in records])
        return GreeksCalculator().calculate_numerical_theta(note, deepcopy(env), _theta_engines()[engine]())

    enforced = theta()
    for module in (base_equity_option, snowball_option):
        monkeypatch.setattr(module, "enforce_clock", lambda product: None)
    assert theta() == enforced and math.isfinite(enforced)
