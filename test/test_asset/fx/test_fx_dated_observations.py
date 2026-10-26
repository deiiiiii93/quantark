"""Dated discrete monitoring for the FX barrier and sharkfin (clock rule 2026-10-26, Decision 6b).

A product is all dates or all floats. A dated (``expiry_date``) discrete FX barrier or sharkfin books its fixings as
``observation_dates``; the product resolves them to year fractions from the valuation date on the environment's day
count (``observation_times_at``), and the engine prices those. So the dated product prices bitwise like the numeric
one booked with the same times at the same valuation date. Both forms given: the dates win.
"""
from datetime import timedelta

import pytest

from quantark.asset.fx.engine.mc import FxBarrierMCEngine, FxSharkfinMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxBarrierOption, FxSharkfinOption
from quantark.util.enum import FxBarrierType, ObservationType, OptionType
from quantark.util.exceptions import ValidationError
from test_asset.fx.test_fx_barrier_mc import _VAL, _smile_env

DAYS = (91, 182, 273, 365)
DATES = [_VAL + timedelta(days=d) for d in DAYS]
EXPIRY = DATES[-1]


def _barrier(**kw):
    return FxBarrierOption(strike=1.20, barrier=1.35, is_up=True, knock_type=FxBarrierType.KNOCK_OUT,
                           option_type=OptionType.CALL, currency_pair=CurrencyPair("EUR", "USD"),
                           monitoring=ObservationType.DISCRETE, **kw)


def _sharkfin(**kw):
    return FxSharkfinOption(strike=1.20, barrier=1.35, is_up=True, option_type=OptionType.CALL, ko_rebate=0.01,
                            currency_pair=CurrencyPair("EUR", "USD"), monitoring=ObservationType.DISCRETE, **kw)


BUILDERS = {"barrier": _barrier, "sharkfin": _sharkfin}
ENGINES = {"barrier": FxBarrierMCEngine, "sharkfin": FxSharkfinMCEngine}


@pytest.mark.parametrize("kind", BUILDERS)
def test_observation_dates_resolve_on_the_environment_day_count(kind):
    env = _smile_env()
    dated = BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES)
    assert dated.observation_times_at(env) == [d / 365.0 for d in DAYS]


@pytest.mark.parametrize("kind", BUILDERS)
def test_a_dated_discrete_product_prices_like_its_numeric_twin(kind):
    env = _smile_env()
    dated = BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES)
    numeric = BUILDERS[kind](maturity=dated.get_maturity(env), observation_times=dated.observation_times_at(env))
    engine = lambda: ENGINES[kind](params=FxMCParams(num_paths=20_000, time_steps=40, seed=7))  # noqa: E731
    assert engine().price(dated, env) == engine().price(numeric, env)


@pytest.mark.parametrize("kind", BUILDERS)
def test_a_numeric_product_reads_its_times_as_booked(kind):
    numeric = BUILDERS[kind](maturity=1.0, observation_times=[0.25, 0.5, 1.0])
    assert numeric.observation_times_at(_smile_env()) == [0.25, 0.5, 1.0]


@pytest.mark.parametrize("kind", BUILDERS)
def test_both_forms_given_the_dates_win(kind):
    both = BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES, observation_times=[0.1, 0.2, 0.3, 0.4])
    assert both.observation_times_at(_smile_env()) == [d / 365.0 for d in DAYS]


@pytest.mark.parametrize("kind", BUILDERS)
def test_observation_dates_are_validated(kind):
    with pytest.raises(ValidationError, match="ascending"):
        BUILDERS[kind](expiry_date=EXPIRY, observation_dates=[DATES[1], DATES[0]])
    with pytest.raises(ValidationError, match="unique"):
        BUILDERS[kind](expiry_date=EXPIRY, observation_dates=[DATES[0], DATES[0]])
    with pytest.raises(ValidationError, match="expiry"):
        BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES + [EXPIRY + timedelta(days=1)])
    with pytest.raises(ValidationError, match="DISCRETE"):
        BUILDERS[kind](expiry_date=EXPIRY)


@pytest.mark.parametrize("kind", BUILDERS)
def test_a_fixing_on_or_before_the_valuation_date_is_refused_as_a_numeric_time_would_be(kind):
    dated = BUILDERS[kind](expiry_date=EXPIRY, observation_dates=[_VAL] + DATES[1:])
    with pytest.raises(ValidationError, match="strictly positive"):
        dated.observation_times_at(_smile_env())
