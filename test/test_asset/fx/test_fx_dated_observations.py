"""Dated discrete monitoring for the FX barrier and sharkfin (clock rule 2026-10-26, Decision 6b).

A product is all dates or all floats. A dated (``expiry_date``) discrete FX barrier or sharkfin books its fixings as
``observation_dates``; the product resolves them to year fractions from the valuation date on the environment's day
count (``observation_times_at``), and the engine prices those. So the dated product prices bitwise like the numeric
one booked with the same times at the same valuation date. Both forms given: the dates win.

Mid-life (user, 2026-10-26: "need mid-life valuation"): ``observed_fixings`` records the realised spot of each fixing
on or before the valuation date. A recorded breach decides the contract (a knock-out keeps only a rebate not yet paid,
a knock-in is the vanilla); otherwise only the fixings ahead are simulated. A past fixing without a record is decided
at the current spot with a ``MissingFixingWarning`` (user ruling 2026-10-09), as the FX range accrual does.
"""
import math
from copy import deepcopy
from datetime import timedelta

import pytest

from quantark.asset.fx.engine.mc import FxBarrierMCEngine, FxSharkfinMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxBarrierOption, FxSharkfinOption
from quantark.param import SpotQuote
from quantark.util.enum import FxBarrierType, ObservationType, OptionType
from quantark.util.exceptions import ValidationError
from test_asset.fx.test_fx_barrier_mc import _VAL, _smile_env

DAYS = (91, 182, 273, 365)
DATES = [_VAL + timedelta(days=d) for d in DAYS]
EXPIRY = DATES[-1]


def _barrier(**kw):
    kw.setdefault("knock_type", FxBarrierType.KNOCK_OUT)
    return FxBarrierOption(strike=1.20, barrier=1.35, is_up=True, option_type=OptionType.CALL,
                           currency_pair=CurrencyPair("EUR", "USD"), monitoring=ObservationType.DISCRETE, **kw)


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


# --- mid-life: fixings on or before the valuation date (user: "need mid-life valuation") -------------------------

def _env_on(date, spot=1.20):
    env = deepcopy(_smile_env())
    env.valuation_date = date
    env.spot_quote = SpotQuote(spot=spot)
    return env


def _mc(kind):
    return ENGINES[kind](params=FxMCParams(num_paths=20_000, time_steps=40, seed=7))


MID = DATES[1] + timedelta(days=10)                      # two fixings behind, two ahead
BELOW = {DATES[0]: 1.25, DATES[1]: 1.30}                 # recorded, no breach of 1.35
HIT = {DATES[0]: 1.25, DATES[1]: 1.40}                   # the second fixing breached


@pytest.mark.parametrize("kind", BUILDERS)
def test_a_mid_life_product_prices_its_remaining_fixings_like_its_numeric_twin(kind):
    env = _env_on(MID)
    dated = BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES, observed_fixings=BELOW)
    hit, ahead = dated.monitoring_state(env)
    assert hit is None and ahead == [(d - MID).days / 365.0 for d in DATES[2:]]
    numeric = BUILDERS[kind](maturity=dated.get_maturity(env), observation_times=ahead)
    assert _mc(kind).price(dated, env) == _mc(kind).price(numeric, env)


@pytest.mark.parametrize("at_hit", [True, False])
def test_a_knock_out_recorded_before_today_leaves_only_an_unpaid_rebate(at_hit):
    env = _env_on(MID)
    dead = _barrier(expiry_date=EXPIRY, observation_dates=DATES, observed_fixings=HIT, rebate=0.02,
                    rebate_at_hit=at_hit)
    expected = 0.0 if at_hit else 0.02 * env.get_domestic_df(dead.get_delivery(env))
    assert _mc("barrier").price(dead, env) == pytest.approx(expected, abs=1e-15)


def test_a_knock_out_on_todays_fixing_pays_its_rebate_at_hit_now():
    env = _env_on(DATES[1])
    dead = _barrier(expiry_date=EXPIRY, observation_dates=DATES, observed_fixings=HIT, rebate=0.02,
                    rebate_at_hit=True)
    assert _mc("barrier").price(dead, env) == pytest.approx(0.02, abs=1e-15)


def test_a_knock_in_recorded_before_today_is_the_vanilla():
    from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
    from quantark.asset.fx.product.option import FxVanillaOption
    env = _env_on(MID)
    knocked = _barrier(expiry_date=EXPIRY, observation_dates=DATES, observed_fixings=HIT,
                       knock_type=FxBarrierType.KNOCK_IN)
    vanilla = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, expiry_date=EXPIRY, notional_foreign=1.0)
    engine = FxBarrierMCEngine(params=FxMCParams(num_paths=200_000, time_steps=40, seed=7))
    assert engine.price(knocked, env) == pytest.approx(GarmanKohlhagenEngine().price(vanilla, env), rel=2e-2)


def test_a_sharkfin_knocked_out_before_today_keeps_only_its_rebate():
    env = _env_on(MID)
    dead = _sharkfin(expiry_date=EXPIRY, observation_dates=DATES, observed_fixings=HIT)
    assert _mc("sharkfin").price(dead, env) == pytest.approx(
        0.01 * env.get_domestic_df(dead.get_delivery(env)), abs=1e-15)


def test_a_missing_past_fixing_is_decided_at_the_current_spot_with_a_warning():
    from quantark.util.fixings import MissingFixingWarning
    product = _barrier(expiry_date=EXPIRY, observation_dates=DATES, observed_fixings={DATES[0]: 1.25})
    with pytest.warns(MissingFixingWarning, match="1 past fixing"):
        assert product.monitoring_state(_env_on(MID, spot=1.36))[0] is not None      # 1.36 breaches 1.35
    with pytest.warns(MissingFixingWarning):
        assert product.monitoring_state(_env_on(MID, spot=1.30))[0] is None


def test_every_fixing_behind_and_none_breached_leaves_the_terminal_payoff():
    env = _env_on(DATES[-1] - timedelta(days=5))
    product = _barrier(expiry_date=EXPIRY, observation_dates=DATES[:-1] + [EXPIRY - timedelta(days=10)],
                       observed_fixings={d: 1.25 for d in DATES[:-1] + [EXPIRY - timedelta(days=10)]})
    assert product.monitoring_state(env) == (None, [])
    assert math.isfinite(_mc("barrier").price(product, env))


@pytest.mark.parametrize("kind", BUILDERS)
def test_theta_records_the_fixings_it_steps_over_at_the_held_spot(kind):
    product = BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES, observed_fixings={DATES[0]: 1.25})
    product.record_passed_fixings(_env_on(DATES[1] + timedelta(days=1), spot=1.22))
    assert product.observed_fixings == {DATES[0]: 1.25, DATES[1]: 1.22}
    greeks = _mc(kind).calculate_greeks(
        BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES, observed_fixings={DATES[0]: 1.25}),
        _env_on(DATES[1] - timedelta(days=1)))
    assert math.isfinite(greeks["theta"])


@pytest.mark.parametrize("kind", BUILDERS)
def test_a_bump_greek_freezes_a_missing_past_fixing_at_the_base_spot(kind):
    product = BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES)
    frozen = product.with_missing_fixings_filled(_env_on(MID, spot=1.22), warn=False)
    assert frozen.observed_fixings == {DATES[0]: 1.22, DATES[1]: 1.22} and product.observed_fixings is None
    assert frozen.with_missing_fixings_filled(_env_on(MID), warn=False) is frozen


@pytest.mark.parametrize("kind", BUILDERS)
def test_observed_fixings_are_validated(kind):
    with pytest.raises(ValidationError, match="not an observation date"):
        BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES, observed_fixings={MID: 1.25})
    with pytest.raises(ValidationError, match="positive"):
        BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES, observed_fixings={DATES[0]: 0.0})
    with pytest.raises(ValidationError, match="not an observation time"):
        BUILDERS[kind](maturity=1.0, observation_times=[0.5, 1.0], observed_fixings={DATES[0]: 1.25})


@pytest.mark.parametrize("kind", BUILDERS)
def test_a_future_fixing_must_not_carry_an_outcome(kind):
    product = BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES, observed_fixings={DATES[3]: 1.25})
    with pytest.raises(ValidationError, match="future"):
        product.monitoring_state(_env_on(MID))


# --- the numeric clock ages its fixing times (user: "fix it", 2026-10-26) -------------------------------------------

@pytest.mark.parametrize("kind", BUILDERS)
def test_a_numeric_time_shift_moves_the_fixings_with_the_clock(kind):
    product = BUILDERS[kind](maturity=1.0, observation_times=[0.25, 0.5, 1.0], observed_fixings=None)
    product.time_shift(0.25)
    assert product.observation_times == [0.0, 0.25, 0.75] and product.maturity == 0.75
    product.record_passed_fixings(_env_on(_VAL, spot=1.22))         # the fixing landing today, at the held spot
    assert product.observed_fixings == {0.0: 1.22}


@pytest.mark.parametrize("kind", BUILDERS)
def test_numeric_theta_ages_its_fixings_like_its_dated_twin(kind):
    env = _smile_env()
    dated = BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES)
    numeric = BUILDERS[kind](maturity=dated.get_maturity(env), observation_times=dated.observation_times_at(env))
    theta = {form: _mc(kind).calculate_greeks(p, env)["theta"] for form, p in (("dated", dated), ("numeric", numeric))}
    assert theta["numeric"] == pytest.approx(theta["dated"], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("kind", BUILDERS)
def test_numeric_theta_steps_over_a_fixing_like_its_dated_twin(kind):
    env = _env_on(DATES[0] - timedelta(days=1))
    dated = BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES)
    numeric = BUILDERS[kind](maturity=dated.get_maturity(env), observation_times=dated.observation_times_at(env))
    theta = {form: _mc(kind).calculate_greeks(p, env)["theta"] for form, p in (("dated", dated), ("numeric", numeric))}
    assert theta["numeric"] == pytest.approx(theta["dated"], rel=1e-9, abs=1e-12)


def test_a_numeric_mid_life_booking_carries_its_past_fixings():
    env = _smile_env()
    dead = _barrier(maturity=0.75, observation_times=[-0.25, 0.25, 0.75], observed_fixings={-0.25: 1.40},
                    rebate=0.02)
    assert dead.monitoring_state(env) == (-0.25, [])
    assert _mc("barrier").price(dead, env) == pytest.approx(0.02 * env.get_domestic_df(0.75), abs=1e-15)
    alive = _barrier(maturity=0.75, observation_times=[-0.25, 0.25, 0.75], observed_fixings={-0.25: 1.30})
    assert alive.monitoring_state(env) == (None, [0.25, 0.75])


def test_a_numeric_past_time_without_its_fixing_is_refused():
    with pytest.raises(ValidationError, match="strictly positive"):
        _barrier(maturity=0.75, observation_times=[-0.25, 0.25, 0.75])


# --- CP2 review: the engines' expiry and inception shortcuts read the recorded fixings too --------------------------

def test_a_knock_out_recorded_earlier_is_dead_on_its_expiry_date():
    env = _env_on(EXPIRY, spot=1.30)
    dead = _barrier(expiry_date=EXPIRY, observation_dates=DATES, observed_fixings={**HIT, DATES[2]: 1.25})
    assert _mc("barrier").price(dead, env) == 0.0
    shark = _sharkfin(expiry_date=EXPIRY, observation_dates=DATES, observed_fixings={**HIT, DATES[2]: 1.25})
    assert _mc("sharkfin").price(shark, env) == pytest.approx(0.01 * env.get_domestic_df(shark.get_delivery(env)),
                                                              abs=1e-15)


def test_on_its_expiry_date_a_knock_in_reads_todays_recorded_fixing_not_the_live_spot():
    env = _env_on(EXPIRY, spot=1.36)
    fixings = {DATES[0]: 1.25, DATES[1]: 1.30, DATES[2]: 1.30, DATES[3]: 1.30}      # never breached 1.35
    knock_in = _barrier(expiry_date=EXPIRY, observation_dates=DATES, observed_fixings=fixings,
                        knock_type=FxBarrierType.KNOCK_IN)
    assert _mc("barrier").price(knock_in, env) == 0.0                                 # its rebate: none


@pytest.mark.parametrize("spot", [1.30, 1.36])
def test_a_rebate_paid_at_an_earlier_hit_is_not_paid_again_whatever_todays_spot(spot):
    dead = _barrier(expiry_date=EXPIRY, observation_dates=DATES, observed_fixings=HIT, rebate=0.02,
                    rebate_at_hit=True)
    assert _mc("barrier").price(dead, _env_on(MID, spot=spot)) == 0.0


@pytest.mark.parametrize("kind", BUILDERS)
def test_a_discrete_product_is_not_knocked_by_a_spot_between_its_fixings(kind):
    """Only a fixing observes a discrete barrier: a spot beyond it on a day without a fixing knocks nothing."""
    env = _env_on(MID, spot=1.36)
    product = BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES, observed_fixings=BELOW)
    hit, ahead = product.monitoring_state(env)
    assert hit is None and len(ahead) == 2
    numeric = BUILDERS[kind](maturity=product.get_maturity(env), observation_times=ahead)
    price = _mc(kind).price(product, env)
    assert price == _mc(kind).price(numeric, env) > 0.0
    knocked_now = 0.0 if kind == "barrier" else 0.01 * env.get_domestic_df(product.get_delivery(env))
    assert price != pytest.approx(knocked_now, abs=1e-12)                # not the inception-hit value


def test_a_missing_fixing_warns_once_per_pricing():
    from quantark.util.fixings import MissingFixingWarning
    import warnings as _warnings
    product = _barrier(expiry_date=EXPIRY, observation_dates=DATES, observed_fixings={DATES[0]: 1.25})
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _mc("barrier").price(product, _env_on(MID))
    assert sum(issubclass(w.category, MissingFixingWarning) for w in caught) == 1


@pytest.mark.parametrize("kind", BUILDERS)
def test_dated_fixing_times_need_an_environment(kind):
    with pytest.raises(ValidationError, match="FxPricingEnvironment required"):
        BUILDERS[kind](expiry_date=EXPIRY, observation_dates=DATES).observation_times_at()
