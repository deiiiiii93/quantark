"""The aged product must describe the SAME contract, restricted to what is left.

Every other accuracy check in this package compares one solver against
another, or against a reference whose deal is itself read off the aged
product.  A mistake in the aging would move both legs together and cancel,
so it has to be checked here, on the schedule alone, with no pricing at
all: the original contract's observations are converted to absolute dates
from the start date, the aged contract's from its own valuation date, and
the two are compared.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.config import CacheConfig, GateConfig
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.pricing.repricing import RepricingPricer
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

from .conftest import SPOT, pde_engine_config, short_snowball

START = pd.Timestamp("2024-01-02")
#: Long enough that observations fall on, before and after the probed dates.
MATURITY_DAYS = 40
KO_DAYS = (10, 20, 30, 40)
KI_DAYS = tuple(range(1, 41))


def _product():
    return short_snowball(maturity_days=MATURITY_DAYS, ko_days=KO_DAYS, ki_days=KI_DAYS)


def _pricer(product):
    return RepricingPricer(
        product, engine_config=pde_engine_config(), start_date=START, underlying="CSI1000",
        cache=StateCache(CacheConfig(memory_bytes=4_000_000)),
        gate=GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0),
    )


def _env(date):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=SPOT, asset_name="CSI1000"),
        vol_surface=FlatVolSurface(volatility=0.25), rate_curve=FlatRateCurve(rate=0.02),
        valuation_date=pd.Timestamp(date).to_pydatetime(),
    )


def _absolute(profile, anchor):
    """Observations as (date, barrier, payoff, settlement date), anchored absolutely."""
    rows = []
    for t, b, p, s in zip(profile["observation_times"], profile["barriers"],
                          profile["payoffs"], profile["settlement_times"]):
        d = pd.Timestamp(anchor) + timedelta(days=round(float(t) * 365.0))
        sd = None if s is None else pd.Timestamp(anchor) + timedelta(days=round(float(s) * 365.0))
        rows.append((d.normalize(), b, p, None if sd is None else sd.normalize()))
    return rows


def _matches(got, want):
    """Same dates exactly; barriers and payoffs to a relative ulp.

    The accrual is recomputed from a shifted time, so the last bit of a
    payoff moves; a date or a barrier has no such excuse.
    """
    if [r[0] for r in got] != [r[0] for r in want]:
        return False
    for a, w in zip(got, want):
        for x, y in zip(a[1:], w[1:]):
            if x is None or y is None:
                if x is not y:
                    return False
            elif isinstance(x, pd.Timestamp):
                if x != y:
                    return False
            elif not np.isclose(float(x), float(y), rtol=1e-12, atol=0.0):
                return False
    return True


PROBE_DAYS = (0, 1, 9, 10, 11, 20, 25, 30, 39)


@pytest.mark.parametrize("knocked_in", [False, True])
@pytest.mark.parametrize("kind", ["ko", "ki"])
def test_the_aged_contract_keeps_every_future_observation_on_its_own_date(kind, knocked_in):
    product = _product()
    pricer = _pricer(product)
    base = _absolute(getattr(product, f"get_{kind}_observation_profile")(_env(START)), START)
    assert base, "the fixture must have observations of this kind"
    for d in PROBE_DAYS:
        date = (START + timedelta(days=d)).normalize()
        aged = pricer.aged_product(date, knocked_in=knocked_in)
        got = _absolute(getattr(aged, f"get_{kind}_observation_profile")(_env(date)), date)
        want = [r for r in base if r[0] > date]
        assert _matches(got, want), f"{kind} schedule differs at day {d}"


def test_an_observation_on_the_valuation_date_is_not_carried_into_the_aged_contract():
    """The run steps the lifecycle BEFORE it prices, so that observation has
    already been consumed; carrying it would double-count the event."""
    product = _product()
    pricer = _pricer(product)
    base = _absolute(product.get_ko_observation_profile(_env(START)), START)
    on_obs = (START + timedelta(days=KO_DAYS[0])).normalize()
    assert on_obs in {r[0] for r in base}, "this test needs a date that IS an observation"
    aged = pricer.aged_product(on_obs, knocked_in=False)
    got = _absolute(aged.get_ko_observation_profile(_env(on_obs)), on_obs)
    assert on_obs not in {r[0] for r in got}
    assert len(got) == len(KO_DAYS) - 1


def test_the_aged_maturity_is_the_original_less_the_elapsed_calendar_time():
    product = _product()
    pricer = _pricer(product)
    base_tau = float(product.get_maturity(_env(START)))
    for d in PROBE_DAYS:
        date = (START + timedelta(days=d)).normalize()
        aged = pricer.aged_product(date, knocked_in=False)
        expected = base_tau - d / 365.0
        assert float(aged.get_maturity(_env(date))) == pytest.approx(expected, abs=1e-12)


def test_the_coupon_accrual_stays_anchored_at_inception():
    """A snowball's knock-out payoff accrues from the trade date.  Aging that
    restarted the accrual at the valuation date would leave every surviving
    observation paying the first observation's coupon."""
    product = _product()
    pricer = _pricer(product)
    base = _absolute(product.get_ko_observation_profile(_env(START)), START)
    first_payoff = float(base[0][2])
    date = (START + timedelta(days=25)).normalize()
    aged = pricer.aged_product(date, knocked_in=False)
    got = _absolute(aged.get_ko_observation_profile(_env(date)), date)
    survivor = got[0]
    original = [r for r in base if r[0] > date][0]
    assert float(survivor[2]) == pytest.approx(float(original[2]), rel=1e-12)
    # Strip the principal: the day-30 observation's COUPON is three times the
    # day-10 one.  A restarted accrual would make it the day-10 coupon again.
    principal = product.initial_price * product.contract_multiplier
    assert (float(survivor[2]) - principal) / (first_payoff - principal) == pytest.approx(
        KO_DAYS[2] / KO_DAYS[0], rel=1e-9)


def test_the_aged_contract_records_the_lifecycle_state_it_was_asked_for():
    product = _product()
    pricer = _pricer(product)
    date = (START + timedelta(days=15)).normalize()
    for flag in (False, True):
        aged = pricer.aged_product(date, knocked_in=flag)
        assert bool(getattr(aged, "_otc_lifecycle_knocked_in")) is flag


def test_an_unanchored_contract_ages_without_losing_its_coupon():
    """A time-based KO schedule with no ``initial_date`` measures its
    accrual from the product's own time origin: at inception the accrual
    factor of a coupon IS its observation time.

    Ageing moves that origin forward, so the elapsed period is banked in
    the accrual rather than dropped -- otherwise every surviving coupon
    would shrink by exactly the time that has passed, on a contract that
    is right on its trade date and wrong from the first day onward.
    ``test/test_aged_accrual.py`` states the same invariant at the product
    level and across the autocallable family; this checks the simulation's
    own ageing path carries it.
    """
    anchorless = short_snowball(maturity_days=MATURITY_DAYS, ko_days=KO_DAYS, ki_days=KI_DAYS)
    anchorless.initial_date = None
    pricer = _pricer(anchorless)
    base = _absolute(anchorless.get_ko_observation_profile(_env(START)), START)
    for d in PROBE_DAYS:
        date = (START + timedelta(days=d)).normalize()
        aged = pricer.aged_product(date, knocked_in=False)
        got = _absolute(aged.get_ko_observation_profile(_env(date)), date)
        want = [r for r in base if r[0] > date]
        assert _matches(got, want), f"unanchored KO schedule differs at day {d}"


def test_one_exhausted_schedule_does_not_un_age_the_other():
    """A snowball whose knock-in observations have all passed while
    knock-out observations remain is a live contract.

    ``BarrierConfig.time_shift`` used to report ``dropped_all`` when EITHER
    schedule emptied, and the aging then threw the whole shifted config
    away: the contract came back with its ORIGINAL observation times
    against a maturity that had kept decaying, so an observation that had
    already happened was alive again and one of the survivors sat beyond
    the remaining life.
    """
    product = short_snowball(maturity_days=6, ko_days=(2, 5), ki_days=(1, 3))
    pricer = _pricer(product)
    base = _absolute(product.get_ko_observation_profile(_env(START)), START)
    for d in (3, 4):
        date = (START + timedelta(days=d)).normalize()
        aged = pricer.aged_product(date, knocked_in=False)
        e = _env(date)
        got = _absolute(aged.get_ko_observation_profile(e), date)
        want = [r for r in base if r[0] > date]
        assert [r[0] for r in got] == [r[0] for r in want], (
            f"day {d}: aged KO dates {[str(r[0].date()) for r in got]} "
            f"against survivors {[str(r[0].date()) for r in want]}"
        )
        remaining = float(aged.get_maturity(e))
        latest = max(aged.get_ko_observation_profile(e)["observation_times"])
        assert latest <= remaining + 1e-12, (
            f"day {d}: an observation at {latest * 365:.1f}d on a contract with "
            f"{remaining * 365:.1f}d left"
        )


def test_a_contract_whose_observations_have_all_passed_keeps_an_empty_schedule():
    """The last day of the fixture: both schedules are spent and the
    contract is its terminal payoff.

    An emptied schedule says exactly that.  Reverting to the original one
    would put every observation back, and on a longer contract the early
    ones land INSIDE the remaining life -- a knock-out that happened
    months ago firing again next week, at its original barrier and coupon.
    """
    product = short_snowball(maturity_days=6, ko_days=(2, 5), ki_days=(1, 3))
    pricer = _pricer(product)
    date = (START + timedelta(days=5)).normalize()
    aged = pricer.aged_product(date, knocked_in=False)
    e = _env(date)
    assert aged.get_ko_observation_profile(e)["observation_times"] == []
    assert aged.get_ki_observation_profile(e)["observation_times"] == []
    assert float(aged.get_maturity(e)) == pytest.approx(1.0 / 365.0, abs=1e-12)
