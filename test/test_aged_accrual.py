"""Ageing an autocallable must not shorten its coupons.

A time-based observation schedule measures its records from the product's
own time origin, so a contract with no ``initial_date`` is saying "my
accrual starts where my schedule starts": at inception the accrual factor
of a knock-out coupon IS its observation time.  ``time_shift`` moves that
origin forward and shrinks every observation time with it, so the elapsed
period has to be banked or every surviving coupon quietly loses exactly
that much -- a contract that is right on its trade date and wrong from the
first day onward.

The invariant is stated without pricing: the coupon a surviving
observation pays is the coupon the ORIGINAL contract promised for that
same point in the contract's life.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.product.option import (
    create_ko_reset_snowball,
    create_standard_phoenix,
    create_standard_snowball,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, PostKOScheduleMode

SPOT = 100.0
VALUATION = datetime(2024, 1, 2)
#: A quarter of a one-year contract: long enough that a lost quarter of
#: accrual is unmistakable, short enough to leave three survivors.
#:
#: Whole days, not round quarters.  The anchored contract recomputes its
#: accrual as a year fraction between two DATES, so a shift of 0.25 years
#: against a valuation date 91 days later would differ from the original
#: by 0.25 - 91/365 -- a property of the test's arithmetic, not of the
#: contract.  Measuring the shift in days removes it.
SHIFT = 91.0 / 365.0
KO_TIMES = [91.0 / 365.0, 182.0 / 365.0, 273.0 / 365.0, 1.0]


def _env(offset_years: float = 0.0) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=SPOT, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02),
        valuation_date=VALUATION + timedelta(days=round(offset_years * 365.0)),
    )


def _snowball(**kwargs):
    """A one-year snowball with quarterly knock-outs and NO ``initial_date``."""
    return create_standard_snowball(
        initial_price=SPOT, strike=SPOT, maturity=1.0, contract_multiplier=1.0,
        ko_barrier=1.03 * SPOT, ki_barrier=0.75 * SPOT, ko_rate=0.12,
        num_observations=len(KO_TIMES), ko_observation_dates=list(KO_TIMES),
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=[182.0 / 365.0, 1.0], include_principal=True, **kwargs,
    )


def _aged(product, shift=SHIFT):
    """``time_shift`` on a deep copy, the way every theta and ageing path does it."""
    aged = copy.deepcopy(product)
    env = _env(shift)
    aged.time_shift(shift, env.valuation_date, env)
    return aged


def _coupons(product, env):
    """(observation time, payoff) for every knock-out still ahead."""
    profile = product.get_ko_observation_profile(env)
    return list(zip(profile["observation_times"], profile["payoffs"]))


def test_an_unanchored_snowball_keeps_its_coupons_when_aged():
    product = _snowball()
    assert product.initial_date is None, "this test is about the unanchored contract"
    base = _coupons(product, _env())
    aged = _coupons(_aged(product), _env(SHIFT))

    survivors = [(t, p) for t, p in base if t > SHIFT]
    assert len(aged) == len(survivors) == 3
    for (t_aged, pay_aged), (t_base, pay_base) in zip(aged, survivors):
        assert t_aged == pytest.approx(t_base - SHIFT, abs=1e-12)
        assert pay_aged == pytest.approx(pay_base, rel=1e-12)


def test_the_lost_accrual_was_exactly_the_elapsed_period():
    """Names the size of the defect, so a regression cannot pass quietly."""
    product = _snowball()
    aged = _aged(product)
    principal = product.initial_price * product.contract_multiplier
    coupon = _coupons(aged, _env(SHIFT))[0][1] - principal
    # First survivor: the original 182-day observation, accruing 182 days.
    assert coupon == pytest.approx(principal * 0.12 * KO_TIMES[1], rel=1e-12)
    # What it would pay with the elapsed 91 days dropped.
    assert coupon != pytest.approx(principal * 0.12 * (KO_TIMES[1] - SHIFT), rel=1e-6)


def test_an_anchored_snowball_is_untouched_by_the_offset():
    """A contract with a real anchor already ages correctly and must not move."""
    product = _snowball()
    product.initial_date = VALUATION
    base = _coupons(product, _env())
    aged = _coupons(_aged(product), _env(SHIFT))
    survivors = [(t, p) for t, p in base if t > SHIFT]
    for (_, pay_aged), (_, pay_base) in zip(aged, survivors):
        assert pay_aged == pytest.approx(pay_base, rel=1e-12)


def test_successive_shifts_accumulate_like_one_shift_of_their_sum():
    """Theta bumps an already-aged contract, so the offset has to compound."""
    product = _snowball()
    half = 182.0 / 365.0
    once = _aged(product, half)
    twice = _aged(_aged(product, SHIFT), half - SHIFT)
    assert twice.accrual_config.accrued_offset == pytest.approx(half, abs=1e-12)
    a, b = _coupons(once, _env(half)), _coupons(twice, _env(half))
    assert len(a) == len(b) == 2
    for (t_a, p_a), (t_b, p_b) in zip(a, b):
        assert t_a == pytest.approx(t_b, abs=1e-12)
        assert p_a == pytest.approx(p_b, rel=1e-12)


def test_explicit_accrual_factors_outrank_the_offset():
    """An externally supplied factor IS the answer; nothing may be added to it."""
    from dataclasses import replace

    product = _snowball()
    factors = [0.3, 0.6, 0.9, 1.2]
    product.accrual_config = replace(product.accrual_config, accrual_factors=factors)
    aged = _aged(product)
    principal = product.initial_price * product.contract_multiplier
    coupons = [p - principal for _, p in _coupons(aged, _env(SHIFT))]
    # The shift drops the first record, so the survivors take factors 1..3.
    assert coupons == pytest.approx([principal * 0.12 * f for f in factors[1:]], rel=1e-12)


def test_an_unanchored_phoenix_keeps_its_coupons_when_aged():
    product = create_standard_phoenix(
        initial_price=SPOT, strike=SPOT, maturity=1.0, contract_multiplier=1.0,
        ko_barrier=1.03 * SPOT, ki_barrier=0.75 * SPOT, coupon_barrier=0.85 * SPOT,
        ko_rate=0.12, coupon_rate=0.08, num_observations=len(KO_TIMES),
        ko_observation_dates=list(KO_TIMES),
    )
    assert product.initial_date is None
    base = _coupons(product, _env())
    aged = _coupons(_aged(product), _env(SHIFT))
    survivors = [(t, p) for t, p in base if t > SHIFT]
    assert len(aged) == len(survivors)
    for (_, pay_aged), (_, pay_base) in zip(aged, survivors):
        assert pay_aged == pytest.approx(pay_base, rel=1e-12)


def test_an_unanchored_ko_reset_snowball_keeps_its_accrual_factor():
    """The KO-reset engines ask the product for the factor directly."""
    product = create_ko_reset_snowball(
        initial_price=SPOT, strike=SPOT, maturity_pre=1.0, maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE, ki_continuous=True,
    )
    assert product.initial_date is None
    records = product.barrier_config.ko_observation_schedule.records
    base = [
        product.compute_ko_accrual_factor(rec.observation_time, rec, _env())
        for rec in records
    ]
    aged = _aged(product)
    aged_records = aged.barrier_config.ko_observation_schedule.records
    got = [
        aged.compute_ko_accrual_factor(rec.observation_time, rec, _env(SHIFT))
        for rec in aged_records
    ]
    survivors = [f for f, rec in zip(base, records) if rec.observation_time > SHIFT]
    assert got == pytest.approx(survivors, rel=1e-12)


@pytest.mark.xfail(strict=True, reason="KNOWN DEFECT: the contract tenor ages with the maturity")
def test_an_annualized_rebate_keeps_its_contract_tenor_when_aged():
    """The rebate and the knock-in participation scale by the CONTRACT tenor
    -- inception to expiry -- which does not change as the contract ages.

    ``BaseEquityOption.get_tenor`` falls back to ``maturity`` when there is
    no explicit ``tenor`` and no ``exercise_date``, and ageing shrinks
    maturity, so a 10% annualized rebate on a one-year note is worth 5% of
    it halfway through.  Unlike the coupon accrual, ``initial_date`` does
    not cure this: ``get_tenor_end_date`` needs an ``exercise_date``.

    Left failing on purpose: the fix belongs to ``get_tenor``, which every
    equity option shares, and is a wider change than the accrual.
    """
    from dataclasses import replace

    product = _snowball()
    product.accrual_config = replace(product.accrual_config, is_annualized_rebate=True)
    base = product.get_contract_tenor(_env())
    aged = _aged(product, 182.0 / 365.0)
    assert aged.get_contract_tenor(_env(182.0 / 365.0)) == pytest.approx(base, rel=1e-12)


def test_the_offset_must_be_a_finite_non_negative_year_fraction():
    from dataclasses import replace

    cfg = _snowball().accrual_config
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="accrued_offset"):
            replace(cfg, accrued_offset=bad)
