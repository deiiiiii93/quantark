"""Cash reads periods only: invariance, precedence, the two legs, §3 regressions (spec §5, §8)."""
from dataclasses import replace
from datetime import timedelta

import pytest

from quantark.asset.equity.product.option.accrual import (
    AccrualPeriod, contract_tenor, coupon_period_fraction, knock_in_anchor_for, ko_accrual_factor,
)
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, PayoffConfig
from quantark.util.calendar.day_counter import DayCountConvention as DC
from quantark.util.enum import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import T0, dates, env, ko_reset, phoenix, rebuilt, snowball, times


def ko_cash(p, e=None):
    return [r.payoff for r in p.resolve_ko_observations(e or env())]


@pytest.mark.parametrize("step", [12, 52, 252])
@pytest.mark.parametrize("make", [snowball, phoenix], ids=["snowball", "phoenix"])
def test_cash_is_invariant_under_ageing_and_snapping(make, step):
    ts = [k / step for k in range(1, 4)]
    p = make(ko=times(*ts), maturity=ts[-1])
    base = ko_cash(p)
    snapped = make(ko=times(*ts), maturity=ts[-1])
    for r in snapped.barrier_config.ko_observation_schedule.records:
        r.observation_time = round(r.observation_time * 365) / 365
    assert ko_cash(snapped) == base
    aged = make(ko=times(*ts), maturity=ts[-1])
    aged.time_shift(0.5 / step, T0 + timedelta(days=1), env())
    assert ko_cash(aged) == base[-len(ko_cash(aged)):]


def test_explicit_factor_wins_over_the_flag():
    p = snowball(ko=times(0.25), accrual=AccrualConfig(accrual_factors=[0.4], is_annualized_ko=False))
    assert ko_cash(p) == [100.0 + 100.0 * 0.12 * 0.4]


def test_phoenix_two_legs():
    p = phoenix(coupon=CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.12, fixed_coupon_year_fraction=1 / 12))
    recs = p.barrier_config.ko_observation_schedule.records
    assert [100 * 0.12 * ko_accrual_factor(p, r) for r in recs] == pytest.approx([1.0, 2.0, 3.0])
    assert [p.get_coupon_payoff(i) for i in range(3)] == pytest.approx([1.0, 1.0, 1.0])


def test_numeric_with_initial_date_accrues_the_quoted_fraction():
    p = snowball(ko=times(0.25), maturity=0.25, initial_date=T0)
    assert ko_cash(p, env(T0 + timedelta(days=30))) == [100.0 + 100.0 * 0.12 * 0.25]


def test_phoenix_get_ko_payoff_keeps_accrued_offset():
    p = phoenix(ko=times(0.2), maturity=0.2, accrual=AccrualConfig(accrued_offset=0.3, first_period_start=0.25))
    assert p.get_ko_payoff(80.0, 0) == pytest.approx(100.0 + 100.0 * 0.12 * 0.5 + 100.0 * 0.12 * 0.25)


def test_aged_dated_phoenix_first_remaining_period_starts_at_the_previous_record():
    p = phoenix(ko=dates(30, 60, 90), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=90))
    p.time_shift(35 / 365, T0 + timedelta(days=35), env())
    assert coupon_period_fraction(p, p.barrier_config.ko_observation_schedule.records[0]) == 30 / 365


def test_tenors():
    assert contract_tenor(snowball(tenor=1.0)) == 1.0
    p = ko_reset(M.REBASED, post=times(0.25, 0.5, level=100.0))   # final offset 0.5 (spec round 1 control)
    with pytest.raises(ValidationError, match="anchor"):
        contract_tenor(p, None, "post")
    ki = p.barrier_config.ki_observation_schedule.records
    assert [contract_tenor(p, None, "post", knock_in=knock_in_anchor_for(r, i)) for i, r in enumerate(ki)] == [
        0.75, 1.0]


def test_rebased_offset_on_an_aged_contract():
    """Review round 2: elapsed 0.3, remaining knock-in 0.2 (anchor 0.5), offset 0.25 -> 0.75."""
    p = ko_reset(M.REBASED, pre=times(0.2), ki=times(0.2, level=75.0), post=times(0.25, level=100.0),
                 accrual=AccrualConfig(accrued_offset=0.3), maturity=0.45)
    anchor = knock_in_anchor_for(p.barrier_config.ki_observation_schedule.records[0], 0)
    post = p.post_barrier_config.ko_observation_schedule.records[0]
    assert ko_accrual_factor(p, post, knock_in=anchor) == pytest.approx(0.75, abs=0.0)


def test_embedded_anchor_terminal_matches_the_mc_route():
    """Round 3 P1-3: knock-in 0.25, final observation offset 0.5, payment offset 0.52 under SETTLEMENT -> 0.75."""
    from quantark.util.enum import AccrualEndType
    post = times(0.25, 0.5, level=100.0)
    post.records[1].settlement_time = 0.52
    p = ko_reset(M.REBASED, post=post, accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT))
    anchor = knock_in_anchor_for(p.barrier_config.ki_observation_schedule.records[0], 0)
    embedded = replace(p._terminal_accrual["post"], anchor=anchor)
    assert contract_tenor(p, None, "post", knock_in=anchor) == 0.75
    from quantark.asset.equity.product.option.accrual import cumulative_accrual
    assert cumulative_accrual(embedded) == 0.75
    paid = p.post_barrier_config.ko_observation_schedule.records[1]
    assert ko_accrual_factor(p, paid, knock_in=anchor) == pytest.approx(0.77, abs=1e-15)


def test_non_accruing_dated_contract_pays_as_today():
    p = snowball(ko=dates(30), maturity=None, exercise_date=T0 + timedelta(days=30),
                 accrual=AccrualConfig(is_annualized=False))
    assert ko_cash(p) == [100.0 + 12.0]
    with pytest.raises(ValidationError, match="no accrual terms"):
        from quantark.asset.equity.product.option.accrual import cumulative_accrual
        cumulative_accrual(p.barrier_config.ko_observation_schedule.records[0].accrual)


def test_changed_coupon_day_count_after_construction_raises():
    p = phoenix(ko=dates(30, 60), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60))
    p.coupon_config = replace(p.coupon_config, day_count_convention=DC.ACT_360)
    with pytest.raises(ValidationError, match="validate"):
        p.get_coupon_payoff(0)


def test_replace_after_ageing_keeps_record_cash():
    p = snowball()
    before = ko_cash(p)
    p.time_shift(1 / 24, T0 + timedelta(days=15), env())
    q = rebuilt(p)
    assert ko_cash(q) == before[-len(ko_cash(q)):]
    assert contract_tenor(q) == pytest.approx(contract_tenor(snowball()), rel=0, abs=1e-15)


def test_v0_unannualized_rebate_never_reads_the_tenor():
    p = snowball(ko=dates(30), maturity=None, exercise_date=T0 + timedelta(days=30),
                 accrual=AccrualConfig(is_annualized=False), payoff_config=PayoffConfig(rebate_rate=0.05))
    assert p.get_maturity_payoff_v0(100.0) == 105.0
