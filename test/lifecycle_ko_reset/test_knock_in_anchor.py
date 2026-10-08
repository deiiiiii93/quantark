"""The KO-reset tracker adopts the persisted knock-in anchor (accrual time spec §5.1, §5.2).

The anchor is the knock-in record whose test fired, at its contractual fixing, resolved once at the knock-in and
stored beside ``ki_date``; ``ki_date`` (the close) stays the reset's timing. A knocked-in REBASED contract's ABSOLUTE
copy embeds the anchor in each converted record's (still relative) period and in its ``post`` terminal.
"""
import pickle
from datetime import timedelta

import pytest

from quantark.asset.equity.lifecycle import AutocallableLifecycleState, LifecycleEventType as E
from quantark.asset.equity.product.option.accrual import (
    KnockInAnchor,
    contract_tenor,
    ko_accrual_factor,
    knock_in_anchor_for,
)
from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.calendar import calculate_year_fraction
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import AccrualEndType
from quantark.util.enum.option_enums import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from accrual_time.conftest import T0, dates, ko_reset, times

from .conftest import env
from .test_tracker_rules import D, day, kr, run, tracker


def _knocked_in_on_day_95():
    """Knock-in records due on days 91 and 93; the run's first close is day 95 and both fire on it."""
    t = tracker(kr(M.REBASED, pre=(91, 182), ki=(91, 93), post=(91, 365), annualized=True))
    assert run(t, {95: 70.0}) == [(95, E.KNOCK_IN, None)]
    return t


def test_the_anchor_is_the_record_that_fired_at_its_contractual_fixing():
    t = _knocked_in_on_day_95()
    anchor = t.lifecycle.knock_in_anchor
    assert isinstance(anchor, KnockInAnchor)
    assert (anchor.source_index, anchor.fixing, anchor.origin) == (0, 91 * D, 0.0)
    assert t.lifecycle.ki_date == day(95).to_pydatetime()            # the close stays the reset's timing


def test_the_converted_copy_accrues_from_the_fixing_and_is_timed_from_the_close():
    t = _knocked_in_on_day_95()
    contract = t.contract()
    first = contract.post_barrier_config.ko_observation_schedule.records[0]
    assert first.observation_time == pytest.approx((95 + 91) * D)      # timing: the knock-in close + offset
    assert ko_accrual_factor(contract, first) == pytest.approx((91 + 91) * D, abs=1e-15)   # cash: fixing + offset
    assert first.accrual.relative and first.accrual.anchor == t.lifecycle.knock_in_anchor
    assert contract_tenor(contract, None, "post") == pytest.approx((91 + 365) * D, abs=1e-15)


def test_the_anchor_survives_ageing_a_checkpoint_and_the_pricing_copy():
    t = _knocked_in_on_day_95()
    anchor = t.lifecycle.knock_in_anchor
    later = day(120)                                                    # both knock-in records are behind it
    copy = t.product_for_pricing(later, env(90.0, later))
    assert copy.barrier_config.ki_observation_schedule.records == []
    assert copy._otc_lifecycle_knock_in_anchor == anchor
    assert [r.accrual.anchor for r in copy.post_barrier_config.ko_observation_schedule.records] == [anchor, anchor]
    assert ko_accrual_factor(copy, copy.post_barrier_config.ko_observation_schedule.records[0]) == pytest.approx(
        182 * D, abs=1e-15)
    assert contract_tenor(copy, None, "post") == pytest.approx(456 * D, abs=1e-15)
    copy.validate()                                                     # a re-validated copy keeps the cash
    assert ko_accrual_factor(copy, copy.post_barrier_config.ko_observation_schedule.records[0]) == pytest.approx(
        182 * D, abs=1e-15)
    assert contract_tenor(copy, None, "post") == pytest.approx(456 * D, abs=1e-15)
    assert t.product_for_lifecycle()._otc_lifecycle_knock_in_anchor == anchor
    # a checkpoint round trip of this test's own in-process state (no untrusted data is loaded)
    resumed = tracker(kr(M.REBASED, pre=(91, 182), ki=(91, 93), post=(91, 365), annualized=True),
                      lifecycle=pickle.loads(pickle.dumps(t.lifecycle)))
    assert resumed.lifecycle.knock_in_anchor == anchor
    assert ko_accrual_factor(resumed.contract(), resumed.contract().post_barrier_config.ko_observation_schedule
                             .records[0]) == pytest.approx(182 * D, abs=1e-15)


def test_a_carried_rebased_knock_in_without_its_anchor_is_refused():
    product = kr(M.REBASED, post=(91, 365))
    setattr(product, "_otc_lifecycle_knocked_in", True)
    setattr(product, "_otc_lifecycle_knock_in_date", day(-10).to_pydatetime())
    with pytest.raises(ValidationError, match="anchor"):
        tracker(product)
    state = AutocallableLifecycleState(knocked_in=True, ki_date=day(-10).to_pydatetime())
    with pytest.raises(ValidationError, match="anchor"):
        tracker(kr(M.REBASED, post=(91, 365)), lifecycle=state)


def test_a_carried_rebased_knock_in_with_its_anchor_is_kept():
    product = kr(M.REBASED, post=(91, 365))
    anchor = knock_in_anchor_for(product.barrier_config.ki_observation_schedule.records[0], 0)
    setattr(product, "_otc_lifecycle_knocked_in", True)
    setattr(product, "_otc_lifecycle_knock_in_date", day(91).to_pydatetime())
    setattr(product, "_otc_lifecycle_knock_in_anchor", anchor)
    t = tracker(product)
    assert t.lifecycle.knock_in_anchor == anchor
    assert t.contract().post_barrier_config.ko_observation_schedule.records[0].accrual.anchor == anchor


def test_with_absolute_schedule_2_refuses_a_missing_anchor():
    product = kr(M.REBASED, post=(91, 365))
    with pytest.raises(ValidationError, match="anchor"):
        product.with_absolute_schedule_2(91 * D, anchor=None)


def test_a_non_accruing_rebased_contract_needs_no_anchor():
    """Dated, no initial_date, nothing annualized: the knock-in records carry NONE periods, so there is no anchor to
    resolve (knock_in_anchor_for gives None) and none is required."""
    p = ko_reset(M.REBASED, pre=dates(91, 182), ki=dates(91, 182, level=75.0),
                 accrual=AccrualConfig(is_annualized=False), maturity=(182 + 270) / 365)
    assert knock_in_anchor_for(p.barrier_config.ki_observation_schedule.records[0], 0) is None
    converted = p.with_absolute_schedule_2(91 / 365, anchor=None)
    assert ko_accrual_factor(converted, converted.post_barrier_config.ko_observation_schedule.records[0]) == 1.0


def _settlement_control():
    """Knock-in at 0.25 with a YEAR_FRACTION lag of 0.01; schedule-2 offsets 0.25 and 0.5, the last paid at 0.52;
    ACCRUAL_END_TYPE = SETTLEMENT (accrual time spec §5.2 and §4.4 controls)."""
    post = times(0.25, 0.5, level=100.0)
    post.records[1].settlement_time = 0.52
    return ko_reset(M.REBASED, post=post, accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT),
                    settlement_convention=SettlementConvention(lag=0.01, lag_unit=SettlementLagUnit.YEAR_FRACTION))


def test_the_converted_record_accrues_the_lag_once():
    p = _settlement_control()
    ki = p.barrier_config.ki_observation_schedule.records[0]
    assert ki.accrual.end == 0.25                                        # the knock-in accrues to its fixing
    converted = p.with_absolute_schedule_2(0.25, anchor=knock_in_anchor_for(ki, 0))
    paid = converted.post_barrier_config.ko_observation_schedule.records[1]
    assert ko_accrual_factor(converted, paid) == pytest.approx(0.77, abs=1e-15)


def test_the_converted_terminal_is_the_mc_routes():
    p = _settlement_control()
    anchor = knock_in_anchor_for(p.barrier_config.ki_observation_schedule.records[0], 0)
    converted = p.with_absolute_schedule_2(0.25, anchor=anchor)
    mc_route = p._resolve_post_contract_tenor(None, knock_in=anchor)    # SnowballMCEngine's per-path call
    assert contract_tenor(converted, None, "post") == mc_route == 0.75
    converted.validate()
    assert contract_tenor(converted, None, "post") == 0.75


def test_a_dated_knock_in_plus_a_quoted_offset_in_the_converted_copy():
    p = ko_reset(M.REBASED, pre=dates(91, 182), ki=dates(91, 182, level=75.0), post=times(0.25, 0.5, level=100.0),
                 initial_date=T0, maturity=(182 + 183) / 365)
    ki = p.barrier_config.ki_observation_schedule.records[0]
    converted = p.with_absolute_schedule_2(91 / 365, anchor=knock_in_anchor_for(ki, 0))
    first = converted.post_barrier_config.ko_observation_schedule.records[0]
    origin_to_fixing = calculate_year_fraction(T0, T0 + timedelta(days=91), DayCountConvention.ACT_365)
    assert ko_accrual_factor(converted, first) == pytest.approx(origin_to_fixing + 0.25, abs=1e-15)
    ninety_one_days = calculate_year_fraction(T0 + timedelta(days=91), T0 + timedelta(days=182),
                                              DayCountConvention.ACT_365)
    assert ko_accrual_factor(converted, first) != pytest.approx(origin_to_fixing + ninety_one_days, abs=1e-6)
