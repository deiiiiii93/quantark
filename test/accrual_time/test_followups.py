"""Follow-ups to the accrual-time final review (M1-M5) and its two "declined to judge" questions."""
import math
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
import pytest

from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.util.enum import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import T0, dates, env, ko_reset, phoenix, snowball, times


# -- M2: a REBASED knocked-in tenor hangs from the knock-in too -------------------------------------------------------

def test_a_rebased_annualized_knock_in_tenor_needs_dated_knock_ins_to_have_an_initial_date():
    """Annualized KI on a REBASED contract reads its knock-in's accrual (spec §4.4): refused at construction (R5),
    not at V1 pricing. settlement_date lets the legacy _validate_maturity_parameters rule through."""
    with pytest.raises(ValidationError, match="R5"):
        ko_reset(M.REBASED, pre=dates(91, 182), ki=dates(91, 182, level=75.0),
                 accrual=AccrualConfig(is_annualized=False, is_annualized_ki=True), maturity=(182 + 270) / 365,
                 settlement_date=T0 + timedelta(days=452))


# -- M4: detaching a schedule copies it whole ---------------------------------------------------------------------

@dataclass
class _TaggedSchedule(ObservationSchedule):
    tag: str = "untagged"


def test_detaching_a_schedule_keeps_every_field_and_its_type():
    source = _TaggedSchedule(records=[ObservationRecord(observation_time=t, barrier=103.0) for t in (1 / 12, 2 / 12)],
                             tag="desk")
    p = snowball(ko=source, maturity=2 / 12)
    detached = p.barrier_config.ko_observation_schedule
    assert type(detached) is _TaggedSchedule and detached.tag == "desk" and detached is not source
    assert all(a is not b for a, b in zip(detached.records, source.records))
    assert all(r.accrual is None for r in source.records)


# -- M3: one place takes the aged-out prefix off the positional factor lists -------------------------------------

def test_ko_reset_ageing_takes_schedule_two_factors_off_through_accrual_config_shifted(monkeypatch):
    calls = []
    original = AccrualConfig.shifted

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("dropped_post", 0))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AccrualConfig, "shifted", spy)
    p = ko_reset(M.ABSOLUTE, accrual=AccrualConfig(post_accrual_factors=[0.1, 0.2, 0.3]))
    p.time_shift(0.3, T0 + timedelta(days=110), env())
    assert 1 in calls
    assert p.accrual_config.post_accrual_factors == [0.2, 0.3]


# -- M5: aged Phoenix memory needs the missed periods' starts ----------------------------------------------------

def test_quad_v2_aged_memory_refuses_a_missed_coupon_whose_period_start_is_unknown():
    """A missed coupon kept in the schedule at a negative time, before the contract's time origin (no accrued
    offset banked): its period has no start, so its arrears are unknown. Missing contract terms are a ValidationError
    (a known missed period prices, below), and the message says it is the aged memory that needs them."""
    from quantark.asset.equity.engine.quad import PhoenixQuadEngineV2
    from quantark.asset.equity.lifecycle import AutocallableLifecycleState
    p = phoenix(ko=times(-1 / 12, 1 / 12, 2 / 12), maturity=2 / 12)
    assert p.barrier_config.ko_observation_schedule.records[0].accrual.start is None
    state = AutocallableLifecycleState(coupon_memory_count=1)
    with pytest.raises(ValidationError, match="aged Phoenix memory needs the missed periods' starts"):
        PhoenixQuadEngineV2().price(p, env(), lifecycle_state=state)


def test_quad_v2_aged_memory_pays_a_missed_coupon_whose_period_is_known():
    from quantark.asset.equity.engine.quad import PhoenixQuadEngineV2
    from quantark.asset.equity.lifecycle import AutocallableLifecycleState
    p = phoenix(ko=times(-1 / 12, 1 / 12, 2 / 12), maturity=2 / 12,
                accrual=AccrualConfig(accrued_offset=2 / 12))
    assert p.barrier_config.ko_observation_schedule.records[0].accrual.start == 0.0
    state = AutocallableLifecycleState(coupon_memory_count=1)
    assert math.isfinite(float(PhoenixQuadEngineV2().price(p, env(), lifecycle_state=state)))


# -- M1: a knock-in anchor's source_index is the contract's own knock-in record ----------------------------------

D = 1 / 365.0


def _tracker_on_two_knock_ins():
    from quantark.asset.equity.lifecycle.autocallable import AutocallableLifecycleTracker
    product = snowball(ko=times(200 * D, 300 * D, level=1e6), ki=times(91 * D, 93 * D, level=75.0), maturity=300 * D)
    return AutocallableLifecycleTracker(product=product, quantity=1.0, start_date=pd.Timestamp(T0))


def test_the_tracker_anchors_a_later_knock_in_at_its_contract_index():
    tracker = _tracker_on_two_knock_ins()
    tracker.lifecycle.observed_ki_indices.add(0)        # the first knock-in test passed without a hit
    day = pd.Timestamp(T0 + timedelta(days=95))
    tracker.observe(day, tracker.product_for_lifecycle(), env(day.to_pydatetime(), spot=70.0), 70.0)
    assert tracker.lifecycle.knock_in_anchor.source_index == 1
    assert tracker.lifecycle.knock_in_anchor.fixing == 93 * D


def test_the_tracker_refuses_to_observe_an_aged_knock_in_schedule():
    """An aged pricing copy has dropped its past knock-in records: its indices are not the contract's, so neither
    observed_ki_indices nor the anchor's source_index could address the contract's records."""
    tracker = _tracker_on_two_knock_ins()
    aged_on = pd.Timestamp(T0 + timedelta(days=92))
    aged = tracker.product_for_pricing(aged_on, env(aged_on.to_pydatetime()))
    assert len(aged.barrier_config.ki_observation_schedule.records) == 1
    day = pd.Timestamp(T0 + timedelta(days=95))
    with pytest.raises(ValidationError, match="product_for_lifecycle"):
        tracker.observe(day, aged, env(day.to_pydatetime(), spot=70.0), 70.0)


def test_mc_anchors_each_knock_in_column_at_its_priced_schedule_index(monkeypatch):
    """Pin (not a RED case: the in-horizon knock-ins are always a prefix of the time-ordered schedule, so the column
    index already equalled the schedule index): MC hands each anchor its record's priced-schedule index."""
    import quantark.asset.equity.engine.mc.snowball_mc_engine as mc
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
    from quantark.asset.equity.param import MCParams
    seen = []
    original = mc.knock_in_anchor_for

    def spy(record, index):
        seen.append((index, record.accrual.end))
        return original(record, index)

    monkeypatch.setattr(mc, "knock_in_anchor_for", spy)
    p = ko_reset(M.REBASED)
    SnowballMCEngine(params=MCParams(seed=3, num_paths=256)).price(p, env(spot=80.0))
    ki = p.barrier_config.ki_observation_schedule.records
    assert seen and set(seen) == {(i, r.accrual.end) for i, r in enumerate(ki)}


# -- declined (b): QUAD V2 prices without rebuilding the caller's product ----------------------------------------

def _identities(p):
    schedules = [p.barrier_config.ko_observation_schedule, p.barrier_config.ki_observation_schedule]
    post = getattr(p, "post_barrier_config", None)
    if post is not None:
        schedules.append(post.ko_observation_schedule)
    records = [r for s in schedules if s is not None for r in s.records]
    terminals = p._terminal_accrual
    return ([id(p.barrier_config), id(post)] + [id(s) for s in schedules] + [id(r) for r in records]
            + [id(r.accrual) for r in records] + [id(terminals)] + [id(v) for v in terminals.values()])


@pytest.mark.parametrize("build", [
    lambda: snowball(ki=times(1 / 12, 2 / 12, 3 / 12, level=75.0)),
    lambda: phoenix(ki=times(1 / 12, 2 / 12, 3 / 12, level=75.0)),
    lambda: ko_reset(M.ABSOLUTE),
], ids=["snowball", "phoenix", "ko_reset"])
def test_quad_v2_pricing_leaves_the_callers_product_untouched(build):
    from copy import deepcopy
    from quantark.asset.equity.engine.quad import (
        KOResetSnowballQuadEngineV2, PhoenixQuadEngineV2, SnowballQuadEngineV2,
    )
    from quantark.pnlexplain.equity.fingerprints import contract_fingerprint
    p = build()
    before, fingerprint, state = _identities(p), contract_fingerprint(p), deepcopy(vars(p))
    engine = {"SnowballOption": SnowballQuadEngineV2, "PhoenixOption": PhoenixQuadEngineV2,
              "KnockOutResetSnowballOption": KOResetSnowballQuadEngineV2}[type(p).__name__]()
    reference = float(engine.price(deepcopy(p), env(spot=90.0)))
    assert float(engine.price(p, env(spot=90.0))) == reference
    engine.price_components(p, env(spot=95.0))
    assert _identities(p) == before
    assert contract_fingerprint(p) == fingerprint and sorted(vars(p)) == sorted(state)
