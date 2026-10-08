"""Follow-ups to the accrual-time final review (M1-M5) and its two "declined to judge" questions."""
import math
from dataclasses import dataclass
from datetime import timedelta

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
