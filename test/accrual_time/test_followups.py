"""Follow-ups to the accrual-time final review (M1-M5) and its two "declined to judge" questions."""
from dataclasses import dataclass
from datetime import timedelta

import pytest

from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.util.enum import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import T0, dates, ko_reset, snowball


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
