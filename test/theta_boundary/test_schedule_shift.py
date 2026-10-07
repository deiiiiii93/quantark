"""ObservationSchedule.time_shift at the bumped instant (spec 2026-10-07 theta boundary, R1, 3.1, 3.3)."""
from datetime import datetime

import pytest

from pde_routes.cases import env_const
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.exceptions import ValidationError

D = 1 / 252


def numeric(times):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=120.0) for t in times])


def dated(dates):
    return ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=120.0) for d in dates])


def test_landing_record_kept_at_zero_only_when_asked():
    sch = numeric([D, 2 * D])
    kept = sch.time_shift(D, keep_landing=True)
    assert kept.records[0].observation_time == 0.0
    assert kept.records[1].observation_time == pytest.approx(D)
    dropped = sch.time_shift(D)
    assert [r.observation_time for r in dropped.records] == [pytest.approx(D)]


def test_landing_residue_snaps_to_zero():
    sch = numeric([0.1 + 0.2])
    assert sch.time_shift(0.3, keep_landing=True).records[0].observation_time == 0.0   # residue 5.6e-17


def test_past_records_dropped_and_t0_gone():
    sch = numeric([0.0, D, 2 * D])                          # 0.0 is the as-of date's own record
    out = sch.time_shift(D, keep_landing=True)
    assert [r.observation_time for r in out.records] == [0.0, pytest.approx(D)]


def test_multi_day_bump_drops_between_keeps_landing():
    out = numeric([D, 2 * D, 3 * D, 4 * D]).time_shift(3 * D, keep_landing=True)
    assert [r.observation_time for r in out.records] == [0.0, pytest.approx(D)]


def test_dated_landing_ignores_time_of_day():
    bumped = datetime(2026, 1, 2, 15, 0)
    sch = dated([datetime(2026, 1, 2), datetime(2026, 1, 5)])
    out = sch.time_shift(D, bumped, keep_landing=True)
    assert out.records[0].observation_date.date() == bumped.date()
    assert out.records[0].observation_time == 0.0
    assert out.records[1].observation_date == datetime(2026, 1, 5)
    env = env_const()
    env.valuation_date = bumped
    assert out.resolve(env, default_barrier=120.0)[0].observation_time == 0.0
    assert [r.observation_date for r in sch.time_shift(D, bumped).records] == [datetime(2026, 1, 5)]


def test_emptied_schedule_is_marked_not_none():
    out = numeric([D]).time_shift(2 * D, keep_landing=True)
    assert out is not None and out.records == [] and out.monitoring_ended
    out.validate()                                          # legal: monitoring has ended


def test_hand_emptied_schedule_is_refused():
    with pytest.raises(ValidationError):
        ObservationSchedule(records=[]).validate()


def test_without_keep_landing_the_old_rule_is_unchanged():
    """Lifecycle aging must stay byte-identical (R2): the old comparisons, not the theta ones."""
    later_same_day = dated([datetime(2026, 1, 2, 16, 0), datetime(2026, 1, 5)])
    out = later_same_day.time_shift(D, datetime(2026, 1, 2, 15, 0))
    assert [r.observation_date for r in out.records] == [datetime(2026, 1, 2, 16, 0), datetime(2026, 1, 5)]
    residue = numeric([0.1 + 0.2]).time_shift(0.3)                 # +5.6e-17: still ahead under the old rule
    assert len(residue.records) == 1 and residue.records[0].observation_time > 0.0
