"""KO-reset validity: the exercise date covers schedule 2 (owner ruling O1 2026-11-27, "Refuse such notes"; AC7).

A knocked-in KO-reset observes schedule 2 up to its last observation and, if it has not knocked out, is a put to the
PRODUCT'S EXERCISE DATE (run 9, D1). A note whose exercise precedes its latest possible schedule-2 end would settle
the put before the schedule it follows ends, which no contract means, so it is refused at construction:

- ABSOLUTE: the exercise date (numeric: ``maturity``) must not precede schedule 2's last booked observation;
- REBASED numeric: ``maturity`` must not precede the last knock-in observation plus schedule 2's last offset (a knock-in
  on that observation ends latest); the dated REBASED floor is ``_validate_dated_rebased_end`` (pre-existing);
- equality is allowed (numeric: round-off is one instant, the contract clock's ``same_instant``).

``create_ko_reset_snowball`` builds valid notes: a numeric REBASED note's maturity covers its latest rebased end.
"""
from datetime import datetime

import pytest

from quantark.asset.equity.product.option import create_ko_reset_snowball
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

S1 = (0.25, 0.5)
START = datetime(2026, 1, 6)
DATED_S1 = (datetime(2026, 4, 6), datetime(2026, 7, 6))


def _times(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=float(t), barrier=level) for t in times])


def _dates(dates, level):
    return ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=level) for d in dates])


def numeric(mode, post, maturity, *, ki="discrete", ki_times=S1):
    if ki == "continuous":
        ki_kw = dict(ki_barrier=80.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    else:
        ki_kw = dict(ki_barrier=80.0, ki_observation_type=ObservationType.DISCRETE,
                     ki_observation_schedule=_times(ki_times, 80.0))
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_times(S1, 103.0), **ki_kw),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_times(post, 95.0)),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=mode,
    )


def dated(post, exercise_date, *, ki="discrete"):
    if ki == "continuous":
        ki_kw = dict(ki_barrier=80.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    else:
        ki_kw = dict(ki_barrier=80.0, ki_observation_type=ObservationType.DISCRETE,
                     ki_observation_schedule=_dates(DATED_S1, 80.0))
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, initial_date=START, exercise_date=exercise_date,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_dates(DATED_S1, 103.0), **ki_kw),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_dates(post, 95.0)),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False),
        post_ko_mode=M.ABSOLUTE,
    )


# --- ABSOLUTE: exercise >= schedule 2's last booked observation ---------------------------------------------------


@pytest.mark.parametrize("ki", ("discrete", "continuous"))
def test_numeric_absolute_maturity_before_schedule_two_end_is_refused_naming_the_times(ki):
    with pytest.raises(ValidationError, match=r"ABSOLUTE.*maturity 0\.75.*schedule 2's last observation 1\.0"):
        numeric(M.ABSOLUTE, (0.5, 0.75, 1.0), 0.75, ki=ki)


@pytest.mark.parametrize("ki", ("discrete", "continuous"))
def test_numeric_absolute_maturity_on_or_after_schedule_two_end_is_allowed(ki):
    numeric(M.ABSOLUTE, (0.5, 0.75, 1.0), 1.0, ki=ki)
    numeric(M.ABSOLUTE, (0.5, 0.75, 1.0), 1.25, ki=ki)


def test_numeric_absolute_round_off_is_one_instant():
    numeric(M.ABSOLUTE, (0.5, 0.75, 1.0), 1.0 - 1e-13)


@pytest.mark.parametrize("ki", ("discrete", "continuous"))
def test_dated_absolute_exercise_before_schedule_two_end_is_refused_naming_the_dates(ki):
    with pytest.raises(ValidationError, match=r"ABSOLUTE.*exercise_date 2026-09-06.*2026-10-06"):
        dated((datetime(2026, 7, 6), datetime(2026, 10, 6)), datetime(2026, 9, 6), ki=ki)


@pytest.mark.parametrize("ki", ("discrete", "continuous"))
def test_dated_absolute_exercise_on_or_after_schedule_two_end_is_allowed(ki):
    dated((datetime(2026, 7, 6), datetime(2026, 10, 6)), datetime(2026, 10, 6), ki=ki)
    dated((datetime(2026, 7, 6), datetime(2026, 10, 6)), datetime(2026, 12, 7), ki=ki)


# --- REBASED numeric: maturity >= last knock-in observation + schedule 2's last offset ------------------------------


def test_numeric_rebased_maturity_before_the_latest_rebased_end_is_refused_naming_the_terms():
    """Last knock-in 0.5 + last offset 0.5 = 1.0 > maturity 0.75: a path knocking in on 0.5 ends at 1.0."""
    with pytest.raises(ValidationError, match=r"REBASED.*maturity 0\.75.*0\.5.*0\.5.*1\.0"):
        numeric(M.REBASED, (0.25, 0.5), 0.75)


def test_numeric_rebased_reads_the_last_knock_in_observation():
    """The first knock-in (0.25 + 0.5 = 0.75) is covered by maturity 0.75; the last one (0.5 + 0.5) is not."""
    with pytest.raises(ValidationError, match="REBASED"):
        numeric(M.REBASED, (0.25, 0.5), 0.75, ki_times=(0.25, 0.5))
    numeric(M.REBASED, (0.25, 0.5), 0.75, ki_times=(0.25,))


def test_numeric_rebased_boundary_equality_and_round_off_are_allowed():
    numeric(M.REBASED, (0.25, 0.5), 1.0)
    numeric(M.REBASED, (0.25, 0.5), 1.0 - 1e-13)
    numeric(M.REBASED, (0.25, 0.5), 1.5)


# --- the helper builds valid notes ----------------------------------------------------------------------------------


def test_the_helper_builds_a_numeric_rebased_note_whose_maturity_covers_the_latest_rebased_end():
    note = create_ko_reset_snowball(initial_price=100.0, strike=100.0, maturity_pre=1.0, maturity_post=0.5,
                                    ki_barrier=80.0, ki_frequency="quarterly", post_frequency="quarterly",
                                    ki_observation_dates=[0.5, 1.0], post_ko_mode=M.REBASED)
    assert note.maturity == 1.0 + 0.5
    note.validate()


def test_the_helper_keeps_the_absolute_maturity():
    note = create_ko_reset_snowball(initial_price=100.0, strike=100.0, maturity_pre=1.0, maturity_post=2.0,
                                    ki_barrier=80.0, post_ko_mode=M.ABSOLUTE)
    assert note.maturity == 2.0


# --- the tracker's day grid keeps the floor (snapping) -------------------------------------------------------------


def _rebased_on_days(ki_days, offset_days, maturity_days):
    d = 1 / 365
    ki_times = tuple(k * d for k in ki_days)
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity_days * d,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_times(ki_times, 103.0), ki_barrier=80.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_times(ki_times, 80.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_times((offset_days * d,), 95.0)),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=M.REBASED,
    )


def test_the_snapped_maturity_of_a_boundary_rebased_note_takes_the_rounded_days_back():
    """Booked on the boundary (364.6 + 273.6 = 638.2 days): the day grid rounds the maturity to 638 but the latest end
    to 365 + 274 = 639, so the snapped maturity moves to 639 and the snapped contract stays valid."""
    from quantark.asset.equity.lifecycle.snapping import snap_contract

    note = _rebased_on_days((91.0, 200.0, 364.6), 273.6, 364.6 + 273.6)
    snapped = snap_contract(note)
    assert snapped.maturity == 639 / 365
    snapped.validate()


def test_a_snapped_rebased_note_whose_maturity_covers_the_end_keeps_it():
    from quantark.asset.equity.lifecycle.snapping import snap_contract

    note = _rebased_on_days((91.0, 200.0, 364.6), 273.6, 700.4)
    assert snap_contract(note).maturity == 700 / 365
