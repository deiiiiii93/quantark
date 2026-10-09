import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime

import numpy as np
import pytest

from quantark.cashleg.leg_schedule import LegSchedule
from quantark.util.exceptions import ValidationError


def test_simple_quarterly_schedule():
    sched = LegSchedule(
        period_starts=np.array([0.0, 0.25, 0.5, 0.75]),
        period_ends=np.array([0.25, 0.5, 0.75, 1.0]),
        payment_times=np.array([0.25, 0.5, 0.75, 1.0]),
    )
    assert len(sched.period_starts) == 4
    assert sched.last_period_end() == 1.0


def test_mismatched_array_lengths_rejected():
    with pytest.raises(ValidationError):
        LegSchedule(
            period_starts=np.array([0.0, 0.25]),
            period_ends=np.array([0.25, 0.5, 0.75]),
            payment_times=np.array([0.25, 0.5]),
        )


def test_period_end_before_start_rejected():
    with pytest.raises(ValidationError):
        LegSchedule(
            period_starts=np.array([0.5]),
            period_ends=np.array([0.25]),
            payment_times=np.array([0.5]),
        )


def test_validate_within_maturity_passes():
    sched = LegSchedule(
        period_starts=np.array([0.0]),
        period_ends=np.array([1.0]),
        payment_times=np.array([1.0]),
    )
    sched.validate_within_maturity(1.0)


def test_validate_within_maturity_rejects_overflow():
    sched = LegSchedule(
        period_starts=np.array([0.0]),
        period_ends=np.array([1.5]),
        payment_times=np.array([1.5]),
    )
    with pytest.raises(ValidationError, match="maturity"):
        sched.validate_within_maturity(1.0)


# ---------------------------------------------------------------------------
# Period dates: all-or-nothing, datetime, start < end (W4a, findings §10)
# ---------------------------------------------------------------------------

_STARTS = [datetime(2026, 1, 1), datetime(2026, 7, 1)]
_ENDS = [datetime(2026, 7, 1), datetime(2027, 1, 1)]


def _dated(**overrides):
    kwargs = dict(
        period_starts=np.array([0.0, 0.5]),
        period_ends=np.array([0.5, 1.0]),
        payment_times=np.array([0.5, 1.0]),
        period_start_dates=list(_STARTS),
        period_end_dates=list(_ENDS),
    )
    kwargs.update(overrides)
    return LegSchedule(**kwargs)


def test_dated_schedule_accepted_and_reports_dates():
    sched = _dated()
    assert sched.has_period_dates()
    assert not LegSchedule(
        period_starts=np.array([0.0]),
        period_ends=np.array([1.0]),
        payment_times=np.array([1.0]),
    ).has_period_dates()


def test_start_dates_without_end_dates_rejected():
    with pytest.raises(ValidationError, match="together"):
        _dated(period_end_dates=None)
    with pytest.raises(ValidationError, match="together"):
        _dated(period_start_dates=None)


def test_some_periods_dated_others_not_rejected():
    with pytest.raises(ValidationError, match="every period"):
        _dated(period_end_dates=[_ENDS[0], None])


def test_non_datetime_period_date_rejected():
    with pytest.raises(ValidationError, match="datetime"):
        _dated(period_start_dates=[_STARTS[0], "2026-07-01"])


def test_period_end_date_not_after_start_date_rejected():
    with pytest.raises(ValidationError, match="after"):
        _dated(period_end_dates=[datetime(2025, 12, 1), _ENDS[1]])  # reversed
    with pytest.raises(ValidationError, match="after"):
        _dated(period_end_dates=[_ENDS[0], _STARTS[1]])  # zero-length period


def test_period_date_list_length_mismatch_rejected():
    with pytest.raises(ValidationError, match="length"):
        _dated(period_end_dates=[_ENDS[0]])
