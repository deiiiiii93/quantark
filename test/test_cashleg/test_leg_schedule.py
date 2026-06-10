import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
