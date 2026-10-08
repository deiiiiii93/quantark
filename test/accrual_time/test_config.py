"""Contract-level accrual terms (spec §4, §4.7)."""
import pytest

from quantark.asset.equity.product.option.accrual import AccrualKind, AccrualPeriod
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord
from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import AccrualEndType


def test_defaults_do_not_move_existing_contracts():
    c = AccrualConfig()
    assert c.accrual_end is AccrualEndType.OBSERVATION
    assert c.post_accrual_factors is None and c.accrual_calendar is None and c.accrual_bus_days_in_year is None
    assert not hasattr(c, "accrual_dates")


@pytest.mark.parametrize("kw", [
    dict(accrual_end="settlement"), dict(post_accrual_factors=(0.1,)), dict(post_accrual_factors=[-0.1]),
    dict(accrual_calendar="SSE"), dict(accrual_bus_days_in_year=0.0),
])
def test_bad_terms_are_refused(kw):
    with pytest.raises(ValueError):
        AccrualConfig(**kw)


def test_shifted_slices_both_positional_lists():
    c = AccrualConfig(accrual_factors=[0.1, 0.2, 0.3], post_accrual_factors=[1.0, 2.0])
    s = c.shifted(0.1, dropped_observations=1, dropped_post=1)
    assert s.accrual_factors == [0.2, 0.3] and s.post_accrual_factors == [2.0]


def test_terms_carry_a_calendar():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2027))
    assert AccrualConfig(accrual_calendar=cal, accrual_bus_days_in_year=244).accrual_bus_days_in_year == 244


def test_record_carries_an_accrual_period():
    p = AccrualPeriod(AccrualKind.TIME, origin=0.0, end=0.25)
    assert ObservationRecord(observation_time=0.25, accrual=p).accrual is p
    assert ObservationRecord(observation_time=0.25).accrual is None
