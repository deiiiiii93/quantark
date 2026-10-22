"""The calendar schedule must fire on exactly the tracker's own days."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker, CalendarSchedule
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.asset.equity.product.option import create_standard_snowball
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

SPOT = 100.0
DATES = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=8, freq="D"))


def _product():
    return create_standard_snowball(
        initial_price=SPOT, strike=SPOT, maturity=6.0 / 365.0, contract_multiplier=1.0,
        ko_barrier=103.0, ki_barrier=97.0, ko_rate=0.02, num_observations=2,
        ko_observation_dates=[2.0 / 365.0, 5.0 / 365.0],
        ki_observation_type=ObservationType.DISCRETE, ki_continuous=False,
        ki_observation_dates=[1.0 / 365.0, 3.0 / 365.0], include_principal=True,
    )


def _env(day: pd.Timestamp) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=SPOT, asset_name="X"), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02), valuation_date=pd.Timestamp(day).to_pydatetime(),
    )


def _tracker():
    return AutocallableLifecycleTracker(
        product=_product(), quantity=-1.0, has_lifecycle=True,
        lifecycle=AutocallableLifecycleState(), start_date=DATES[0],
    )


def test_the_schedule_fires_on_the_days_the_tracker_observes():
    schedule = _tracker().resolve_calendar_schedule(DATES, _env(DATES[0]))
    assert isinstance(schedule, CalendarSchedule)

    # Walk a real tracker day by day on a spot that never touches a barrier
    # and record which observation index each day marks as observed.
    tracker = _tracker()
    product = tracker.product_for_lifecycle()
    observed_ko, observed_ki = {}, {}
    for d, day in enumerate(DATES):
        before_ko = set(tracker.lifecycle.observed_ko_indices)
        before_ki = set(tracker.lifecycle.observed_ki_indices)
        tracker.observe(day, product, _env(day), SPOT)      # 100: between 97 and 103
        for idx in sorted(set(tracker.lifecycle.observed_ko_indices) - before_ko):
            observed_ko[idx] = d
        for idx in sorted(set(tracker.lifecycle.observed_ki_indices) - before_ki):
            observed_ki[idx] = d

    assert {int(k): int(v) for k, v in enumerate(schedule.ko_due_day)} == observed_ko
    assert {int(k): int(v) for k, v in enumerate(schedule.ki_due_day)} == observed_ki


def test_the_terminal_day_is_the_day_the_tracker_settles_maturity():
    schedule = _tracker().resolve_calendar_schedule(DATES, _env(DATES[0]))
    tracker = _tracker()
    product = tracker.product_for_lifecycle()
    settled_on = None
    for d, day in enumerate(DATES):
        tracker.observe(day, product, _env(day), SPOT)
        if tracker.settle_maturity_if_due(day, product, _env(day), SPOT) is not None:
            settled_on = d
            break
    assert schedule.terminal_due_day == settled_on
    assert schedule.terminal_settlement_day >= schedule.terminal_due_day


def test_barriers_payoffs_and_flags_come_off_the_resolved_records():
    schedule = _tracker().resolve_calendar_schedule(DATES, _env(DATES[0]))
    assert schedule.ko_barrier == pytest.approx([103.0, 103.0])
    assert np.all(schedule.ko_payoff > 0.0)
    assert schedule.ki_barrier == pytest.approx([97.0, 97.0])
    assert schedule.ki_continuous is False and schedule.is_reverse is False


def test_a_continuous_ki_product_reports_its_single_barrier():
    product = create_standard_snowball(
        initial_price=SPOT, strike=SPOT, maturity=6.0 / 365.0, contract_multiplier=1.0,
        ko_barrier=103.0, ki_barrier=97.0, ko_rate=0.02, num_observations=2,
        ko_observation_dates=[2.0 / 365.0, 5.0 / 365.0],
        ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True, include_principal=True,
    )
    tracker = AutocallableLifecycleTracker(product=product, quantity=-1.0, has_lifecycle=True,
                                           lifecycle=AutocallableLifecycleState(), start_date=DATES[0])
    schedule = tracker.resolve_calendar_schedule(DATES, _env(DATES[0]))
    assert schedule.ki_continuous is True
    assert schedule.ki_continuous_barrier == pytest.approx(97.0)
    assert schedule.ki_due_day.size == 0


def test_resolution_does_not_depend_on_the_day_the_environment_carries():
    early = _tracker().resolve_calendar_schedule(DATES, _env(DATES[0]))
    late = _tracker().resolve_calendar_schedule(DATES, _env(DATES[-1]))
    assert list(early.ko_due_day) == list(late.ko_due_day)
    assert list(early.ki_due_day) == list(late.ki_due_day)
    assert early.terminal_due_day == late.terminal_due_day


def test_an_observation_beyond_the_calendar_is_marked_minus_one():
    short = DATES[:2]
    schedule = _tracker().resolve_calendar_schedule(short, _env(short[0]))
    assert schedule.ko_due_day[-1] == -1
    assert schedule.terminal_due_day == -1
