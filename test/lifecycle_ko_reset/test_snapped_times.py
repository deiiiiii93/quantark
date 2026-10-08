"""Snapped numeric observation times (spec 2026-10-08 KO-reset lifecycle, section 4a)."""
import pandas as pd
import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.snapping import is_off_day, snap_contract, snap_time
from quantark.asset.equity.param import PDEParams
from quantark.backtest.simulation.pricing.repricing import product_fingerprint

from .conftest import START, env, snowball

MONTHS = [k / 12.0 for k in range(1, 13)]


def test_snap_time_moves_an_off_day_time_to_its_rounded_day():
    assert snap_time(1.4 / 365.0) == 1 / 365.0
    assert snap_time(1 / 12.0) == 30 / 365.0
    assert snap_time(None) is None
    assert not is_off_day(30 / 365.0) and is_off_day(1 / 12.0)


def test_a_whole_day_time_keeps_its_exact_float():
    t = 30 * (1 / 365.0)                # 30 / 365 spelled another way
    assert snap_time(t) == t


def test_a_whole_day_contract_is_bit_identical():
    product = snowball([30 / 365, 60 / 365], [30 / 365], 60 / 365)
    assert product_fingerprint(snap_contract(product)) == product_fingerprint(product)


def test_the_tracker_holds_the_snapped_contract():
    tracker = AutocallableLifecycleTracker(product=snowball(MONTHS, MONTHS, 1.0), quantity=1.0, start_date=START)
    times = [r.observation_time for r in tracker.lifecycle_contract.barrier_config.ko_observation_schedule.records]
    assert times == [round(t * 365) / 365 for t in MONTHS]


def test_the_observed_record_leaves_the_post_close_copy_on_its_day():
    tracker = AutocallableLifecycleTracker(product=snowball(MONTHS, MONTHS, 1.0), quantity=1.0, start_date=START)
    day = START + pd.Timedelta(days=30)             # 1/12 year rounds to day 30
    post = tracker.product_for_pricing(day, env(date=day))
    initial = tracker.product_for_pricing(day, env(date=day), initial=True)
    assert len(post.barrier_config.ko_observation_schedule.records) == 11
    assert len(initial.barrier_config.ko_observation_schedule.records) == 12
    assert initial.barrier_config.ko_observation_schedule.records[0].observation_time == 0.0


def test_a_phoenix_coupon_schedule_snaps_with_its_knock_out_schedule():
    from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix

    phoenix = create_standard_phoenix(initial_price=100.0, strike=100.0, maturity=1.0, ko_barrier=103.0,
                                      ki_barrier=75.0, coupon_barrier=80.0, num_observations=12)
    tracker = AutocallableLifecycleTracker(product=phoenix, quantity=1.0, start_date=START)
    day = START + pd.Timedelta(days=30)
    post = tracker.product_for_pricing(day, env(date=day))
    assert len(post.barrier_config.ko_observation_schedule.records) == 11


def test_snapped_knock_out_cash_matches_the_engine():
    """Review focus 3: annualized off-day cash is booked and priced alike, and it is the contract's quoted 1/12, not
    the snapped day's 30/365 (accrual time spec R1; the KO-reset spec's "cash follows snap" is withdrawn)."""
    tracker = AutocallableLifecycleTracker(product=snowball(MONTHS, MONTHS, 1.0, annualized=True), quantity=1.0,
                                           start_date=START)
    day = START + pd.Timedelta(days=30)
    events = tracker.observe(day, tracker.product_for_lifecycle(), env(110.0, day), 110.0)
    assert events and events[0].event_type.name == "KNOCK_OUT"
    copy = tracker.product_for_pricing(day, env(110.0, day), initial=True)
    decided = SnowballPDESolver(PDEParams()).price(copy, env(110.0, day))
    assert events[0].cashflow == pytest.approx(decided, rel=1e-9)
    assert events[0].cashflow == pytest.approx(100.0 * 0.15 / 12.0, rel=1e-12)


def test_records_snapped_onto_a_kept_whole_day_stay_ordered():
    """A whole-day time kept as its exact float (a hair above k/365) and an off-day neighbour that rounds onto the
    same day: the neighbour shares the kept time, so the schedule stays non-decreasing (case study i/36 x tenor)."""
    kept = 30 / 365.0 + 1e-12                       # whole day within Tolerance.PRECISION: kept as is
    product = snowball([kept, 30.4 / 365.0, 60 / 365.0], [30 / 365.0], 60 / 365.0)
    times = [r.observation_time for r in snap_contract(product).barrier_config.ko_observation_schedule.records]
    assert times == [kept, kept, 60 / 365.0]


def test_a_record_snapped_onto_a_kept_whole_day_from_below_stays_ordered():
    """The case study's i/36 x tenor: 0.6 days snaps to day 1, next to a whole day kept a hair BELOW 1/365."""
    kept = 1 / 365.0 - 1e-18 if 1 / 365.0 - 1e-18 != 1 / 365.0 else (1 / 365.0) * (1 - 2 ** -52)
    product = snowball([0.6 / 365.0, kept, 60 / 365.0], [30 / 365.0], 60 / 365.0)
    times = [r.observation_time for r in snap_contract(product).barrier_config.ko_observation_schedule.records]
    assert times == [kept, kept, 60 / 365.0]


@pytest.mark.parametrize("k", [3, 6, 12, 45])
def test_a_whole_day_record_an_ulp_above_its_day_leaves_the_post_close_copy(k):
    """Self-review (the C2 failure on schedule 1): ``k * (1/365)`` is a whole day kept as its exact float, an ulp above
    ``k / 365`` for 47 of the first 730 days; ageing by the elapsed ``k / 365`` left the record observed on that close
    ~1e-18 years ahead (``ObservationSchedule.time_shift`` keeps it on purpose: lifecycle ageing stays byte-identical,
    theta spec R2), and QUAD refused the copy. The tracker, which owns the day rule, drops it from its post-close copy."""
    from quantark.asset.equity.engine.quad import SnowballQuadEngine
    from quantark.asset.equity.param import QuadParams

    d = 1 / 365.0
    assert k * d > k / 365.0
    tracker = AutocallableLifecycleTracker(product=snowball([k * d, 200 * d], [k * d, 200 * d], 200 * d),
                                           quantity=1.0, start_date=START)
    day = START + pd.Timedelta(days=k)
    tracker.observe(day, tracker.product_for_lifecycle(), env(90.0, day), 90.0)
    copy = tracker.product_for_pricing(day, env(90.0, day))
    assert [r.observation_time for r in copy.barrier_config.ko_observation_schedule.records] == [
        pytest.approx((200 - k) * d)]
    assert len(copy.barrier_config.ki_observation_schedule.records) == 1
    assert SnowballQuadEngine(params=QuadParams(grid_points=801)).price(copy, env(90.0, day)) == pytest.approx(
        SnowballQuadEngine(params=QuadParams(grid_points=801)).price(
            AutocallableLifecycleTracker(product=snowball([200 * d], [200 * d], 200 * d), quantity=1.0,
                                         start_date=START).product_for_pricing(day, env(90.0, day)),
            env(90.0, day)), rel=1e-12)
