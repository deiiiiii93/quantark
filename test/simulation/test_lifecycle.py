from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.backtest.simulation.lifecycle import VectorLifecycle, receivable_pv
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

from .conftest import SPOT, short_snowball

DATES = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=8, freq="D"))
KO = 1.03 * SPOT
KI = 0.75 * SPOT


def _env(day) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=SPOT, asset_name="X"), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02), valuation_date=pd.Timestamp(day).to_pydatetime(),
    )


def _tracker(product, quantity=-1.0):
    return AutocallableLifecycleTracker(product=product, quantity=quantity, has_lifecycle=True,
                                        lifecycle=AutocallableLifecycleState(), start_date=DATES[0])


def _vector(product, quantity=-1.0, n_paths=1):
    schedule = _tracker(product, quantity).resolve_calendar_schedule(DATES, _env(DATES[0]))
    return VectorLifecycle([schedule], np.array([quantity]), n_paths=n_paths)


def _walk_tracker(product, spots, quantity=-1.0):
    """The tracker's own answer on one hand-built path."""
    tracker = _tracker(product, quantity)
    lifecycle_product = tracker.product_for_lifecycle()
    out = []
    for day, spot in zip(DATES, spots):
        tracker.observe(day, lifecycle_product, _env(day), float(spot))
        tracker.settle_maturity_if_due(day, lifecycle_product, _env(day), float(spot))
        if not tracker.lifecycle.settled and tracker.lifecycle.settlement_date is not None:
            if pd.Timestamp(day).normalize() >= pd.Timestamp(tracker.lifecycle.settlement_date).normalize():
                tracker.lifecycle.settle()
        out.append({
            "alive": tracker.lifecycle.alive, "knocked_in": tracker.lifecycle.knocked_in,
            "knocked_out": tracker.lifecycle.knocked_out, "matured": tracker.lifecycle.matured,
            "settled": tracker.lifecycle.settled,
            "realized": tracker.lifecycle.realized_cashflows,
            "pending": tracker.lifecycle.pending_settlement_cashflow,
        })
    return out


def _walk_vector(product, spots, quantity=-1.0):
    vec = _vector(product, quantity)
    out = []
    for d, spot in enumerate(spots):
        vec.step(d, np.array([float(spot)]),
                 lambda p, paths, ki, spot=spot: np.array([
                     float(product.get_payoff(float(spot), _env(DATES[d]), knocked_in=bool(k))) for k in ki
                 ]))
        out.append({
            "alive": bool(vec.alive[0, 0]), "knocked_in": bool(vec.knocked_in[0, 0]),
            "knocked_out": bool(vec.knocked_out[0, 0]), "matured": bool(vec.matured[0, 0]),
            "settled": bool(vec.settled[0, 0]), "realized": float(vec.realized[0, 0]),
            "pending": float(vec.pending[0, 0]),
        })
    return out


@pytest.mark.parametrize("name, spots", [
    ("never_touches", [SPOT] * 8),
    ("ko_on_the_first_observation", [SPOT, SPOT, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]),
    ("ko_only_on_a_due_day", [KO + 1.0, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT, SPOT]),
    ("ki_then_maturity", [SPOT, KI - 1.0, SPOT, SPOT, SPOT, SPOT, SPOT * 0.8, SPOT * 0.8]),
    ("ki_and_ko_on_the_same_day", [SPOT, KI - 1.0, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]),
    ("exactly_on_the_barrier", [SPOT, KI, KO, SPOT, SPOT, SPOT, SPOT, SPOT]),
])
def test_the_vector_lifecycle_matches_the_tracker_day_by_day(name, spots):
    product = short_snowball()
    expected = _walk_tracker(product, spots)
    actual = _walk_vector(product, spots)
    for d, (e, a) in enumerate(zip(expected, actual)):
        assert a == pytest.approx(e), f"{name} day {d}: {a} != {e}"


def test_continuous_ki_fires_on_any_day():
    product = short_snowball(continuous_ki=True)
    spots = [SPOT, SPOT, KI - 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]
    assert _walk_vector(product, spots) == _walk_tracker(product, spots)


def test_disable_ko_after_ki_stops_marking_ko_observations():
    product = short_snowball()
    product.barrier_config = dataclasses.replace(product.barrier_config, disable_ko_after_ki=True)
    spots = [SPOT, KI - 1.0, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]
    assert _walk_vector(product, spots) == _walk_tracker(product, spots)


def test_paths_are_independent_within_one_batch():
    product = short_snowball()
    vec = _vector(product, n_paths=3)
    lanes = [[SPOT] * 8, [SPOT, SPOT, KO + 1.0] + [SPOT] * 5, [SPOT, KI - 1.0] + [SPOT] * 6]
    for d in range(8):
        spot = np.array([lane[d] for lane in lanes])
        vec.step(d, spot, lambda p, paths, ki: np.zeros(paths.size))
    assert list(vec.knocked_out[0]) == [False, True, False]
    assert list(vec.knocked_in[0]) == [False, False, True]


def test_book_flags_use_the_replay_aggregation():
    product = short_snowball()
    schedules = [_tracker(product).resolve_calendar_schedule(DATES, _env(DATES[0]))] * 2
    vec = VectorLifecycle(schedules, np.array([-1.0, -1.0]), n_paths=1)
    vec.knocked_out[0, 0] = True
    vec.alive[0, 0] = False
    flags = vec.book_flags()
    assert bool(flags["alive"][0])              # ANY product alive: the second one is
    assert not bool(flags["knocked_out"][0])    # ALL products out: only the first one is


def test_records_report_each_event_once():
    product = short_snowball()
    vec = _vector(product, n_paths=1)
    spots = [SPOT, KI - 1.0, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]
    events = []
    for d, spot in enumerate(spots):
        events += vec.step(d, np.array([spot]), lambda p, paths, ki: np.zeros(paths.size))
    kinds = [e.event for e in events]
    assert kinds.count("knock_in") == 1 and kinds.count("knock_out") == 1


@pytest.mark.parametrize("name, spots", [
    ("ko_then_two_days_of_receivable", [SPOT, SPOT, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]),
    ("late_ko_settles_after_maturity", [SPOT] * 5 + [KO + 1.0, SPOT, SPOT]),
    ("maturity_with_a_lag", [SPOT] * 8),
    ("ki_then_maturity_with_a_lag", [SPOT, KI - 1.0, SPOT, SPOT, SPOT, SPOT, SPOT * 0.8, SPOT * 0.8]),
])
def test_a_settlement_lag_matches_the_tracker_day_by_day(name, spots):
    product = short_snowball(settlement_lag_days=2)
    expected = _walk_tracker(product, spots)
    actual = _walk_vector(product, spots)
    assert any(day["pending"] != 0.0 for day in expected), "the fixture must actually park a receivable"
    for d, (e, a) in enumerate(zip(expected, actual)):
        assert a == pytest.approx(e), f"{name} day {d}: {a} != {e}"


def test_the_receivable_pv_matches_the_ledger_on_the_numeric_clock():
    product = short_snowball(settlement_lag_days=2)
    spots = [SPOT, SPOT, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]
    tracker = _tracker(product)
    lifecycle_product = tracker.product_for_lifecycle()
    vec = _vector(product)
    rate = np.array([0.02])
    seen_pending = False
    for d, (day, spot) in enumerate(zip(DATES, spots)):
        env = _env(day)
        tracker.observe(day, lifecycle_product, env, float(spot))
        tracker.settle_maturity_if_due(day, lifecycle_product, env, float(spot))
        if not tracker.lifecycle.settled and tracker.lifecycle.settlement_date is not None:
            if pd.Timestamp(day).normalize() >= pd.Timestamp(tracker.lifecycle.settlement_date).normalize():
                tracker.lifecycle.settle()
        vec.step(d, np.array([float(spot)]), lambda p, paths, ki: np.zeros(paths.size))
        point = tracker.lifecycle.valuation_point
        expected = float(tracker.lifecycle.ledger.pending_pv(point, env)) if point is not None else 0.0
        got = receivable_pv(vec, rate, d)
        assert got[0] == expected, f"day {d}: {got[0]!r} != {expected!r}"
        seen_pending |= expected != 0.0
    assert seen_pending


def test_the_schedule_carries_the_delays():
    product = short_snowball(settlement_lag_days=2)
    schedule = _tracker(product).resolve_calendar_schedule(DATES, _env(DATES[0]))
    assert schedule.uses_date_timing is False
    assert schedule.ko_settlement_delay == pytest.approx([2.0 / 365.0] * 2)
    assert schedule.terminal_settlement_delay == pytest.approx(2.0 / 365.0)
    plain = _tracker(short_snowball()).resolve_calendar_schedule(DATES, _env(DATES[0]))
    assert plain.ko_settlement_delay == pytest.approx([0.0, 0.0])
    assert plain.terminal_settlement_delay == 0.0
