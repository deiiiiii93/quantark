"""Half-step refinement of the time grid (spec 3A §5.3) and the delivered step count (§5.6)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.pde.grid import GridRequest, TimeRefinement, build_time, time_fill
from quantark.asset.equity.engine.pde.grid import time as time_mod
from quantark.asset.equity.engine.pde.grid.config import GridConfig, resolve_config
from quantark.util.exceptions import ValidationError

CFG = resolve_config("standard", GridConfig(steps_per_day=1.0, max_steps=10**6))


def req(events=(), extras=(), refinement=TimeRefinement(), tau=20 / 252):
    return GridRequest(tau=tau, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None,
                       hard_upper=None, event_times=events, extra_times=extras, refinement=refinement)


def test_zero_refinement_is_the_unrefined_grid():
    base, zero = build_time(req(), CFG), build_time(req(refinement=TimeRefinement(0, 0)), CFG)
    assert base.t.tobytes() == zero.t.tobytes() and base.dt.tobytes() == zero.dt.tobytes()
    assert zero.actual_steps == zero.requested_steps == 20


def test_terminal_only_splits_the_last_intervals():
    base = build_time(req(), CFG)
    tl = build_time(req(refinement=TimeRefinement(2, 0)), CFG)
    assert tl.actual_steps == 22 and tl.requested_steps == 20
    assert tl.dt[-4:].tolist() == [base.dt[-2] / 2.0] * 2 + [base.dt[-1] / 2.0] * 2
    assert tl.t[-2] == 0.5 * (base.t[-2] + base.t[-1])
    assert np.all(np.diff(tl.t) > 0)


@pytest.mark.parametrize("gap", [1, 2, 5])
def test_event_nodes_split_the_intervals_below_them(gap):
    e1, e2 = 5 / 252, (5 + gap) / 252
    base = build_time(req(events=(e1, e2)), CFG)
    tl = build_time(req(events=(e1, e2), refinement=TimeRefinement(0, 2)), CFG)
    k1, k2 = base.step_of[e1], base.step_of[e2]
    split = {k1 - 1, k1 - 2, k2 - 1, k2 - 2}
    assert tl.actual_steps == base.actual_steps + len(split)
    assert tl.t[tl.step_of[e1]] == e1 and tl.t[tl.step_of[e2]] == e2


def test_an_interval_wanted_twice_is_split_once():
    e = 19 / 252                                 # the interval below e is also the terminal's second
    tl = build_time(req(events=(e,), refinement=TimeRefinement(2, 1)), CFG)
    assert tl.actual_steps == 20 + 2


def test_an_event_next_to_the_valuation_date_splits_only_what_exists():
    e = 1 / 252
    tl = build_time(req(events=(e,), refinement=TimeRefinement(0, 3)), CFG)
    assert tl.actual_steps == 21 and tl.step_of[e] == 2


def test_extra_times_trigger_no_split():
    x = 7 / 252
    tl = build_time(req(extras=(x,), refinement=TimeRefinement(0, 2)), CFG)
    assert tl.actual_steps == 20 and tl.t[tl.step_of[x]] == x


def test_half_steps_are_the_constructed_constants():
    base = build_time(req(), CFG)
    tl = build_time(req(refinement=TimeRefinement(1, 0)), CFG)
    assert tl.dt[-1] == tl.dt[-2] == base.dt[-1] / 2.0


@pytest.mark.parametrize("refinement", [TimeRefinement(0, 0), TimeRefinement(1, 1), TimeRefinement(2, 2),
                                        TimeRefinement(2, 1)])
@pytest.mark.parametrize("events", [(), (5 / 252,), (1 / 252, 2 / 252, 10 / 252)])
def test_time_fill_is_the_delivered_count(refinement, events):
    r = req(events=events, refinement=refinement)
    assert time_fill(r, CFG) == build_time(r, CFG).actual_steps


def test_time_fill_ignores_the_cap_in_stage_one():
    r = req(refinement=TimeRefinement(1, 1))
    assert time_fill(r, resolve_config("standard", GridConfig(steps_per_day=1.0, max_steps=5))) == 21


def test_an_interval_too_short_to_split_raises():
    # Review Focus 1: a midpoint that rounds onto an endpoint must never become a node
    a = 0.25
    t = np.array([0.0, a, np.nextafter(a, 1.0)])
    dt = np.diff(t)
    with pytest.raises(ValidationError, match="too short to split"):
        time_mod._refine(t, dt, {}, {1})


def test_refinement_validates_its_counts():
    for bad in (-1, 1.5, True):
        with pytest.raises(ValidationError):
            TimeRefinement(bad, 0)
