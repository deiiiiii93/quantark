"""Spec §4.2 precedence tables, row by row (decision 17). Normative.

Rows marked "existing" pin what the builders already do; rows marked "new"
are the resolution-time errors the 0.6.0 surface adds for combinations that
were silent before.
"""
import types

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.grid import (
    GridBinder,
    GridConfig,
    GridRequest,
    MarketSnapshot,
    resolve_config,
)
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.engine.pde.grid.space import build_space
from quantark.asset.equity.engine.pde.grid.time import build_time
from quantark.util.exceptions import PricingError, ValidationError

MKT = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.03, q_ref=0.01)


def req(tau=1.0, events=(), criticals=(100.0,), hard_lower=None, hard_upper=None,
        anchors=(100.0,)):
    return GridRequest(tau=tau, bound_anchors=anchors, critical_prices=criticals,
                       hard_lower=hard_lower, hard_upper=hard_upper, event_times=events)


def cfg(**kw):
    return resolve_config("standard", GridConfig(**kw) if kw else None)


# ---- time axis, highest authority first ---------------------------------------

def test_t1_mandatory_nodes_survive_a_cap_the_intervals_alone_exceed():  # existing
    events = tuple(k / 20 for k in range(1, 20))  # 20 intervals
    tl = build_time(req(events=events), cfg(max_steps=10))
    assert tl.cap_exceeded and tl.actual_steps == 20
    assert all(tl.t[tl.step_of[e]] == e for e in events)


def test_t2_density_over_cap_scales_extras_only():  # existing: 5y standard asks 5040
    tl = build_time(req(tau=5.0, events=(1.0, 2.0, 3.0, 4.0)), cfg())
    assert tl.requested_steps == 5040
    assert tl.actual_steps <= 5000 and tl.fill_scaled and not tl.cap_exceeded


def test_t3_budget_below_interval_count_floors_to_one_per_interval():
    events = tuple(k / 10 for k in range(1, 10))
    tl = build_time(req(events=events), cfg(time_steps=3))
    assert tl.actual_steps == 10 > tl.requested_steps == 3


def test_t4_damping_never_changes_fill_and_spills_backward():  # existing
    # damping lives on the stepper (spec 2 §4.2): the time layout cannot see it,
    # and the schedule spills backward from the event, clipped at step 0
    r = req(events=(0.5,))
    tl = build_time(r, cfg(time_steps=10))
    theta = CrankNicolson(theta=0.5, event_theta=0.7, event_damping_steps=7,
                          terminal_damping_steps=3).schedule(types.SimpleNamespace(time=tl, request=r))
    k = tl.step_of[0.5]  # node 5
    assert [i for i, th in enumerate(theta) if th == 0.7] == list(range(k - 5, k))  # 7 asked
    assert [i for i, th in enumerate(theta) if th == 1.0] == [7, 8, 9]


def test_t4_extra_times_are_not_damped():  # existing
    r = GridRequest(tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,),
                    hard_lower=None, hard_upper=None, event_times=(), extra_times=(0.5,))
    tl = build_time(r, cfg(time_steps=10))
    theta = CrankNicolson(event_theta=0.7, terminal_damping_steps=0, event_damping_steps=2).schedule(
        types.SimpleNamespace(time=tl, request=r))
    assert (theta == 0.5).all()
    assert tl.t[tl.step_of[0.5]] == 0.5


@pytest.mark.parametrize("overlay,match", [
    (dict(steps_per_day=4.0, time_steps=100), "both"),  # new
    (dict(day_count=244, time_steps=100), "day_count"),  # new
    (dict(time_steps=6000), "max_steps"),  # new
])
def test_t_resolution_conflicts_raise(overlay, match):
    with pytest.raises(ValidationError, match=match):
        resolve_config("standard", GridConfig(**overlay))


def test_t_overlay_basis_clears_the_route_basis():  # new
    assert cfg(time_steps=300).steps_per_day is None
    assert cfg(steps_per_day=6.0).time_steps is None


# ---- space axis, highest authority first --------------------------------------

def test_s1_hard_edge_is_the_domain_edge_and_bounds_on_that_side_raise():  # existing
    with pytest.raises(ValidationError, match="hard_upper"):
        GridBinder("standard", GridConfig(bounds=(None, 150.0))).bind(
            req(hard_upper=130.0, criticals=(100.0, 130.0)), MKT)
    lay = GridBinder("standard", None).bind(req(hard_upper=130.0, criticals=(100.0, 130.0)), MKT)
    assert lay.spatial.bounds[1] == 130.0


def test_s2_bounds_beat_the_envelope_on_their_side_only():  # existing
    auto = build_space(req(), MKT, cfg())
    one = build_space(req(), MKT, cfg(bounds=(60.0, None)))
    assert one.bounds[0] == 60.0 and one.bounds[1] == auto.bounds[1]


def test_s2_bound_excluding_every_anchor_raises():  # existing
    with pytest.raises(PricingError):
        GridBinder("standard", GridConfig(bounds=(None, 90.0))).bind(req(), MKT)


def test_s4_points_is_exact():  # existing
    sp = build_space(req(criticals=(80.0, 100.0, 120.0)), MKT, cfg(points=333))
    assert len(sp.s) == 333


def test_s5_critical_pushed_outside_by_bounds_is_listed():  # existing
    sp = build_space(req(criticals=(100.0, 60.0)), MKT, cfg(bounds=(70.0, None)))
    assert 60.0 in sp.ignored_critical_prices
    assert 60.0 not in sp.active_critical_prices


def test_s5_missed_eps_crit_raises_rather_than_coarsening():  # existing
    crits = tuple(float(x) for x in range(60, 160, 2))
    with pytest.raises(ValidationError, match="points"):
        build_space(req(criticals=crits), MKT, cfg(points=40, eps_crit=1e-5))


def test_s6_points_over_max_points_raises_at_resolution():  # existing
    with pytest.raises(ValidationError, match="max_points"):
        cfg(points=3000)


def test_the_axes_do_not_interact():  # existing + the new budget basis
    a = GridBinder("standard", GridConfig(points=300, bounds=(50.0, 200.0))).bind(
        req(events=(0.5,)), MKT)
    b = GridBinder("standard", GridConfig(points=300, bounds=(50.0, 200.0), time_steps=50)).bind(
        req(events=(0.5,)), MKT)
    assert np.array_equal(a.spatial.s, b.spatial.s)
    c = GridBinder("standard", GridConfig(points=500)).bind(req(events=(0.5,)), MKT)
    assert np.array_equal(a.time.t, c.time.t)
