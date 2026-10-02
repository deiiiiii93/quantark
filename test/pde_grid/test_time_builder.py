"""Tier-1 tests for the ONE time builder (spec §4.3)."""

import types

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.grid import GridConfig, GridRequest, resolve_config
from quantark.asset.equity.engine.pde.grid.time import build_time
from quantark.util.exceptions import ValidationError


def CFG(**kw):
    return resolve_config("standard", GridConfig(**kw) if kw else None)


def schedule(tl, request, term=1, event=2):
    """The stepper's θ schedule on a built time layout: 0.5 plain, 0.7 event-damped,
    1.0 terminal-damped (damping left TimeLayout for the stepper, spec 2 §4.2)."""
    from quantark.asset.equity.engine.pde.core.components import CrankNicolson

    return CrankNicolson(theta=0.5, event_theta=0.7, terminal_damping_steps=term, damping="full_step",
                         event_damping_steps=event).schedule(types.SimpleNamespace(time=tl, request=request))


def req(tau=1.0, events=(0.25, 0.5, 0.75)):
    return GridRequest(
        tau=tau,
        bound_anchors=(100.0,),
        critical_prices=(100.0,),
        hard_lower=None,
        hard_upper=None,
        event_times=events,
        day_count=252,
    )


def test_every_event_time_is_exact_node_with_verbatim_key():
    tl = build_time(req(), CFG())
    for e in (0.25, 0.5, 0.75):
        k = tl.step_of[e]  # exact float key — no searching
        assert tl.t[k] == e  # exact placement


def test_fill_formula_uncapped():
    # each 0.25y interval: 0.25 * 252 = 63 days * 4/day = 252 steps
    tl = build_time(req(), CFG())
    assert tl.actual_steps == 4 * 252
    assert tl.requested_steps == 4 * 252


def test_no_events_degenerates_to_uniform():
    tl = build_time(req(events=()), CFG())
    assert np.allclose(np.diff(tl.t), tl.dt)
    assert np.allclose(tl.dt, tl.dt[0])


def test_one_exact_dt_float_per_interval():
    # operator caches key on the dt float: each interval must contribute
    # exactly one distinct value (no linspace ULP wobble)
    tl = build_time(req(), CFG())
    assert len({float(d) for d in tl.dt}) <= 4  # at most one distinct dt per interval
    tl2 = build_time(req(events=(0.1, 0.5)), CFG())
    assert len({float(d) for d in tl2.dt}) <= 3


def test_single_interior_event():
    tl = build_time(req(events=(0.5,)), CFG())
    k = tl.step_of[0.5]
    assert tl.t[k] == 0.5
    # both halves: 0.5y * 252 = 126 days * 4 = 504 steps each
    assert tl.actual_steps == 1008


def test_sub_day_maturity_floor():
    tl = build_time(req(tau=0.5 / 252, events=()), CFG())
    assert tl.actual_steps == 4  # interval_days floored at 1.0 → 4 steps


def test_a_fill_over_the_cap_raises_never_scales():
    # spec 3A §5.6 (decision 7): many tiny intervals + one huge, tight cap
    events = tuple(np.linspace(1e-4, 99e-4, 99))
    with pytest.raises(ValidationError, match="max_steps"):
        build_time(req(tau=1.0, events=events), CFG(max_steps=100))


def test_mandatory_overflow_raises():
    events = tuple(np.linspace(0.001, 0.999, 400))  # 401 intervals > cap 100
    with pytest.raises(ValidationError, match="max_steps"):
        build_time(req(events=events), CFG(max_steps=100))


def test_damping_steps_after_each_event_node_and_terminal():
    tl = build_time(req(), CFG())
    theta = schedule(tl, req())
    for e in (0.25, 0.5, 0.75):
        k = tl.step_of[e]
        assert theta[k - 1] == theta[k - 2] == 0.7  # 2 after, backward time
    n = tl.actual_steps
    assert theta[n - 1] == 1.0  # first backward step


def test_damping_overlap_event_near_maturity():
    # event node 1 fill-step from maturity: a step both damped kinds claim
    r = req(tau=2.0 / 252, events=(1.0 / 252,))
    tl = build_time(r, CFG(steps_per_day=1.0))
    theta = schedule(tl, r, term=1, event=2)
    n = tl.actual_steps
    k = tl.step_of[1.0 / 252]
    # the terminal damping takes precedence on the shared step
    assert theta[n - 1] == 1.0
    assert all(theta[s] == 0.7 for s in (k - 1, k - 2) if 0 <= s != n - 1)


def test_zero_damping_counts_disable():
    tl = build_time(req(), CFG())
    assert (schedule(tl, req(), term=0, event=0) == 0.5).all()


def test_immutability_and_identity():
    tl = build_time(req(), CFG())
    with pytest.raises(ValueError):
        tl.t[0] = 99.0
    assert isinstance(tl.step_of, types.MappingProxyType)
    with pytest.raises(TypeError):
        tl.step_of[0.5] = 1  # proxy over a private copy
    other = build_time(req(), CFG())
    assert tl != other and tl == tl  # eq=False → identity semantics


def test_the_default_request_is_unchanged_by_the_new_field():
    plain, explicit = req(), GridRequest(
        tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None, hard_upper=None,
        event_times=(0.25, 0.5, 0.75), extra_times=(),
        day_count=252,
    )
    assert plain == explicit and hash(plain) == hash(explicit)
    a, b = build_time(plain, CFG()), build_time(explicit, CFG())
    assert np.array_equal(a.t, b.t) and np.array_equal(a.dt, b.dt)
    assert np.array_equal(schedule(a, plain), schedule(b, explicit))


def test_extra_times_are_exact_nodes_without_damping():
    extras = tuple(d / 365.0 for d in range(1, 365))
    request = GridRequest(tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None,
                          hard_upper=None, event_times=(0.25, 0.5, 0.75), extra_times=extras, day_count=252)
    tl = build_time(request, CFG())
    for t in extras:
        assert tl.t[tl.step_at(t)] == pytest.approx(t, abs=1e-15)
    plain = build_time(req(), CFG())
    theta, theta_plain = schedule(tl, request), schedule(plain, req())
    assert (theta == 0.7).sum() == (theta_plain == 0.7).sum()    # three events, damped alike
    assert np.flatnonzero(theta == 1.0).tolist() == [tl.actual_steps - 1]


def test_an_extra_time_on_an_event_is_folded_into_it():
    request = GridRequest(tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None,
                          hard_upper=None, event_times=(0.5,), extra_times=(0.5 + 1e-13, 0.25), day_count=252)
    assert request.extra_times == (0.25,)
    tl = build_time(request, CFG())
    assert tl.step_at(0.5) == tl.step_of[0.5]


def test_extra_times_at_the_endpoints_are_rejected():
    from quantark.util.exceptions import ValidationError

    with pytest.raises(ValidationError):
        GridRequest(tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None,
                    hard_upper=None, event_times=(), extra_times=(1.0,), day_count=252)


def test_the_over_cap_message_names_the_count_and_both_fixes():
    events = tuple(np.linspace(1e-4, 99e-4, 99))
    with pytest.raises(ValidationError) as exc:
        build_time(req(tau=1.0, events=events), CFG(max_steps=100))
    msg = str(exc.value)
    assert "GridConfig(max_steps=" in msg and "steps_per_day" in msg and "100" in msg


def test_fill_under_the_cap_is_delivered_as_asked(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="quantark.asset.equity.engine.pde.grid.time"):
        tl = build_time(req(), CFG())
    assert tl.actual_steps == tl.requested_steps
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---- 0.6.0: the time_steps budget basis (spec §4.2, §5.3, decision 16) ----

import dataclasses  # noqa: E402


def test_budget_no_events_is_linspace_bitwise():
    tl = build_time(req(tau=1.3, events=()), CFG(time_steps=200))
    assert np.array_equal(tl.t, np.linspace(0.0, 1.3, 201))
    assert tl.requested_steps == 200 and tl.actual_steps == 200


def test_budget_spread_proportionally_with_floor_one():
    tl = build_time(req(events=(0.25, 0.5, 0.75)), CFG(time_steps=400))
    assert tl.actual_steps == 400
    assert [tl.step_of[e] for e in (0.25, 0.5, 0.75)] == [100, 200, 300]


def test_budget_below_interval_count_floors_to_one_each():
    events = tuple(k / 10 for k in range(1, 10))  # 10 intervals
    tl = build_time(req(events=events), CFG(time_steps=4))
    assert tl.actual_steps == 10 and tl.requested_steps == 4
    assert tl.actual_steps > tl.requested_steps


def test_budget_capped_by_max_steps():
    events = tuple(k / 10 for k in range(1, 10))
    # a built config (not re-resolved): a budget over the cap raises, never scales
    cfg = dataclasses.replace(CFG(time_steps=100), max_steps=50)
    with pytest.raises(ValidationError, match="max_steps"):
        build_time(req(events=events), cfg)


def test_budget_equal_to_density_fill_is_bit_identical():
    dens = build_time(req(), CFG())
    bud = build_time(req(), CFG(time_steps=dens.actual_steps))
    assert np.array_equal(dens.t, bud.t) and np.array_equal(dens.dt, bud.dt)
    assert dict(dens.step_of) == dict(bud.step_of)


def test_density_branch_unchanged():
    tl = build_time(req(), CFG())
    assert tl.actual_steps == tl.requested_steps == 4 * 252


#: The 0.6.0 TimeLayout's damping sets for these four requests, rendered as a
#: θ schedule (0.5 plain, 0.7 event-damped, 1.0 terminal-damped); captured once
#: at the Task 11 commit, before the sets left TimeLayout.
_FROZEN_SCHEDULES = {
    (1, 2): [0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 1.0],
    (0, 1): [0.5, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5],
    (3, 0): [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
    (40, 40): [1.0] * 20,
}


@pytest.mark.parametrize("term,event", [(1, 2), (0, 1), (3, 0), (40, 40)])
def test_stepper_schedule_equals_the_0_6_time_layout_schedule(term, event):
    from quantark.asset.equity.engine.pde.core.components import CrankNicolson
    req = GridRequest(tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,),
                      hard_lower=None, hard_upper=None, event_times=(0.25, 0.5, 0.75), extra_times=(0.6,), day_count=252)
    time = build_time(req, resolve_config("standard", GridConfig(time_steps=20)))
    assert not hasattr(time, "event_damping_steps") and not hasattr(time, "terminal_damping_steps")
    layout = types.SimpleNamespace(time=time, request=req)
    got = CrankNicolson(theta=0.5, event_theta=0.7, terminal_damping_steps=term, event_damping_steps=event,
                        damping="full_step").schedule(layout)
    assert got.tolist() == _FROZEN_SCHEDULES[(term, event)]
