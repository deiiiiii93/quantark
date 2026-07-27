"""Tier-1 tests for the ONE time builder (spec §4.3)."""

import types

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.grid import GridConfig, GridRequest, resolve_config
from quantark.asset.equity.engine.pde.grid.time import build_time


def CFG(**kw):
    return resolve_config("standard", GridConfig(**kw) if kw else None)


def req(tau=1.0, events=(0.25, 0.5, 0.75)):
    return GridRequest(
        tau=tau,
        bound_anchors=(100.0,),
        critical_prices=(100.0,),
        hard_lower=None,
        hard_upper=None,
        event_times=events,
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
    assert not tl.cap_exceeded


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


def test_cap_extras_scaling_enforces_cap():
    # spec §5 adversarial case: many tiny intervals + one huge, tight cap
    events = tuple(np.linspace(1e-4, 99e-4, 99))
    tl = build_time(req(tau=1.0, events=events), CFG(max_steps=100))
    assert tl.actual_steps <= 100
    assert not tl.cap_exceeded
    for e in events:
        assert tl.t[tl.step_of[e]] == e  # nodes survive scaling


def test_mandatory_overflow_keeps_nodes_sets_flag():
    events = tuple(np.linspace(0.001, 0.999, 400))  # 401 intervals > cap 100
    tl = build_time(req(events=events), CFG(max_steps=100))
    for e in events:
        assert tl.t[tl.step_of[e]] == e  # inviolable
    assert tl.cap_exceeded and tl.actual_steps == 401


def test_damping_steps_after_each_event_node_and_terminal():
    tl = build_time(req(), CFG())
    for e in (0.25, 0.5, 0.75):
        k = tl.step_of[e]
        assert {k - 1, k - 2} <= tl.event_damping_steps  # 2 after, backward time
    n = tl.actual_steps
    assert {n - 1} <= tl.terminal_damping_steps  # first backward step


def test_damping_overlap_event_near_maturity():
    # event node 1 fill-step from maturity: overlapping step in BOTH sets
    tl = build_time(
        req(tau=2.0 / 252, events=(1.0 / 252,)),
        CFG(steps_per_day=1.0, terminal_damping_steps=1, event_damping_steps=2),
    )
    n = tl.actual_steps
    k = tl.step_of[1.0 / 252]
    overlap = {n - 1} & {k - 1, k - 2} if k >= 1 else set()
    # both memberships visible; θ-precedence is the consumer's job
    assert (n - 1) in tl.terminal_damping_steps
    assert tl.event_damping_steps >= {s for s in (k - 1, k - 2) if s >= 0}
    assert overlap <= tl.terminal_damping_steps


def test_zero_damping_counts_disable():
    tl = build_time(
        req(), CFG(terminal_damping_steps=0, event_damping_steps=0)
    )
    assert tl.event_damping_steps == frozenset()
    assert tl.terminal_damping_steps == frozenset()


def test_immutability_and_identity():
    tl = build_time(req(), CFG())
    with pytest.raises(ValueError):
        tl.t[0] = 99.0
    assert isinstance(tl.step_of, types.MappingProxyType)
    with pytest.raises(TypeError):
        tl.step_of[0.5] = 1  # proxy over a private copy
    other = build_time(req(), CFG())
    assert tl != other and tl == tl  # eq=False → identity semantics
