from datetime import datetime, time, timedelta

import numpy as np
import pytest

from quantark.intraday import EventKind, Fixing, VarianceProfile, resolve_context
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.resolution import (REQUIRED_CELLS_PER_LAYER, REQUIRED_GRID_MODE_DAMPING, REQUIRED_STEPS_PER_LAYER,
                                          diffusion_layer, grid_mode_damping, pde_resolution, variance_steps_per_layer)
from intraday.conftest import SHANGHAI, dated_snowball, flat_env

T0 = datetime(2026, 3, 16)


def _ctx(sse_calendar, sse_sessions, profile, ts, **kw):
    prod = dated_snowball(sse_calendar, T0)
    probe = resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
                                                     session_calendar=sse_sessions, variance_profile=profile, **kw))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    stamp = ts(kos) if callable(ts) else ts
    return resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(stamp), session_calendar=sse_sessions,
                                                    variance_profile=profile, fixings=tuple(Fixing(k.timestamp, 100.0) for k in kos[:5]),
                                                    **kw))


def test_layer_shrinks_towards_the_fixing_and_resolution_follows(sse_calendar, sse_sessions):
    desk = VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))
    statuses, required = [], []
    for delta in (timedelta(days=1), timedelta(minutes=15), timedelta(seconds=1)):
        ctx = _ctx(sse_calendar, sse_sessions, desk, lambda kos, d=delta: kos[5].timestamp - d)
        res = pde_resolution(ctx, dx_at_spot=0.19 / 400, domain_log_width=0.19)
        statuses.append(res.status)
        required.append(res.required_points)
    assert statuses == ["resolved", "resolved", "unqualified"]
    assert required == sorted(required)
    one_second = _ctx(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(seconds=1))
    assert diffusion_layer(one_second) == pytest.approx((0.04 * 0.35 / 244 / 7200.0) ** 0.5, rel=1e-12)
    res = pde_resolution(one_second, dx_at_spot=0.19 / 400, domain_log_width=0.19)
    assert res.cells_per_layer < REQUIRED_CELLS_PER_LAYER and "points needed" in res.reason


def test_one_step_carrying_the_session_variance_is_unqualified_however_fine_the_space_grid(sse_calendar, sse_sessions):
    only = VarianceProfile.sessions_only(sse_sessions, 244)
    ctx = _ctx(sse_calendar, sse_sessions, only, lambda kos: kos[5].timestamp - timedelta(days=1))
    t_first = min(t for t in ctx.numerical.event_taus.values() if t > 0.0)
    layer = diffusion_layer(ctx)
    coarse = np.linspace(0.0, t_first, 5)                      # 6-hour steps: one holds both sessions of the day
    fine = np.linspace(0.0, t_first, 1025)
    assert variance_steps_per_layer(ctx, coarse) < 2.0
    assert variance_steps_per_layer(ctx, fine) >= REQUIRED_STEPS_PER_LAYER
    res = pde_resolution(ctx, dx_at_spot=layer / 40.0, domain_log_width=0.2, time_nodes=coarse)
    assert res.status == "unqualified" and "steps per day" in res.reason and "points" not in res.reason
    assert pde_resolution(ctx, dx_at_spot=layer / 40.0, domain_log_width=0.2, time_nodes=fine).status == "resolved"


def test_crank_nicolson_keeps_the_jump_mode_alive_when_each_step_spans_many_cells(sse_calendar, sse_sessions):
    only = VarianceProfile.sessions_only(sse_sessions, 244)
    ctx = _ctx(sse_calendar, sse_sessions, only, lambda kos: kos[5].timestamp - timedelta(days=1))
    t_first = min(t for t in ctx.numerical.event_taus.values() if t > 0.0)
    layer = diffusion_layer(ctx)
    nodes = np.linspace(0.0, t_first, 129)
    dx = layer / 40.0                                                     # 40 cells: mu = dW/dx^2 >> 1 per step
    cn = np.full(128, 0.5)
    rannacher = cn.copy()
    rannacher[-2:] = 1.0                                                  # the solver's implicit steps next to the event
    assert variance_steps_per_layer(ctx, nodes) >= REQUIRED_STEPS_PER_LAYER
    assert grid_mode_damping(ctx, nodes, cn, dx) < REQUIRED_GRID_MODE_DAMPING
    assert grid_mode_damping(ctx, nodes, rannacher, dx) > grid_mode_damping(ctx, nodes, cn, dx)
    res = pde_resolution(ctx, dx_at_spot=dx, domain_log_width=0.2, time_nodes=nodes, theta=cn, dx_min=dx)
    assert res.status == "unqualified" and "nepers" in res.reason
    one_step = np.array([0.0, t_first])                                   # dx^2 == dW: CN annihilates the mode in one step
    assert grid_mode_damping(ctx, one_step, [0.5], layer) > 30.0


def test_a_fixing_inside_a_zero_variance_window_is_deterministic(sse_calendar, sse_sessions):
    only = VarianceProfile.sessions_only(sse_sessions, 244)
    at_noon = _ctx(sse_calendar, sse_sessions, only,
                   lambda kos: kos[5].timestamp.replace(hour=12, minute=0), fixing_time_of_day=time(13, 0))
    assert pde_resolution(at_noon, dx_at_spot=1e-3, domain_log_width=0.2).status == "deterministic"
    close_fixing = _ctx(sse_calendar, sse_sessions, only, lambda kos: kos[5].timestamp.replace(hour=12, minute=0))
    assert pde_resolution(close_fixing, dx_at_spot=1e-3, domain_log_width=0.2).status != "deterministic"
