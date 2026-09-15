from datetime import datetime, time, timedelta

import pytest

from quantark.intraday import EventKind, Fixing, VarianceProfile, resolve_context
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.resolution import REQUIRED_CELLS_PER_LAYER, diffusion_layer, pde_resolution
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


def test_a_fixing_inside_a_zero_variance_window_is_deterministic(sse_calendar, sse_sessions):
    only = VarianceProfile.sessions_only(sse_sessions, 244)
    at_noon = _ctx(sse_calendar, sse_sessions, only,
                   lambda kos: kos[5].timestamp.replace(hour=12, minute=0), fixing_time_of_day=time(13, 0))
    assert pde_resolution(at_noon, dx_at_spot=1e-3, domain_log_width=0.2).status == "deterministic"
    close_fixing = _ctx(sse_calendar, sse_sessions, only, lambda kos: kos[5].timestamp.replace(hour=12, minute=0))
    assert pde_resolution(close_fixing, dx_at_spot=1e-3, domain_log_width=0.2).status != "deterministic"
