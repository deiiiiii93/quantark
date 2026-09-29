"""One intraday PDE-route snowball for the byte-capture book (spec §8.1).

Mirrors test/intraday/test_route_pde.py::test_one_day_out_refines_the_time_fill_so_no_step_jumps_the_layer:
the route refines the time fill, which is the code path the surface cleanup touches.
Not collected by pytest (no test_ prefix).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

from intraday.conftest import SHANGHAI, dated_snowball, flat_env
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.intraday import EventKind, Fixing, VarianceProfile, resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.session import TradingSession, TradingSessionCalendar
from quantark.util.calendar import CalendarType, create_calendar

T0 = datetime(2026, 3, 16)


def intraday_price() -> float:
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    sessions = TradingSessionCalendar(name="SSE", tz=SHANGHAI, calendar=cal,
                                      sessions=(TradingSession(time(9, 30), time(11, 30)),
                                                TradingSession(time(13, 0), time(15, 0))))
    desk = VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))
    prod = dated_snowball(cal, T0)
    probe = resolve_context(IntradayValuationRequest(
        product=prod, pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
        session_calendar=sessions, variance_profile=desk))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    ts = kos[5].timestamp - timedelta(days=1)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    ctx = resolve_context(IntradayValuationRequest(
        product=prod, pricing_env=flat_env(ts), session_calendar=sessions,
        variance_profile=desk, fixings=fixings, event_phase="before"))
    engine = SnowballPDESolver(PDEParams())
    return float(route_for(ctx, engine).price(ctx, engine).contingent_pv)
