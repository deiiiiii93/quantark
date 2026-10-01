"""The intraday PDE route budgets and searches on delivered steps (spec 3A §5.9)."""
from dataclasses import replace

import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.param import PDEParams
from quantark.intraday.engines import pde as route


@pytest.fixture
def ctx():
    """The daily-knock-in context of ``test/pde_bitwise/intraday_fixture.py``: one day before the sixth KO date."""
    from datetime import datetime, time, timedelta

    from intraday.conftest import SHANGHAI, dated_snowball, flat_env
    from quantark.intraday import EventKind, Fixing, VarianceProfile, resolve_context
    from quantark.intraday.request import IntradayValuationRequest
    from quantark.intraday.session import TradingSession, TradingSessionCalendar
    from quantark.util.calendar import CalendarType, create_calendar

    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    sessions = TradingSessionCalendar(name="SSE", tz=SHANGHAI, calendar=cal,
                                      sessions=(TradingSession(time(9, 30), time(11, 30)),
                                                TradingSession(time(13, 0), time(15, 0))))
    desk = VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))
    prod = dated_snowball(cal, datetime(2026, 3, 16))
    probe = resolve_context(IntradayValuationRequest(
        product=prod, pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
        session_calendar=sessions, variance_profile=desk))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    ts = kos[5].timestamp - timedelta(days=1)
    return resolve_context(IntradayValuationRequest(
        product=prod, pricing_env=flat_env(ts), session_calendar=sessions, variance_profile=desk,
        fixings=tuple(Fixing(k.timestamp, 100.0) for k in kos[:5]), event_phase="before"))


def _numbers(solver, ctx):
    solver.price(ctx.numerical.product, ctx.pricing_env)
    plan = route._solved_plan(solver, ctx.numerical.product, ctx.pricing_env)
    return plan, route._layout_numbers(plan, ctx.spot)


def test_the_memory_test_uses_delivered_steps(ctx):
    half = SnowballPDESolver(PDEParams(stepper=CrankNicolson(damping="half_step", terminal_damping_steps=1,
                                                             event_damping_steps=1)))
    plan, numbers = _numbers(half, ctx)
    assert numbers["actual_steps"] > numbers["requested_steps"]
    assert plan["time"]["actual_steps"] == numbers["actual_steps"]


def test_a_search_whose_next_trial_is_over_budget_returns_the_last_feasible_fill(ctx):
    solver = SnowballPDESolver(PDEParams(stepper=CrankNicolson(damping="half_step", terminal_damping_steps=1,
                                                               event_damping_steps=1)))
    plan, numbers = _numbers(solver, ctx)
    grid = solver.grid_binder.config
    budget = numbers["actual_steps"] + 1                      # the first doubling is over it
    factor, delivered = route._time_fill_for(ctx, solver, ctx.numerical.product, plan, grid, numbers,
                                             route._dx_min(plan), budget)
    assert (factor, delivered) == (1.0, numbers["actual_steps"])


def test_a_budget_basis_search_terminates(ctx):
    # Review Focus 5: delivered differs from time_steps by rounding; the search still ends
    solver = SnowballPDESolver(PDEParams(grid=GridConfig(time_steps=300)))
    plan, numbers = _numbers(solver, ctx)
    factor, delivered = route._time_fill_for(ctx, solver, ctx.numerical.product, plan,
                                             solver.grid_binder.config, numbers, route._dx_min(plan), 5000)
    assert delivered <= 5000 and factor >= 1.0


def test_a_feasible_trial_marks_its_half_steps(ctx, monkeypatch):
    solver = SnowballPDESolver(PDEParams(stepper=CrankNicolson(damping="half_step", terminal_damping_steps=1,
                                                               event_damping_steps=1)))
    plan, numbers = _numbers(solver, ctx)
    marked = []
    orig = CrankNicolson.schedule

    def spy(self, layout):
        theta = orig(self, layout)
        marked.append(int((theta != self.theta).sum()))
        return theta

    monkeypatch.setattr(CrankNicolson, "schedule", spy)
    # room for two trials (x2 and x4), so the search stays short
    route._time_fill_for(ctx, solver, ctx.numerical.product, plan, solver.grid_binder.config, numbers,
                         route._dx_min(plan), 4 * numbers["actual_steps"] + 64)
    assert marked and all(m > 0 for m in marked)
