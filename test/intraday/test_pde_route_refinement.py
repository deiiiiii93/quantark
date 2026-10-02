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


def _discrete_touch(sse_sessions, params):
    """The discretely observed one-touch of test_route_pde's domain test, priced through the route under ``params``."""
    from datetime import datetime

    from intraday.conftest import SHANGHAI, flat_env
    from quantark.asset.equity.engine.pde import OneTouchPDESolver
    from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
    from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
    from quantark.intraday import VarianceProfile, resolve_context
    from quantark.intraday.engines import route_for
    from quantark.intraday.request import IntradayValuationRequest
    from quantark.util.enum.option_enums import BarrierDirection, ObservationType

    dates = [datetime(2026, 9, 18), datetime(2026, 9, 24), datetime(2026, 9, 30)]
    product = OneTouchOption(barrier=103.0, barrier_direction=BarrierDirection.UP, rebate=1.0,
                             exercise_date=datetime(2026, 9, 30), observation_type=ObservationType.DISCRETE,
                             observation_schedule=ObservationSchedule(
                                 records=[ObservationRecord(observation_date=d, barrier=103.0) for d in dates]))
    desk = VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))
    ctx = resolve_context(IntradayValuationRequest(
        product=product, pricing_env=flat_env(datetime(2026, 9, 15, 10, 0, tzinfo=SHANGHAI), spot=101.0, r=0.0, q=0.0),
        session_calendar=sse_sessions, variance_profile=desk))
    out = route_for(ctx, OneTouchPDESolver(params)).price(ctx, OneTouchPDESolver(params))
    solver = out.engine_used
    plan = route._solved_plan(solver, ctx.numerical.product, ctx.pricing_env)
    return route._barrier_placement(ctx, plan, solver)


def test_a_discrete_barrier_is_not_reported_misplaced_under_the_projection(sse_sessions):
    # 3A §5.4 (flip c): the projection enters the barrier's exact level into each dual cell; NODAL overwrites nodes
    assert _discrete_touch(sse_sessions, PDEParams()) == ""
    assert "node overwrite" in _discrete_touch(sse_sessions, PDEParams(event_projection="nodal"))


def _route_price(ctx, solver):
    from quantark.intraday.engines import route_for
    return route_for(ctx, solver).price(ctx, solver)



def test_a_budget_below_the_base_count_leaves_the_fill_and_says_why(ctx, monkeypatch):
    # spec 3A section 7.2: the memory budget allows fewer steps than the base grid delivers
    base = SnowballPDESolver(PDEParams())
    _, numbers = _numbers(base, ctx)
    pv = base.price(ctx.numerical.product, ctx.pricing_env)
    steps = numbers["actual_steps"]
    cells = numbers["points"] * steps // 2                      # the base grid alone is over the budget
    monkeypatch.setattr(route, "INTRADAY_PDE_MAX_GRID_BYTES",
                        cells * route.bytes_per_grid_cell(base, ctx.numerical.product))
    out = _route_price(ctx, SnowballPDESolver(PDEParams()))
    assert out.contingent_pv == pv
    assert (out.numerical["actual_steps"], out.numerical["points"]) == (steps, numbers["points"])
    assert out.records == ()
    assert out.numerical["resolution"] == route.UNDER_RESOLVED
    assert "refinement capped by the grid memory budget" in out.numerical["resolution_reason"]


def test_the_refined_solve_lifts_max_steps_to_the_chosen_trial(ctx):
    # spec 3A section 5.9 item 4: the chosen trial delivers more than the 12,000 cap; the refined solve prices it
    solver = SnowballPDESolver(PDEParams())
    plan, numbers = _numbers(solver, ctx)
    max_cells = route.INTRADAY_PDE_MAX_GRID_BYTES // route.bytes_per_grid_cell(solver, ctx.numerical.product)
    budget = min(route.INTRADAY_PDE_MAX_STEPS, max_cells // numbers["points"] - 1)
    factor, delivered = route._time_fill_for(ctx, solver, ctx.numerical.product, plan, solver.grid_binder.config,
                                             numbers, route._dx_min(plan), budget)
    assert delivered > solver.grid_binder.config.max_steps    # the lift is needed, not incidental
    out = _route_price(ctx, SnowballPDESolver(PDEParams()))
    refined = out.engine_used.grid_binder.config
    assert refined.max_steps == delivered == out.numerical["actual_steps"]
    assert out.numerical["steps_per_day"] == factor * solver.grid_binder.config.steps_per_day
