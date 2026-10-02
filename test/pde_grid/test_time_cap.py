"""A grid the cap cannot serve fails with a diagnostic; nothing is scaled (spec 3A §5.6, decision 7)."""
import pytest

from quantark.asset.equity.engine import PDEEngine
from quantark.asset.equity.engine.pde.grid import GridRequest, TimeRefinement, build_time, time_fill
from quantark.asset.equity.engine.pde.grid.config import GridConfig, resolve_config
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_snowball import make_environment, make_snowball
from quantark.util.exceptions import ValidationError

from pde_3a.products import ENV, SNOWBALL


def _req(events=(), refinement=TimeRefinement(), tau=1.0):
    return GridRequest(tau=tau, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None, hard_upper=None,
                       event_times=events, refinement=refinement, day_count=252)


def test_the_default_cap_is_12000():
    assert resolve_config("standard", None).max_steps == 12000


def test_a_density_fill_over_the_cap_raises_naming_both_fixes():
    cfg = resolve_config("standard", GridConfig(steps_per_day=4.0, max_steps=1000))
    with pytest.raises(ValidationError) as exc:
        build_time(_req(), cfg)
    msg = str(exc.value)
    assert "1008" in msg and "1000" in msg and "GridConfig(max_steps=" in msg and "steps_per_day" in msg


def test_half_nodes_that_tip_a_grid_over_the_cap_raise():
    cfg = resolve_config("standard", GridConfig(steps_per_day=4.0, max_steps=1008))
    build_time(_req(), cfg)
    with pytest.raises(ValidationError):
        build_time(_req(refinement=TimeRefinement(1, 0)), cfg)


def test_more_intervals_than_the_cap_raise():
    cfg = resolve_config("standard", GridConfig(steps_per_day=4.0, max_steps=10))
    with pytest.raises(ValidationError):
        build_time(_req(events=tuple(k / 20.0 for k in range(1, 20))), cfg)


def test_time_fill_is_what_build_time_checks():
    cfg = resolve_config("standard", GridConfig(steps_per_day=4.0, max_steps=12000))
    r = _req(events=(0.25, 0.5), refinement=TimeRefinement(1, 1))
    assert time_fill(r, cfg) == build_time(r, cfg).actual_steps


def test_five_year_high_prices_at_the_default_cap():
    p = make_snowball(dict(SNOWBALL, months=60, maturity=5.0))
    plan = PDEEngine(params=PDEParams(accuracy="high"))._get_solver(p).explain(p, make_environment(ENV))
    assert plan.time.actual_steps <= 12000 and plan.time.actual_steps >= 10080
