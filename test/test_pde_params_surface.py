"""The 0.6.0 PDE parameter surface contract (spec §4.4, §4.6)."""
import copy
import dataclasses
import pickle

import pytest

from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.param import PDEParams
from quantark.util.exceptions import ValidationError


def test_positional_pdeparams_raises_type_error():
    # without the wrapper, "high" binds EngineParams.bump_size and only fails by
    # accident further down; the contract is an explicit keyword-only refusal
    with pytest.raises(TypeError, match="keyword arguments only"):
        PDEParams("high")
    with pytest.raises(TypeError, match="keyword arguments only"):
        PDEParams(0.01)


def test_unknown_name_raises_validation_error():
    with pytest.raises(ValidationError, match="no_such_knob"):
        PDEParams(no_such_knob=1)
    with pytest.raises(ValidationError, match="no_such_knob"):
        GridConfig(no_such_knob=1)


def test_resolved_grid_is_the_requested_preset_expansion():
    g = PDEParams(accuracy="high", grid=GridConfig(points=900)).resolved_grid()
    assert (g.points, g.steps_per_day, g.max_steps) == (900, 8.0, 5000)
    doc = PDEParams.resolved_grid.__doc__
    assert "max_steps" in doc and "5,040" in doc and "TimeLayout" in doc


def test_params_clone_round_trips():
    # Review Focus 2: adapters deep-copy, pools pickle, callers replace().
    # pickle round-trips objects this test just built -- no untrusted input.
    p = PDEParams(accuracy="fast", grid=GridConfig(points=300, bounds=(50.0, None)), theta=1.0)
    assert copy.deepcopy(p) == p
    assert pickle.loads(pickle.dumps(p)) == p
    q = dataclasses.replace(p, theta=0.5)
    assert q.theta == 0.5 and q.grid == p.grid
    g = pickle.loads(pickle.dumps(p.grid))
    assert g == p.grid and dataclasses.replace(g, points=301).points == 301


@pytest.mark.parametrize("name", ["cache_strategy", "grid_cache_max_entries",
                                  "banded_cache_max_entries", "use_banded_solver"])
def test_cache_knobs_removed(name):
    with pytest.raises(ValidationError, match="cache_enabled|solver family"):
        PDEParams(**{name: 1})


@pytest.mark.parametrize("name", ["use_rannacher", "rannacher_steps",
                                  "rannacher_at_events", "event_rannacher_steps"])
def test_damping_knobs_removed(name):
    with pytest.raises(ValidationError, match="damping_steps"):
        PDEParams(**{name: 1})


def test_intraday_method_label_unchanged():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.intraday.engines.pde import PDERoute

    assert PDERoute._method(SnowballPDESolver(PDEParams())) == "pde_rannacher_theta0.5"
    no_term = SnowballPDESolver(PDEParams(grid=GridConfig(terminal_damping_steps=0)))
    assert PDERoute._method(no_term) == "pde_cn_theta0.5"


def test_cache_off_is_the_instance_flag_and_changes_no_number():
    from term_structure_benchmarks import make_term_env

    from quantark.asset.equity.engine.pde import BasePDESolver, EuropeanPDESolver
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType

    for gone in ("set_cache_enabled", "set_cache_strategy", "clear_grid_cache",
                 "set_grid_cache_max_entries", "_resolve_cache_strategy"):
        assert not hasattr(BasePDESolver, gone)
    p = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    env = make_term_env("flat")
    on, off = EuropeanPDESolver(PDEParams()), EuropeanPDESolver(PDEParams(cache_enabled=False))
    assert on._is_cache_enabled() and not off._is_cache_enabled()
    assert on.price(p, env) == off.price(p, env)
