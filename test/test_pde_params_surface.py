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


@pytest.mark.parametrize("name,value", [("continuous_ki_correction", "none"),
                                        ("boundary_mode", "default")])
def test_legacy_modes_removed(name, value):
    with pytest.raises(ValidationError, match="removed"):
        PDEParams(**{name: value})


def test_continuous_ki_correction_enum_gone():
    import quantark.util.enum as e

    assert not hasattr(e, "ContinuousKICorrection")


def test_ki_monitoring_mode_removed_from_pde_only():
    from quantark.asset.equity.param import QuadParams

    with pytest.raises(ValidationError, match="QuadParams.ki_monitoring_mode is unchanged"):
        PDEParams(ki_monitoring_mode="bgk_approximation")
    quad = QuadParams(ki_monitoring_mode="bgk_approximation")
    assert quad.ki_monitoring_mode.value == "bgk_approximation"


def test_pdeparams_field_list_is_final():
    own = [f.name for f in dataclasses.fields(PDEParams)
           if f.name not in ("bump_size", "bump_config", "bus_days_in_year")]
    assert own == ["accuracy", "grid", "theta", "event_theta", "event_projection",
                   "cache_enabled"]


def test_gridconfig_field_list_is_final():
    assert [f.name for f in dataclasses.fields(GridConfig)] == [
        "points", "steps_per_day", "time_steps", "eps_crit", "num_std", "bounds",
        "max_points", "max_steps", "day_count", "terminal_damping_steps",
        "event_damping_steps"]


def _removed_maps():
    from quantark.asset.equity.param.pde_removed_names import (
        REMOVED_DCN_NAMES,
        REMOVED_PDE_NAMES,
    )

    return REMOVED_PDE_NAMES, REMOVED_DCN_NAMES


@pytest.mark.parametrize("name", sorted(set(_removed_maps()[0]) - {"method"}))
def test_every_removed_pdeparams_name_names_its_replacement(name):
    removed = _removed_maps()[0]
    with pytest.raises(ValidationError) as exc:
        PDEParams(**{name: 1})
    assert removed[name] in str(exc.value)


@pytest.mark.parametrize("name", sorted(_removed_maps()[1]))
def test_every_removed_dcn_name_names_its_replacement(name):
    from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine

    removed = _removed_maps()[1]
    with pytest.raises(ValidationError) as exc:
        DCNPDEEngine(**{name: 1})
    assert removed[name] in str(exc.value)


def test_from_config_round_trip_and_yaml_bounds_become_a_tuple(tmp_path):
    import json

    cfg = {"engine": "pde", "accuracy": "high", "theta": 1.0,
           "grid": {"points": 900, "bounds": [None, 400.0], "time_steps": 800}}
    path = tmp_path / "pde.json"
    path.write_text(json.dumps(cfg))
    p = PDEParams.from_config(path, cache_enabled=False)
    assert p.accuracy == "high" and p.theta == 1.0 and p.cache_enabled is False
    assert p.grid.bounds == (None, 400.0) and hash(p.grid.key)   # Review Focus 1
    assert p.resolved_grid().time_steps == 800


@pytest.mark.parametrize("bad", [{"profile": "fast"}, {"reverse": True}, {"engine": "quad"},
                                 {"grid_size": 400}])
def test_from_config_rejects(bad):
    with pytest.raises(ValidationError):
        PDEParams.from_config(bad)


def test_pde_has_no_profile_tier():
    from quantark.asset.equity.param import engine_param_profiles as prof
    import quantark.asset.equity.param as param_pkg

    for fn, args in ((prof.make_engine_params, ("pde",)), (prof.list_param_profiles, ("pde",))):
        with pytest.raises(ValidationError, match="no profile tier"):
            fn(*args)
    assert not hasattr(param_pkg, "make_pde_params")
    assert not hasattr(prof, "PDE_PARAM_PRESETS")


def test_no_removed_name_is_read_in_the_library():
    """Spec §11: zero hits outside the removal map (equity/intraday/backtest/
    sacva/modelvalidation/execution): no ``params.<removed>`` read and no
    removed keyword passed to a PDE constructor."""
    import ast
    import pathlib

    pde_names, dcn_names = _removed_maps()
    names = (set(pde_names) | set(dcn_names)) - {"method", "time_steps"}
    ctors = {"PDEParams", "DCNPDEEngine", "LocalVolDCNPDEEngine", "PDEEngine"}
    roots = ["quantark/asset/equity", "quantark/intraday", "quantark/backtest",
             "quantark/sacva", "quantark/modelvalidation", "quantark/execution"]
    hits = []
    for root in roots:
        for p in pathlib.Path(root).rglob("*.py"):
            if p.name == "pde_removed_names.py" or "/mc/" in str(p) or "quad" in p.name:
                continue
            for n in ast.walk(ast.parse(p.read_text())):
                if (isinstance(n, ast.Attribute) and n.attr in names
                        and isinstance(n.value, ast.Attribute) and n.value.attr == "params"):
                    hits.append(f"{p}:{n.lineno} params.{n.attr}")
                if isinstance(n, ast.Call):
                    fname = getattr(n.func, "id", getattr(n.func, "attr", ""))
                    if fname in ctors:
                        hits += [f"{p}:{n.lineno} {fname}({k.arg}=)"
                                 for k in n.keywords if k.arg in names]
    assert hits == []


def test_dcn_positional_legacy_node_count_names_its_replacement():
    """0.5's first positional DCN argument was num_space_nodes."""
    from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine

    with pytest.raises(ValidationError, match=r"GridConfig\(points"):
        DCNPDEEngine(801)


def test_use_rannacher_replacement_covers_the_2d_event_damping():
    """0.5's use_rannacher=False also switched off 2D event damping, so the
    migration text must say how to keep that."""
    removed = _removed_maps()[0]
    assert "event_damping_steps=0" in removed["use_rannacher"]
