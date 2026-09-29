"""Tier-1 tests for GridConfig + accuracy profiles (spec §4.7)."""

import pytest

from quantark.asset.equity.engine.pde.grid import GridConfig, resolve_config
from quantark.util.exceptions import ValidationError


def test_resolve_standard_fully_populated():
    c = resolve_config("standard", None)
    assert c.points == 400 and c.steps_per_day == 4.0 and c.day_count == 252
    assert c.terminal_damping_steps == 1 and c.event_damping_steps == 2
    assert c.eps_crit == 0.003 and c.num_std == 4.0
    assert c.max_points == 2000 and c.max_steps == 5000
    assert c.bounds == (None, None)


def test_profiles_differ_as_specified():
    fast, high = resolve_config("fast", None), resolve_config("high", None)
    assert fast.points == 200 and fast.steps_per_day == 4.0
    assert high.points == 800 and high.steps_per_day == 8.0 and high.eps_crit == 0.002


def test_field_by_field_override():
    c = resolve_config("standard", GridConfig(steps_per_day=8.0))
    assert c.steps_per_day == 8.0 and c.points == 400  # only that field moved


def test_unknown_accuracy_rejected():
    with pytest.raises(ValidationError):
        resolve_config("ultra", None)


def test_points_over_max_rejected():
    with pytest.raises(ValidationError):
        resolve_config("standard", GridConfig(points=4000))  # > max_points 2000


def test_negative_counts_rejected():
    with pytest.raises(ValidationError):
        resolve_config("standard", GridConfig(event_damping_steps=-1))
    with pytest.raises(ValidationError):
        resolve_config("standard", GridConfig(points=0))


def test_per_side_bounds_kept():
    c = resolve_config("standard", GridConfig(bounds=(50.0, None)))
    assert c.bounds == (50.0, None)
    with pytest.raises(ValidationError):
        resolve_config("standard", GridConfig(bounds=(120.0, 80.0)))


def test_pdeparams_front_door():
    from quantark.asset.equity.param import PDEParams

    p = PDEParams(accuracy="high", grid=GridConfig(points=600))
    assert p.accuracy == "high" and p.grid.points == 600
    with pytest.raises(ValidationError):
        PDEParams(accuracy="ultra")
    with pytest.raises(ValidationError):
        PDEParams(grid="not-a-config")


def test_key_fingerprint():
    assert resolve_config("fast", None).key == resolve_config("fast", None).key
    assert resolve_config("fast", None).key != resolve_config("high", None).key
    assert (
        resolve_config("standard", GridConfig(points=444)).key
        != resolve_config("standard", None).key
    )


def test_legacy_resolution_knobs_rejected_on_grid_layer():
    """Non-default legacy knobs must fail LOUDLY on grid-layer solvers
    (silently-inert resolution controls are a mispricing hazard), while the
    standalone vol-model solvers keep consuming them."""
    from quantark.asset.equity.engine.pde import EuropeanPDESolver, SnowballPDESolver
    from quantark.asset.equity.param import PDEParams

    for kwargs in (
        {"grid_size": 120},
        {"time_steps": 60},
        {"s_min": 50.0},
        {"s_max": 200.0},
    ):
        solver = EuropeanPDESolver(params=PDEParams(**kwargs))
        with pytest.raises(ValidationError, match="legacy resolution control"):
            _ = solver.grid_binder

    # default values pass, and the replacement path works
    assert SnowballPDESolver(params=PDEParams()).grid_binder.config.points == 400
    assert (
        EuropeanPDESolver(
            params=PDEParams(grid=GridConfig(points=120))
        ).grid_binder.config.points
        == 120
    )


def test_pde_param_profiles_target_grid_layer():
    """make_pde_params presets must configure accuracy/grid (the declarative
    layer), not the legacy inert knobs."""
    from quantark.asset.equity.param import make_pde_params

    assert make_pde_params("fast").accuracy == "fast"
    assert make_pde_params("accurate").accuracy == "high"
    bs = make_pde_params("barrier_sensitive")
    assert bs.grid.points == 600 and bs.grid.steps_per_day == 6.0
    # presets compose with the grid layer without rejection
    from quantark.asset.equity.engine.pde import EuropeanPDESolver

    assert EuropeanPDESolver(params=bs).grid_binder.config.points == 600


# ---- 0.6.0: time_steps budget, one-basis rule, route tables, evidence rule ----

import dataclasses  # noqa: E402

from quantark.asset.equity.engine.pde.grid.config import (  # noqa: E402
    ACCURACY_PROFILES,
    DCN_ACCURACY_PROFILES,
    LV_UNIFORM_ACCURACY_PROFILES,
)
from quantark.modelvalidation.engine_config import engine_config  # noqa: E402


def test_positional_construction_raises_type_error():
    with pytest.raises(TypeError):
        GridConfig(400)


def test_key_has_eleven_entries():
    assert len(resolve_config("standard", None).key) == 11


def test_shared_tables_unchanged():
    assert ACCURACY_PROFILES == {
        "fast": dict(points=200, steps_per_day=4.0, eps_crit=0.003),
        "standard": dict(points=400, steps_per_day=4.0, eps_crit=0.003),
        "high": dict(points=800, steps_per_day=8.0, eps_crit=0.002),
    }


def test_resolved_shared_config_has_density_basis_only():
    cfg = resolve_config("standard", None)
    assert cfg.steps_per_day == 4.0 and cfg.time_steps is None


def test_evidence_omits_unset_basis_and_nothing_else():
    cfg = resolve_config("standard", None)
    ev = cfg.evidence()
    assert "time_steps" not in ev
    assert list(ev) == [
        f.name for f in dataclasses.fields(GridConfig) if f.name != "time_steps"
    ]
    budget = resolve_config("standard", GridConfig(time_steps=300))
    ev_b = budget.evidence()
    assert ev_b["time_steps"] == 300 and "steps_per_day" not in ev_b


def test_engine_config_of_banked_grid_equals_legacy_asdict():
    # decision 14: a banked (density) resolved grid serializes exactly as before
    cfg = resolve_config("standard", None)
    legacy = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in dataclasses.asdict(cfg).items()
        if k != "time_steps"
    }
    assert engine_config(cfg) == legacy
    assert list(engine_config(cfg)) == list(legacy)


def test_overlay_time_steps_clears_preset_density():
    cfg = resolve_config("standard", GridConfig(time_steps=500))
    assert cfg.time_steps == 500 and cfg.steps_per_day is None


def test_overlay_density_clears_route_budget():
    cfg = resolve_config(
        "standard", GridConfig(steps_per_day=2.0), profiles=LV_UNIFORM_ACCURACY_PROFILES
    )
    assert cfg.steps_per_day == 2.0 and cfg.time_steps is None


def test_overlay_with_both_bases_raises():
    with pytest.raises(ValidationError, match="both"):
        resolve_config("standard", GridConfig(steps_per_day=4.0, time_steps=100))


def test_overlay_day_count_with_budget_raises():
    with pytest.raises(ValidationError, match="day_count"):
        resolve_config("standard", GridConfig(day_count=244, time_steps=100))
    with pytest.raises(ValidationError, match="day_count"):
        # the route's basis is the budget: a day basis has nothing to act on
        resolve_config(
            "standard", GridConfig(day_count=244), profiles=LV_UNIFORM_ACCURACY_PROFILES
        )


def test_explicit_budget_over_cap_raises():
    with pytest.raises(ValidationError, match="max_steps"):
        resolve_config("standard", GridConfig(time_steps=6000))


def test_lv_table():
    expected = {"fast": (200, 200), "standard": (400, 200), "high": (800, 400)}
    for acc, (pts, n) in expected.items():
        cfg = resolve_config(acc, None, profiles=LV_UNIFORM_ACCURACY_PROFILES)
        assert (cfg.points, cfg.time_steps, cfg.steps_per_day) == (pts, n, None)


def test_dcn_table_standard_is_todays_plan():
    cfg = resolve_config("standard", None, profiles=DCN_ACCURACY_PROFILES)
    assert (cfg.points, cfg.steps_per_day, cfg.max_points) == (801, 0.1, 2000)
    assert (cfg.terminal_damping_steps, cfg.event_damping_steps) == (2, 2)
    assert (cfg.eps_crit, cfg.num_std, cfg.max_steps, cfg.day_count) == (0.003, 4.0, 5000, 252)
    assert [
        resolve_config(a, None, profiles=DCN_ACCURACY_PROFILES).points
        for a in ("fast", "standard", "high")
    ] == [401, 801, 1601]


def test_resolved_density_config_reresolves_to_itself():
    # Review Focus 3: the intraday route and the builders feed a resolved config back
    cfg = resolve_config("high", GridConfig(points=900, max_points=2000))
    assert resolve_config("standard", cfg) == cfg


def test_with_time_steps_on_resolved_config_resolves():
    # Review Focus 3: day_count cleared with the density, so no conflict
    cfg = resolve_config("standard", None).with_time_steps(1234)
    assert cfg.day_count is None and cfg.steps_per_day is None
    again = resolve_config("standard", cfg)
    assert again.time_steps == 1234 and again.day_count == 252


def test_with_steps_per_day_clears_budget():
    cfg = resolve_config("standard", GridConfig(time_steps=300)).with_steps_per_day(6.0)
    assert cfg.time_steps is None and cfg.steps_per_day == 6.0
