"""The definition of done (spec 3C §7.2): ROUTE_RULES keeps only the 2D row, explain reports every field honoured on
the four solvers, and no deleted name imports."""
import importlib

import pytest

from quantark.asset.equity.engine.pde.route_applicability import ROUTE_RULES

from pde_routes.cases import dcn_flat_surface

FLAT_DCN = dcn_flat_surface()


def test_route_rules_keep_only_the_2d_row():
    assert set(ROUTE_RULES) == {"2D"}


@pytest.mark.parametrize("module,name", [
    ("quantark.asset.equity.engine.pde.grid", "GridLayerMixin"),
    ("quantark.asset.equity.engine.pde.grid.config", "LV_UNIFORM_ACCURACY_PROFILES"),
    ("quantark.asset.equity.engine.pde.grid.config", "DCN_ACCURACY_PROFILES"),
    ("quantark.asset.equity.engine.pde.dcn_pde_solver", "_sinh_grid"),
    ("quantark.asset.equity.engine.pde.dcn_pde_solver", "_operator_coeffs"),
    ("quantark.asset.equity.engine.pde.dcn_pde_solver", "_theta_step"),
    ("quantark.volmodels.localvol", "price_barrier_lv_pde"),
    ("quantark.volmodels.localvol.pde_kernel", "price_barrier_lv_pde"),           # the test oracle (decision 14)
    ("quantark.volmodels.localvol.pde_kernel", "_solve_ko_lv"),
    ("quantark.volmodels.heston.pde_kernel", "_deterministic_barrier_price"),     # sigma -> 0 marches on the core
    ("quantark.volmodels.slv.slv_pde_kernel", "_deterministic_slv_barrier"),
    ("quantark.asset.equity.engine.mc.dcn_execution_adapters", "DCNLocalVolPDEAdapter"),
    ("quantark.asset.equity.engine.analytical.double_barrier_option_engine", "DoubleBarrierOptionAnalyticalEngine._prob_inside_at_expiry"),
])
def test_deleted_names_are_gone(module, name):
    obj = importlib.import_module(module)
    for part in name.split("."):
        if not hasattr(obj, part):
            return
        obj = getattr(obj, part)
    pytest.fail(f"{module}.{name} still exists")


def test_damping_rows():
    from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS
    assert "DCN" not in DAMPING_DEFAULTS and "local-vol" not in DAMPING_DEFAULTS


@pytest.mark.parametrize("make", ["lv_vanilla", "lv_barrier", "dcn", "lv_dcn"])
def test_explain_reports_every_field_honoured(make):
    from pde_routes.cases import barrier, dcn_case, env_const, surface_const, vanilla
    from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import LocalVolBarrierPDESolver
    from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
    from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import LocalVolDCNPDEEngine
    from quantark.asset.equity.engine.pde.local_vol_pde_solver import LocalVolPDESolver
    if make == "lv_vanilla":
        solver, p, env = LocalVolPDESolver(local_vol_surface=surface_const()), vanilla(), env_const()
    elif make == "lv_barrier":
        solver, p, env = LocalVolBarrierPDESolver(local_vol_surface=surface_const()), barrier("UO", True, 130.0), env_const()
    elif make == "dcn":
        (p, env), solver = dcn_case("A"), DCNPDEEngine()
    else:
        (p, env), solver = dcn_case("A"), LocalVolDCNPDEEngine(local_vol_surface=FLAT_DCN)
    assert solver.adapter.route is None
    solver.explain(p, env)                                        # raises if a field were still refused
