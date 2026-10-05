"""LocalVolPDESolver on the core (spec 3C §5.3; §7.2 rows 'local-vol vanilla, constant surface', '..., book surface',
'local-vol vanilla Greeks', 'one-step nodal solve', 'session parity', 'contract multiplier')."""
import json
from pathlib import Path

import pytest

from pde_routes.cases import CONST, black_scholes, env_book, env_const, surface_const, vanilla
from pde_routes.sessions import GRID, PV, session
from quantark.asset.equity.engine.pde import EuropeanPDESolver
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare
from quantark.asset.equity.engine.pde.grid import GridConfig
from quantark.asset.equity.engine.pde.local_vol_pde_solver import LocalVolPDESolver
from quantark.asset.equity.param import PDEParams

BANK = json.loads((Path(__file__).parent / "reference_values.json").read_text())["lv_book"]
CELLS = [(call, k, T) for call in (True, False) for k in (80.0, 100.0, 120.0) for T in (0.25, 1.0, 2.0)]


@pytest.mark.parametrize("call,strike,T", CELLS)
def test_constant_surface_against_black_scholes(call, strike, T):
    got = LocalVolPDESolver(local_vol_surface=surface_const()).price(vanilla(call, strike, T), env_const())
    assert abs(got - black_scholes(call, 100.0, strike, T)[0]) <= 1e-3           # measured 2.5e-4; today 4.6e-3


def test_book_surface_against_the_kernel_oracle():
    from pde_routes.cases import dupire
    surface = dupire(env_book())
    for cell in (c for c in BANK if c["family"] == "vanilla"):
        p = vanilla(cell["call"], cell["strike"], cell["T"])
        got = LocalVolPDESolver(local_vol_surface=surface).price(p, env_book(cell["spot"]))
        assert abs(got - cell["value"]) <= 2e-3, cell                              # measured 4.0e-4


def test_greeks_keep_rho_and_theta():
    p, env = vanilla(True, 100.0, 1.0), env_const()
    g = LocalVolPDESolver(local_vol_surface=surface_const()).calculate_greeks(p, env)
    assert set(g) == {"price", "delta", "gamma", "rho", "theta"}                   # decision 7
    assert abs(g["delta"] - black_scholes(True, 100.0, 100.0, 1.0)[1]) <= 1e-4     # measured 4.7e-6
    bump = PDEParams().get_effective_bump_config()
    r, q, vol = CONST["rate"], CONST["div"], CONST["vol"]
    rho = (black_scholes(True, 100.0, 100.0, 1.0, r=r + bump.rate_bump)[0]
           - black_scholes(True, 100.0, 100.0, 1.0, r=r - bump.rate_bump)[0]) / (2 * bump.rate_bump) / 100.0
    dt = bump.time_bump_days / 365.0
    theta = (black_scholes(True, 100.0, 100.0, 1.0 - dt)[0] - black_scholes(True, 100.0, 100.0, 1.0)[0]) / (dt * 365.0)
    assert abs(g["rho"] - rho) <= 5e-5                                             # measured 1.1e-5
    assert abs(g["theta"] - theta) <= 5e-6                                         # measured 1.2e-6


def test_one_step_solve_marches_nodal_sets():
    from pde_routes.cases import dupire
    params = PDEParams(grid=GridConfig(time_steps=1))
    solver, p, env = LocalVolPDESolver(params, local_vol_surface=dupire(env_book())), vanilla(True, 100.0, 1.0), env_book()
    problem = solver._with_surface(env, lambda: prepare(solver, p, env, OutputRequest(Operation.PRICE)))
    assert problem.coefficients.sets is not None
    assert solver.price(p, env) != EuropeanPDESolver(params).price(p, env)        # today: equal (5.676243)


def test_session_parity():
    solver, p, env = LocalVolPDESolver(local_vol_surface=surface_const()), vanilla(True, 100.0, 1.0), env_const()
    assert session(solver, p, env, {PV}) == solver.price(p, env)
    out = session(solver, p, env, {PV, GRID}, levels=[90.0, 100.0, 110.0])
    assert out.pv == solver.price(p, env)
    assert list(out.grid) == solver.calculate_spot_greeks_curve(p, env, [90.0, 100.0, 110.0])


def test_maturity_prices():
    from pde_bitwise.capture_book import _aged_to_maturity
    p = _aged_to_maturity(vanilla(True, 100.0, 1.0))
    g = LocalVolPDESolver(local_vol_surface=surface_const()).calculate_greeks(p, env_const(125.0))
    assert set(g) == {"price", "delta", "gamma"} and abs(g["price"] - 25.0) <= 1e-12   # P9; the kernel raises


def test_no_route_row_and_explain_runs():
    solver = LocalVolPDESolver(local_vol_surface=surface_const())
    assert solver.adapter.route is None                       # every PDEParams field honoured (no applicability row)
    solver.explain(vanilla(True, 100.0, 1.0), env_const())
