"""LocalVolBarrierPDESolver on the core (spec 3C §5.3; §7.2 rows 'local-vol barrier, constant surface', 'book
surface', 'EXPIRY', 'knock-in rebate', 'contract multiplier', 'session grid' (v), 'maturity', 'session parity')."""
import json
from pathlib import Path

import pytest

from pde_routes.barrier_reference import barrier_reference
from pde_routes.cases import CONST, analytic_barrier, barrier, dupire, env_book, env_const, obs_dates, surface_const
from pde_routes.sessions import GRID, PV, session
from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import LocalVolBarrierPDESolver

M = dict(vol=CONST["vol"], r=CONST["rate"], q=CONST["div"])
BANK = json.loads((Path(__file__).parent / "reference_values.json").read_text())["lv_book"]


def solver(surface=None):
    return LocalVolBarrierPDESolver(local_vol_surface=surface or surface_const())


@pytest.mark.parametrize("kind,call,level", [("UO", True, 130.0), ("DO", False, 85.0), ("UI", True, 130.0)])
def test_continuous_constant_surface(kind, call, level):
    p = barrier(kind, call, level)
    assert abs(solver().price(p, env_const()) - analytic_barrier(p, env_const())) <= 1e-3   # measured 4.1e-4


@pytest.mark.parametrize("freq,bound", [("monthly", 1e-3), ("daily", 3e-3)])
def test_discrete_constant_surface(freq, bound):
    p = barrier("UO", True, 130.0, obs=freq)
    exact = barrier_reference(kind="UO", call=True, strike=100.0, level=130.0, rebate=0.0, at_hit=False,
                              obs=obs_dates(freq), spots=[100.0], **M)[100.0][0]
    assert abs(solver().price(p, env_const()) - exact) <= bound                   # 2.5e-4 / 1.2e-3; today 1.1e-2


def test_expiry_constant_surface():
    p = barrier("UO", True, 130.0, obs="expiry")
    exact = barrier_reference(kind="UO", call=True, strike=100.0, level=130.0, rebate=0.0, at_hit=False, obs=[1.0],
                              spots=[100.0], **M)[100.0][0]
    assert abs(solver().price(p, env_const()) - exact) <= 1e-3


def test_book_surface_against_the_kernel_oracle():
    surface = dupire(env_book())
    for cell in (c for c in BANK if c["family"] in ("continuous", "monthly", "expiry")):
        p = barrier(cell["kind"], cell["call"], cell["level"], obs=cell["obs"], rebate=cell["rebate"])
        assert abs(solver(surface).price(p, env_book(cell["spot"])) - cell["value"]) <= 2e-3, cell


def test_knock_in_rebate_constant_surface():
    p = barrier("UI", True, 130.0, rebate=5.0)
    assert abs(solver().price(p, env_const()) - analytic_barrier(p, env_const())) <= 1e-3   # the kernel prices it too


def test_greeks_keys_and_multiplier():
    g = solver().calculate_greeks(barrier("UO", True, 130.0), env_const())
    assert set(g) == {"price", "delta", "gamma"}                                  # decision 21
    one, ten = barrier("UO", True, 130.0, rebate=5.0), barrier("UO", True, 130.0, rebate=5.0)
    ten.contract_multiplier = 10.0
    assert abs(solver().price(ten, env_const()) - 10.0 * solver().price(one, env_const())) <= 1e-12 * 10 * abs(
        solver().price(one, env_const()))


def test_knock_in_grid_on_a_constant_surface():
    """§7.2 session grid (v): the composed local-vol grid against the closed forms of Task 10 (i)."""
    from pde_routes.test_session_grid import BOUND, reference
    rows = solver().calculate_spot_greeks_curve(barrier("UI", True, 130.0), env_const(), [80.0, 100.0, 120.0, 131.0])
    for row in rows:
        p, d, g = reference("cont_ui_call_130", row["spot"])
        assert abs(row["price"] - p) <= BOUND["price"] and abs(row["delta"] - d) <= BOUND["delta"]


def test_maturity_and_session_parity():
    from pde_bitwise.capture_book import _aged_to_maturity
    p = _aged_to_maturity(barrier("UO", True, 130.0, rebate=5.0))
    assert abs(solver().price(p, env_const(135.0)) - 5.0) <= 1e-12                # the kernel raises at tau = 0
    q, env = barrier("UO", True, 130.0), env_const()
    assert session(solver(), q, env, {PV}) == solver().price(q, env)
    out = session(solver(), q, env, {PV, GRID}, levels=[100.0, 125.0, 135.0])
    assert out.pv == solver().price(q, env)
    assert list(out.grid) == solver().calculate_spot_greeks_curve(q, env, [100.0, 125.0, 135.0])
