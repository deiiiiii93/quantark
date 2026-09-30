"""The literal group table of spec 2 §9.3: each family's declared column groups
equal the solve calls its legacy march made. Filled in family by family."""
import pytest

from quantark.asset.equity.engine.pde import (AmericanPDESolver, BarrierPDESolver, DoubleBarrierPDESolver,
                                             DoubleOneTouchPDESolver, EuropeanPDESolver, OneTouchPDESolver)
from quantark.asset.equity.engine.pde.core.groups import VALUE, Backend
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare
from quantark.asset.equity.param import PDEParams

from pde_bitwise.capture_book import (american_tenor, barrier, double_barrier, double_one_touch,
                                      env_flat, european, one_touch)
from quantark.util.enum import ObservationType

# (solver, product) -> [(group name, surfaces, backend, policy)] -- the historical solve calls
BASE_ROWS = [
    (EuropeanPDESolver, european, [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
    (AmericanPDESolver, american_tenor, [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
    (BarrierPDESolver, lambda: barrier(ObservationType.CONTINUOUS), [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
    (DoubleBarrierPDESolver, double_barrier, [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
    (OneTouchPDESolver, one_touch, [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
    (DoubleOneTouchPDESolver, double_one_touch, [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
]


@pytest.mark.parametrize("cls,make,want", BASE_ROWS)
def test_group_declaration_equals_the_historical_solve_calls(cls, make, want):
    p = prepare(cls(PDEParams()), make(), env_flat(), OutputRequest(Operation.PRICE))
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in p.groups] == want


# --- the snowball family (Task 6) ------------------------------------------------
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.grid import GridConfig
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver

from pde_bitwise.capture_book import env_grid_vol, snowball

SNOWBALL_ROWS = [
    (SnowballPDESolver, lambda: snowball("continuous"), env_flat,
     [("value", ("alive", "ki"), Backend.BANDED, VALUE)]),
    (LocalVolSnowballPDESolver, lambda: snowball("continuous"), env_grid_vol,
     [("value", ("alive", "ki"), Backend.BANDED, VALUE)]),
]


@pytest.mark.parametrize("cls,make,env,want", SNOWBALL_ROWS)
def test_snowball_group_declaration_equals_the_historical_solve_calls(cls, make, env, want):
    p = prepare(cls(PDEParams()), make(), env(), OutputRequest(Operation.PRICE))
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in p.groups] == want


def test_snowball_with_two_interior_nodes_falls_back_to_two_lu_groups():
    # measured at 302fa2ed: points=4, eps_crit=1.0 prices (4 nodes, 2 interior); points=3 is rejected
    s = SnowballPDESolver(PDEParams(grid=GridConfig(points=4, eps_crit=1.0)))
    p = prepare(s, snowball("continuous"), env_flat(), OutputRequest(Operation.PRICE))
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in p.groups] == [
        ("alive", ("alive",), Backend.SPARSE_LU, VALUE), ("ki", ("ki",), Backend.SPARSE_LU, VALUE)]


# --- KO-reset (Task 7) ---------------------------------------------------------------
def test_ko_reset_group_declaration_equals_the_historical_solve_calls():
    from quantark.asset.equity.engine.pde.core.adapters.ko_reset import KOResetAdapter
    from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
    from pde_bitwise.capture_book import KO_RESET, make_ko_reset

    p = prepare(KOResetSnowballPDESolver(PDEParams()), make_ko_reset(KO_RESET), env_flat(),
                OutputRequest(Operation.PRICE))
    assert type(p.adapter) is KOResetAdapter
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in p.groups] == [
        ("value", ("alive", "ki"), Backend.BANDED, VALUE)]


# --- Phoenix (Task 8) ----------------------------------------------------------------
from quantark.asset.equity.engine.pde import PhoenixPDESolver
from quantark.asset.equity.engine.pde.core.groups import VALUE_NO_ZERO_DIFFUSION_OVERRIDE
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix

from pde_bitwise.capture_book import PHOENIX, phoenix_plain


def _phoenix_groups(max_k, termination, banded=True):
    b = Backend.BANDED if banded else Backend.SPARSE_LU
    g = [(f"v0[{k}]", (f"v0[{k}]",), b, VALUE_NO_ZERO_DIFFUSION_OVERRIDE) for k in range(max_k + 1)]
    g += [(f"v1[{k}]", (f"v1[{k}]",), b, VALUE_NO_ZERO_DIFFUSION_OVERRIDE) for k in range(max_k + 1)]
    if termination:
        g += [("w0", ("w0",), b, VALUE_NO_ZERO_DIFFUSION_OVERRIDE), ("w1", ("w1",), b, VALUE_NO_ZERO_DIFFUSION_OVERRIDE)]
    return g


def test_phoenix_memory_groups():
    # max_k = num_obs with memory coupons (_solve): the book's PHOENIX has 12 observations
    p = prepare(PhoenixPDESolver(PDEParams()), make_phoenix(PHOENIX), env_flat(), OutputRequest(Operation.PRICE))
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in p.groups] == _phoenix_groups(12, False)
    assert all(g.one_dimensional_rhs for g in p.groups)


def test_phoenix_non_memory_groups():
    p = prepare(PhoenixPDESolver(PDEParams()), phoenix_plain(), env_flat(), OutputRequest(Operation.PRICE))
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in p.groups] == _phoenix_groups(0, False)


def test_phoenix_expiry_coupons_add_the_termination_surfaces():
    p = prepare(PhoenixPDESolver(PDEParams()), make_phoenix(dict(PHOENIX, memory_coupon=False, coupon_pay_type="expiry")),
                env_flat(), OutputRequest(Operation.PRICE))
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in p.groups] == _phoenix_groups(0, True)


# --- the statistics group (Task 9) ------------------------------------------------------
from quantark.asset.equity.engine.pde.core.groups import STATISTICS_LEGACY


def test_statistics_group_is_one_banded_group_of_both_surfaces():
    p = prepare(SnowballPDESolver(PDEParams()), snowball("discrete"), env_flat(),
                OutputRequest(Operation.PRICE_WITH_EVENTS, event_stats=True))
    names = [(g.name, g.surfaces, g.backend, g.policy, g.rolling) for g in p.groups]
    assert names == [("value", ("alive", "ki"), Backend.BANDED, VALUE, False),
                     ("stats", ("stats_v0", "stats_v1"), Backend.BANDED, STATISTICS_LEGACY, True)]


def test_statistics_only_request_has_no_value_group():
    p = prepare(SnowballPDESolver(PDEParams()), snowball("discrete"), env_flat(),
                OutputRequest(Operation.EVENT_STATS, price=False, event_stats=True, external_pv=1.0))
    assert [g.name for g in p.groups] == ["stats"]
