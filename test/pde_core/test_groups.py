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
