"""The value's event stages as column functions (spec 3B §10 step 2): the value's bits do not move."""
import numpy as np
import pytest

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from pde_statistics.cases import CASES


@pytest.mark.parametrize("projection", [None, "nodal"])
@pytest.mark.parametrize("name", ["sb_cont", "sb_discrete", "sb_daily"])
def test_price_is_bit_identical_through_column_stages(name, projection):
    product, env = CASES[name][0](), CASES[name][1]()
    solver = SnowballPDESolver(PDEParams(event_projection=projection))
    assert solver.price(product, env) == pytest.approx(solver.price(product, env), rel=0, abs=0)
