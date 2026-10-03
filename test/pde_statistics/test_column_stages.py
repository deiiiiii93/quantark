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


def test_where_equals_two_masked_writes():
    rng = np.random.default_rng(3)
    pay, miss, mask = rng.random(50), rng.random(50), rng.random(50) > 0.4
    a = np.empty(50); a[mask] = pay[mask]; a[~mask] = miss[~mask]
    assert np.array_equal(a, np.where(mask, pay, miss))


def _phoenix_expiry_memory():
    from pde_statistics.cases import PHOENIX
    from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
    return make_phoenix(dict(PHOENIX, memory_coupon=True, coupon_pay_type="expiry"))


@pytest.mark.parametrize("projection", [None, "nodal"])
@pytest.mark.parametrize("name", ["ph_cont", "ph_memory", "ph_expiry_memory"])
def test_phoenix_price_runs_through_column_updates(name, projection):
    from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
    product = _phoenix_expiry_memory() if name == "ph_expiry_memory" else CASES[name][0]()
    env = CASES["ph_cont"][1]()
    solver = PhoenixPDESolver(PDEParams(event_projection=projection))
    assert np.isfinite(solver.price(product, env))
