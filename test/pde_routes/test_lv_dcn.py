"""LocalVolDCNPDEEngine on the core (spec 3C §5.4; §7.2 rows 'local-vol DCN, constant surface', '..., skewed surface')."""
import json
from pathlib import Path

import numpy as np
import pytest

from pde_routes.cases import dcn_case, dcn_flat_surface, dcn_skew_surface
from pde_routes.sessions import GRID, PV, session
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import LocalVolDCNPDEEngine

S0 = 6000.0
FLAT, SKEW = dcn_flat_surface(), dcn_skew_surface()
FINEST = json.loads((Path(__file__).parent / "reference_values.json").read_text())["dcn_skew_finest"]["A"]


@pytest.mark.parametrize("m", [0.76, 1.00, 1.05])
def test_constant_surface_equals_the_flat_engine(m):
    product, env = dcn_case("A", spot=m * S0)
    lv = LocalVolDCNPDEEngine(local_vol_surface=FLAT).price(product, env)
    assert abs(lv - DCNPDEEngine().price(product, env)) <= 1e-5                    # measured 1.1e-7 (absolute)


def test_skewed_surface_against_the_finest_rung():
    for cell in FINEST:
        product, env = dcn_case("A", spot=cell["moneyness"] * S0)
        got = LocalVolDCNPDEEngine(local_vol_surface=SKEW).price(product, env)
        assert 1e4 * abs(got - cell["value"]) / product.notional <= 0.5, cell        # measured 0.18 bp


def test_session_parity():
    product, env = dcn_case("A")
    engine = LocalVolDCNPDEEngine(local_vol_surface=SKEW)
    assert session(engine, product, env, {PV}) == engine.price(product, env)
    levels = [m * product.initial_price for m in (0.8, 1.0)]
    out = session(engine, product, env, {PV, GRID}, levels=levels)
    assert list(out.grid) == engine.calculate_spot_greeks_curve(product, env, levels)
