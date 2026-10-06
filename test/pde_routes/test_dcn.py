"""DCNPDEEngine on the core (spec 3C §5.4; §7.2 rows 'DCN price', 'DCN Greeks', 'term structure' (DCN), 'session
parity'; Review Focus 5)."""
import numpy as np
import pytest

from pde_routes.cases import dcn_case
from pde_routes.dcn_reference import dcn_exact
from pde_routes.sessions import GRID, PV, session
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine, DCNPDEResult
from quantark.asset.equity.param import PDEParams
from quantark.util.exceptions import ValidationError

MONEYNESS = (0.70, 0.76, 0.85, 1.00, 1.05, 1.15)          # $E/d3_core_vs_today.py
BOUND_BP = {"fast": 1.0, "standard": 1.0, "high": 0.3}    # measured 0.38 / 0.35 / 0.10


def bp(err, product):
    return 1e4 * abs(err) / product.notional


@pytest.mark.parametrize("accuracy", ["fast", "standard", "high"])
@pytest.mark.parametrize("name", ["A", "B", "SA", "SB"])
def test_price_against_the_exact_recursion(name, accuracy):
    product, env = dcn_case(name)
    spots = [m * product.initial_price for m in MONEYNESS]
    exact = dcn_exact(product, env, spots=spots)["pv"]
    engine = DCNPDEEngine(PDEParams(accuracy=accuracy))
    for spot, want in zip(spots, exact):
        _, e = dcn_case(name, spot=spot)
        assert bp(engine.price(product, e) - want, product) <= BOUND_BP[accuracy], (name, spot)


@pytest.mark.parametrize("name", ["A", "B", "SA", "SB"])
def test_grid_delta(name):
    product, env = dcn_case(name)
    spots = [m * product.initial_price for m in MONEYNESS]
    exact = dcn_exact(product, env, spots=spots)
    for spot, d in zip(spots, exact["delta"]):
        _, e = dcn_case(name, spot=spot)
        g = DCNPDEEngine().calculate_greeks(product, e)
        assert abs(g["delta"] - d) <= 1e-3 * max(abs(d), 1e-12) + 1e-9, (name, spot)   # 0.1 % (measured 0.040 %)


def test_price_detailed_reports_the_delivered_steps():
    product, env = dcn_case("A")
    res = DCNPDEEngine().price_detailed(product, env)
    assert isinstance(res, DCNPDEResult) and res.num_time_steps > 485        # 2,953 at standard (decision 20)
    with pytest.raises(ValidationError, match="removed in 0.10.0.*GridConfig"):
        DCNPDEEngine().price_detailed(product, env, layout=None)              # a removed name (REMOVED_DCN_NAMES)


def test_nodal_projection_applies_the_legacy_masks():
    product, env = dcn_case("A")
    nodal = DCNPDEEngine(PDEParams(event_projection="nodal")).price(product, env)
    assert nodal != DCNPDEEngine().price(product, env)                       # decision 23


def test_session_parity():
    product, env = dcn_case("A")
    engine = DCNPDEEngine()
    assert session(engine, product, env, {PV}) == engine.price(product, env)
    levels = [m * product.initial_price for m in (0.8, 1.0, 1.1)]
    out = session(engine, product, env, {PV, GRID}, levels=levels)
    assert out.pv == engine.price(product, env)
    assert list(out.grid) == engine.calculate_spot_greeks_curve(product, env, levels)


def test_dcn_knocked_in_session_grid():
    """Review Focus 5 (P8): flagged knocked in at valuation, every row is read from the ki surface."""
    product, env = dcn_case("A", knocked_in_at_valuation=True)
    engine = DCNPDEEngine()
    levels = [m * product.initial_price for m in (0.7, 1.0, 1.2)]
    out = session(engine, product, env, {PV, GRID}, levels=levels)
    assert list(out.grid) == engine.calculate_spot_greeks_curve(product, env, levels)
