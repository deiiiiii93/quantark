"""A bump context freezes one layout per composite part (spec 3C §5.2 (c), decision 10).

A continuous knock-in is participation x vanilla + R DF - knock-out; the knock-out leg is hard-edged at the
barrier. A single frozen layout bound for the knock-in itself has no hard edge, so the leg refused it and every
continuous knock-in bump Greek raised PricingError. Each part now keeps its own base-market layout.
"""
from copy import deepcopy

import pytest

import pde_bitwise.capture_book as book
from quantark.asset.equity.engine import PDEEngine
from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.riskmeasures import GreeksCalculator

# --- copied verbatim from docs/superpowers/specs/2026-10-04-pde-1d-remaining-routes-evidence/m1_knock_in_bump.py
# (the both_frozen arm's continuous cells, section 3 of that study) -------------------------------------------------


def with_spot(env, spot):
    out = deepcopy(env)
    out.spot_quote.spot = float(spot)
    return out


def pointwise(product, env, engine, h=1e-4):
    """Closed-form price, delta and gamma by a small central difference (h relative)."""
    s = float(env.spot)
    p0, pu, pd = (engine.price(product, with_spot(env, s * f)) for f in (1.0, 1.0 + h, 1.0 - h))
    return {"price": float(p0), "delta": float((pu - pd) / (2 * h * s)), "gamma": float((pu - 2 * p0 + pd) / (h * s) ** 2)}


analytic = {"up_in_call": BarrierAnalyticalEngine, "down_in_put": BarrierAnalyticalEngine,
            "double_in": DoubleBarrierOptionAnalyticalEngine}
spots = {"up_in_call": (100.0, 115.0, 125.0, 128.5), "down_in_put": (100.0, 90.0, 83.0, 81.0),
         "double_in": (100.0, 90.0, 110.0, 117.0)}
# ------------------------------------------------------------------------------------------------------------------

CELLS = [{"name": f"{name}|{spot}", "product": (lambda n=name: book.knock_in_products()[n][0]),
          "env": (lambda s=spot: with_spot(book.env_flat(), s)), "analytic": analytic[name]}
         for name in spots for spot in spots[name]]

BOUNDS = dict(delta=3e-4, gamma=1e-3, vega=1.5e-4, theta=5e-6, rho=8e-5)


@pytest.mark.parametrize("cell", CELLS, ids=[c["name"] for c in CELLS])
def test_knock_in_bump_greeks_against_closed_forms(cell):
    product, env = cell["product"](), cell["env"]()
    pde = GreeksCalculator().calculate_numerical_greeks(product=product, pricing_env=env, engine=PDEEngine())
    ref = GreeksCalculator().calculate_numerical_greeks(product=product, pricing_env=env, engine=cell["analytic"]())
    for k, bound in BOUNDS.items():
        assert abs(pde[k] - ref[k]) <= bound, (cell["name"], k)     # today 8 of 9 byte-book rows raise PricingError


def test_bump_context_freezes_one_layout_per_part():
    from pde_routes.cases import barrier, env_const
    from quantark.asset.equity.engine.pde import BarrierPDESolver
    solver = BarrierPDESolver()
    ctx = solver.create_bump_context(barrier("UI", True, 130.0, rebate=5.0), env_const(100.0))
    assert set(ctx._frozen_parts) == {"vanilla", "knock_out"} and ctx._frozen_base_layout is None
    knocked = solver.create_bump_context(barrier("UI", True, 130.0), env_const(135.0))
    assert set(knocked._frozen_parts) == {"vanilla"}        # no knock-out leg at the base: a crossing bump binds fresh


def test_an_expired_knock_in_has_a_bump_context():
    """Final review I3: decision 30 builds no part product at maturity, so a bump context of an expired knock-in
    freezes nothing and prices as the solver does (create_bump_context raised building a zero-maturity vanilla)."""
    from pde_routes.cases import env_const
    from pde_routes.test_maturity import at, single
    from quantark.asset.equity.engine.pde import BarrierPDESolver
    from quantark.util.enum import BarrierType, ObservationType
    product, env = at(single(BarrierType.UP_IN), ObservationType.CONTINUOUS, 0.0), env_const(125.0)
    solver = BarrierPDESolver()
    assert solver.create_bump_context(product, env).price(product, env) == solver.price(product, env)
