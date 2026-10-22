from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.asset.equity.report.autocallable_risk_report import build_snowball_risk_snapshot
from quantark.asset.equity.report.surfaces import GridSpec, compute_surfaces_from_pv
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment


class DummyEngine(BaseEngine):
    def price(self, product, pricing_env) -> float:
        return float(pricing_env.spot) * 0.01


def test_surface_derivatives_match_known_function():
    # V(S,q) = S*q
    spot_grid = np.linspace(60.0, 120.0, 31)
    q_grid = np.linspace(-0.02, 0.08, 21)
    vol_grid = np.linspace(0.15, 0.25, 11)

    pv_sq = np.outer(spot_grid, q_grid)

    # Dummy spot-vol surface: independent of q → rhoq_sv must be ~0
    pv_sv = np.outer(spot_grid, vol_grid)
    pv_sv_q_up = pv_sv.copy()

    surfaces = compute_surfaces_from_pv(
        spot_grid=spot_grid,
        q_grid=q_grid,
        vol_grid=vol_grid,
        pv_sq=pv_sq,
        pv_sv=pv_sv,
        pv_sv_q_up=pv_sv_q_up,
        q_bump_for_rho=1e-4,
    )

    # Delta = q (broadcasted over spot)
    expected_delta = np.tile(q_grid, (spot_grid.size, 1))
    assert np.max(np.abs(surfaces.delta_sq - expected_delta)) < 1e-6

    # RhoQ = dV/dq * 0.01 = S * 0.01
    expected_rhoq = np.tile(spot_grid.reshape(-1, 1) * 0.01, (1, q_grid.size))
    assert np.max(np.abs(surfaces.rhoq_sq - expected_rhoq)) < 1e-6

    # Mixed partial d^2V/(dS dq) = 1
    assert np.max(np.abs(surfaces.v_sq - 1.0)) < 1e-6

    # Spot-vol rhoq should be zero for q-independent PV
    assert np.max(np.abs(surfaces.rhoq_sv)) < 1e-12


def test_snowball_risk_snapshot_serializes_full_surface_suite():
    product = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        ki_barrier=75.0,
        num_observations=4,
        include_principal=False,
    )
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )

    snapshot = build_snowball_risk_snapshot(
        product=product,
        pricing_env=env,
        engine=DummyEngine(),
        label="2024-01-01",
        grid_spec=GridSpec(spot_nodes=5, q_nodes=5, vol_nodes=5),
    )

    surface_keys = {surface["key"] for surface in snapshot["surfaces"]}
    assert "rhoq_spot_div" in surface_keys
    assert "rhob_spot_div" in surface_keys
    assert "cross_s_q" in surface_keys
    assert "vanna_spot_vol" in surface_keys
    assert "volga_spot_vol" in surface_keys
    rhoq_surface = next(
        surface for surface in snapshot["surfaces"] if surface["key"] == "rhoq_spot_div"
    )
    assert len(rhoq_surface["z"]) == 5
    assert len(rhoq_surface["z"][0]) == 5
    assert snapshot["scenario_ladder"]["worst_pnl"] is not None
    assert len(snapshot["bucketed_greeks"]) > 0
