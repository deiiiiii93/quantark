import numpy as np

from asset.equity.report.surfaces import compute_surfaces_from_pv


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

