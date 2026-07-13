"""Surface-aware DCN MC engines (spec WP1.5): flat-surface consistency."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    HestonDCNMCEngine,
    LocalVolDCNMCEngine,
)
from quantark.param import GridVolSurface
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn

PATHS = 2 ** 14


def _grid_env():
    env = flat_env(**FLAT)
    strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
    maturities = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    iv = np.full((len(maturities), len(strikes)), FLAT["sigma"])
    env.vol_surface = GridVolSurface(
        strikes=strikes, maturities=maturities, iv_grid=iv
    )
    return env


def test_local_vol_flat_surface_matches_gbm():
    # On a flat surface, Dupire local vol == flat vol -> LV engine must agree
    # with the GBM engine well within joint MC noise.
    p = make_dcn(DCN_A)
    env = _grid_env()
    gbm = DCNMCEngine(num_paths=PATHS, seed=42).price_detailed(p, env)
    lv = LocalVolDCNMCEngine(num_paths=PATHS, seed=42).price_detailed(p, env)
    assert abs(lv.pv - gbm.pv) < 4.0 * max(lv.std_error, gbm.std_error)


def test_heston_degenerate_params_match_gbm():
    # vol-of-vol -> 0 with v0 = theta = sigma_flat^2 collapses Heston to GBM
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    params = HestonParams(
        v0=FLAT["sigma"] ** 2, kappa=1.0, theta=FLAT["sigma"] ** 2,
        sigma=1e-6, rho=0.0,
    )
    gbm = DCNMCEngine(num_paths=PATHS, seed=42).price_detailed(p, env)
    hz = HestonDCNMCEngine(
        model_params=params, num_paths=PATHS, seed=42
    ).price_detailed(p, env)
    assert abs(hz.pv - gbm.pv) < 4.0 * max(hz.std_error, gbm.std_error)


def test_leg_invariant_holds_for_vol_engines():
    p = make_dcn(DCN_A)
    r = LocalVolDCNMCEngine(num_paths=PATHS, seed=7).price_detailed(
        p, _grid_env()
    )
    assert r.pv == r.pv_fixed_coupons + r.pv_ko_coupons + r.pv_loss_leg
