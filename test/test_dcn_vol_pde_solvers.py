"""Surface-aware DCN PDE engines: LV (1-D) and Heston (2-D) cross-checks."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    HestonDCNMCEngine,
    LocalVolDCNMCEngine,
)
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import (
    HestonDCNPDESolver,
    LocalVolDCNPDEEngine,
)
from quantark.param import GridVolSurface
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn

PATHS = 2 ** 14
FELLER_VIOLATING = HestonParams(
    v0=0.0249, kappa=9.92, theta=0.045, sigma=1.583, rho=-0.025,
)


def _grid_env(skew: float = 0.0):
    env = flat_env(**FLAT)
    strikes = np.array([3000.0, 4500.0, 6000.0, 7500.0, 9000.0])
    maturities = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    base = FLAT["sigma"] + skew * (6000.0 - strikes) / 6000.0
    iv = np.tile(base, (len(maturities), 1))
    env.vol_surface = GridVolSurface(
        strikes=strikes.tolist(), maturities=maturities, iv_grid=iv
    )
    return env


def test_lv_pde_flat_surface_matches_flat_pde():
    # Flat grid surface -> Dupire local vol == flat vol -> the LV PDE must
    # reproduce the flat two-surface PDE almost exactly (same solver path).
    p = make_dcn(DCN_A)
    env = _grid_env(skew=0.0)
    flat = DCNPDEEngine(num_space_nodes=601).price_detailed(p, env)
    lv = LocalVolDCNPDEEngine(num_space_nodes=601).price_detailed(p, env)
    assert abs(lv.pv - flat.pv) < 5e-4 * p.notional


def test_lv_pde_matches_lv_mc_on_skewed_surface():
    p = make_dcn(DCN_A)
    env = _grid_env(skew=0.05)
    mc = LocalVolDCNMCEngine(
        num_paths=PATHS, seed=42, use_sobol=True, num_batches=8
    ).price_detailed(p, env)
    pde = LocalVolDCNPDEEngine(num_space_nodes=1201).price_detailed(p, env)
    tol = max(4.0 * mc.std_error, 5e-4 * p.notional)
    assert abs(pde.pv - mc.pv) < tol


def test_lv_pde_rejects_bad_surface_argument():
    with pytest.raises(ValidationError):
        LocalVolDCNPDEEngine(local_vol_surface="not a surface")


def test_heston_pde_validations():
    with pytest.raises(ValidationError):
        HestonDCNPDESolver(model_params="nope")
    with pytest.raises(ValidationError):
        HestonDCNPDESolver(FELLER_VIOLATING, scheme="MCS")
    with pytest.raises(ValidationError):
        HestonDCNPDESolver(FELLER_VIOLATING, n_x=10)
    with pytest.raises(ValidationError):
        HestonDCNPDESolver(FELLER_VIOLATING, substeps_per_interval=0)
    with pytest.raises(ValidationError):
        HestonDCNPDESolver(FELLER_VIOLATING, v0_boundary="dirichlet")


def test_heston_pde_degenerate_params_match_flat_pde():
    # vol-of-vol -> 0 with v0 = theta = sigma_flat^2 collapses Heston to the
    # flat lognormal model, so the 2-D solver must agree with the validated
    # 1-D two-surface PDE within 2-D grid tolerance.
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    params = HestonParams(
        v0=FLAT["sigma"] ** 2, kappa=1.0, theta=FLAT["sigma"] ** 2,
        sigma=1e-6, rho=0.0,
    )
    flat = DCNPDEEngine(num_space_nodes=2401).price_detailed(p, env)
    heston = HestonDCNPDESolver(
        params, n_x=501, n_v=201
    ).price_detailed(p, env)
    assert abs(heston.pv - flat.pv) < 2e-3 * p.notional


def test_heston_pde_matches_heston_mc_feller_violating():
    # The real gate pattern: deterministic 2-D PDE vs QE-M MC under the
    # market-like Feller-violating calibration. The power-graded v-grid is
    # what makes this pass: an ungraded grid misprices by >10x this
    # tolerance in the 2*kappa*theta << sigma^2 regime.
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    mc = HestonDCNMCEngine(
        model_params=FELLER_VIOLATING, scheme=HestonMCScheme.QUADEXP_M,
        substeps_per_interval=2, num_paths=PATHS, seed=42,
        use_sobol=True, num_batches=8,
    ).price_detailed(p, env)
    pde = HestonDCNPDESolver(
        FELLER_VIOLATING, n_x=501, n_v=201
    ).price_detailed(p, env)
    tol = max(4.0 * mc.std_error, 2e-3 * p.notional)
    assert abs(pde.pv - mc.pv) < tol


def test_heston_pde_deterministic_and_valuation_date_ki():
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    solver = HestonDCNPDESolver(FELLER_VIOLATING, n_x=151, n_v=41)
    a = solver.price(p, env)
    b = HestonDCNPDESolver(FELLER_VIOLATING, n_x=151, n_v=41).price(p, env)
    assert a == b
    # spot at/below the KI barrier must price off the knocked-in surface:
    # for the BUYER that removes all coupon value, so PV strictly drops.
    env_ki = flat_env(**FLAT, spot=0.99 * p.ki_barrier)
    ki_pv = HestonDCNPDESolver(
        FELLER_VIOLATING, n_x=151, n_v=41
    ).price(p, env_ki)
    env_near = flat_env(**FLAT, spot=1.01 * p.ki_barrier)
    near_pv = HestonDCNPDESolver(
        FELLER_VIOLATING, n_x=151, n_v=41
    ).price(p, env_near)
    assert ki_pv < near_pv


def test_adi_core_power_graded_v_grid_vanilla_accuracy():
    # Direct core check behind the DCN solver's default: at the market-like
    # Feller-violating parameters (2*kappa*theta/sigma^2 ~= 0.36) an
    # ungraded v-grid misprices a 2y vanilla by >10%; the power-graded grid
    # must land within 2% of the analytical Heston price.
    from quantark.util.enum.engine_enums import ADIScheme
    from quantark.volmodels.adi_core import HestonSLVADICore
    from quantark.volmodels.heston.analytical_kernel import (
        price_european_lewis,
    )

    s0, r, q, K, T = 6000.0, 0.0356, 0.1406, 6000.0, 2.0
    ana = price_european_lewis(s0, r, q, FELLER_VIOLATING, K, T)
    core = HestonSLVADICore(
        s0, K, T, r, q, FELLER_VIOLATING, 301, 201, 486,
        grid_style="concentrated", v0_boundary="degenerate_pde",
        v_grid_power=2.5,
    )
    U = core.solve(True, ADIScheme.CRAIG_SNEYD, 0.5, rannacher=True)
    px = core.interpolate(U, np.log(s0), FELLER_VIOLATING.v0)
    assert abs(px - ana) < 0.02 * ana
    # graded grid is strictly clustered at v = 0 and spans the same extent
    dv = np.diff(core.V_grid)
    assert dv[0] < dv[-1] / 10.0
    assert core.V_grid[0] == 0.0


def test_adi_core_v_grid_power_validations():
    from quantark.util.exceptions import ValidationError as VE
    from quantark.volmodels.adi_core import HestonSLVADICore

    common = dict(
        s0=6000.0, strike=6000.0, T=1.0, r=0.03, carry=0.1,
        params=FELLER_VIOLATING, n_x=61, n_v=31, n_t=10,
    )

    def build(**kw):
        args = dict(common)
        args.update(kw)
        return HestonSLVADICore(
            args["s0"], args["strike"], args["T"], args["r"], args["carry"],
            args["params"], args["n_x"], args["n_v"], args["n_t"],
            grid_style=args.get("grid_style", "concentrated"),
            v_grid_power=args.get("v_grid_power", 0.0),
        )

    build(v_grid_power=0.0)  # default path untouched
    build(v_grid_power=2.5)
    with pytest.raises(VE):
        build(v_grid_power=0.5)
    with pytest.raises(VE):
        build(v_grid_power=-1.0)
    with pytest.raises(VE):
        build(grid_style="uniform", v_grid_power=2.0)
