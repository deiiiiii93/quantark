import numpy as np
import pytest

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.volmodels.black_scholes import bs_call_price
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig
from quantark.volmodels.slv.slv_pde_kernel import price_european_slv_pde

_ZERO_DIV = (lambda t: 0.0)


def _surface(skew=False):
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    if skew:
        ks = np.linspace(-0.6, 0.6, 9)
        grid = np.array([[0.22 - 0.04 * ks[j] for j in range(9)] for _ in mats])
    else:
        grid = np.full((6, 9), 0.20)
    return GridVolSurface(strikes, mats, grid)


def _calibrate(surf, params, n=40, fp_config=None):
    lv = build_dupire_local_vol(surf, spot=100.0, rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    t_grid = np.linspace(0.0, 1.0, n + 1)
    r_fwd = np.full(n, 0.02)
    carry_fwd = np.full(n, 0.0)
    cfg = fp_config or FpCalibrationConfig(n_x=201, n_z=101)
    return calibrate_leverage_surface(100.0, params, lv, np.diff(t_grid), r_fwd, carry_fwd,
                                      eta=1.0, fp_config=cfg)


def test_ffp_surface_reprices_flat_vanilla_via_backward_pde():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    lev = _calibrate(_surface(skew=False), p)
    price = price_european_slv_pde(100.0, 100.0, True, 1.0, p, lev, 0.02, 0.0,
                                   eta=1.0, n_x=180, n_v=70, n_t=70)
    bs = bs_call_price(100.0, 100.0, 1.0, 0.20, 0.02, 0.0)
    assert abs(price - bs) < 0.25            # exact-vanilla property within discretization tol


def test_ffp_surface_reprices_skewed_vanilla_via_backward_pde():
    # leverage-nontrivial case: skewed local vol => spatially-varying leverage L(x)
    # (the Heston L=1 oracle cannot cover this). sigma matches the flat case (Feller-satisfied).
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    surf = _surface(skew=True)
    lev = _calibrate(surf, p)
    for K, iv in ((100.0, 0.22), (100.0 * np.exp(0.3), 0.22 - 0.04 * 0.5)):
        price = price_european_slv_pde(100.0, K, True, 1.0, p, lev, 0.02, 0.0,
                                       eta=1.0, n_x=180, n_v=70, n_t=70)
        bs = bs_call_price(100.0, K, 1.0, iv, 0.02, 0.0)
        assert abs(price - bs) < 0.4


def test_ffp_determinism():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    a = _calibrate(_surface(), p, n=20, fp_config=FpCalibrationConfig(n_x=121, n_z=61))
    b = _calibrate(_surface(), p, n=20, fp_config=FpCalibrationConfig(n_x=121, n_z=61))
    assert np.array_equal(a.leverage_grid, b.leverage_grid)   # no RNG -> bit-identical


def test_ffp_mass_conserved_under_feller_violation():
    # 2*kappa*theta < (eta*sigma)^2 : the log-variance + zero-flux scheme must still complete
    p = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.6, rho=-0.5)
    assert 2 * p.kappa * p.theta < p.sigma ** 2          # Feller violated
    lev = _calibrate(_surface(), p, n=30, fp_config=FpCalibrationConfig(n_x=161, n_z=121))
    assert np.all(np.isfinite(lev.leverage_grid)) and np.all(lev.leverage_grid > 0)
    assert max(lev.diagnostics["mass_residual"]) < 1e-3   # mass conserved every slice


def test_ffp_stable_under_time_refinement_and_high_correlation():
    # backward-Euler march must NOT blow up with more steps or stronger rho (the ADI variant did:
    # negative mass grew to 1e7+). Negative mass stays tiny and bounded across refinement.
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    cfg = FpCalibrationConfig(n_x=161, n_z=101)
    prev = None
    for n in (40, 100):
        lev = _calibrate(_surface(), p, n=n, fp_config=cfg)
        neg = lev.diagnostics["max_negative_mass"]
        assert neg < 1e-3                                  # tiny, not the 1e7 of the unstable ADI
        if prev is not None:
            assert neg < 10 * prev                         # bounded under refinement (no blow-up)
        prev = neg


def _max_neg(sigma=0.7, rho=-0.9, n=60):
    p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=sigma, rho=rho)
    cfg = FpCalibrationConfig(n_x=161, n_z=121, tol_neg=0.9)
    lev = _calibrate(_surface(skew=True), p, n=n, fp_config=cfg)
    return lev.diagnostics["max_negative_mass"]


def test_directional_operators_are_already_m_matrices():
    # Investigation finding (2026-07): the FP negativity is NOT a directional-convection artifact.
    # The x/z directional operators already have zero negative off-diagonals (they are M-matrices), so
    # exponential-fitting / Chang-Cooper on them cannot help. All negativity is the anisotropic mixed
    # (correlation) term, which is the sole M-matrix violator (grid-aligned positivity is infeasible
    # there: b/a ~ up to 4900x, |c| > min(a,b) almost everywhere). This regression pins the diagnosis.
    from quantark.volmodels.slv.fokkerplanck.coordinates import concentrated_grid
    from quantark.volmodels.slv.fokkerplanck.fp_operators import build_directional_operators
    p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.7, rho=-0.9)
    x = concentrated_grid(np.log(60.0), np.log(160.0), np.log(100.0), 31, 0.1)
    z = concentrated_grid(np.log(0.01), np.log(0.30), np.log(p.v0), 25, 0.1)
    L = np.linspace(0.8, 1.3, x.size)
    Ax, Az, Axz = build_directional_operators(x, z, L, p, 1.0, 0.05)
    for M in (Ax, Az):
        Md = M.toarray().copy()
        np.fill_diagonal(Md, 0.0)
        assert Md.min() >= -1e-12                # directional op is an M-matrix (no negative off-diagonal)
    Axzd = Axz.toarray().copy()
    np.fill_diagonal(Axzd, 0.0)
    assert Axzd.min() < -1.0                     # the mixed operator is the sole M-matrix violator


def test_negativity_is_mixed_term_only():
    # rho=0 turns the mixed term off -> the density stays non-negative to machine precision: direct
    # confirmation that the correlation term is the entire negativity budget.
    assert _max_neg(sigma=0.7, rho=0.0) < 1e-12


def test_negativity_within_operating_bound_justifies_tol_neg():
    # tol_neg is tightened 0.5 -> 0.05 as an empirical operating-domain tripwire (observed bound with
    # clamped read-off, NOT a positivity claim). Even on the worst realistic fixture (sigma=0.7,
    # |rho|=0.9) the clamped negative mass stays well under 0.03, so 0.05 keeps ~2.7x margin while still
    # catching genuine divergence (mass ~ unit total).
    assert _max_neg(sigma=0.7, rho=-0.9) < 0.03


def test_bilinear_seed_still_reprices_flat_vanilla():
    # the opt-in seed_split must not perturb the exact-vanilla property (seed accuracy only)
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    cfg = FpCalibrationConfig(n_x=201, n_z=101, seed_split=True)
    lev = _calibrate(_surface(skew=False), p, fp_config=cfg)
    price = price_european_slv_pde(100.0, 100.0, True, 1.0, p, lev, 0.02, 0.0,
                                   eta=1.0, n_x=180, n_v=70, n_t=70)
    bs = bs_call_price(100.0, 100.0, 1.0, 0.20, 0.02, 0.0)
    assert abs(price - bs) < 0.25


def test_ffp_agrees_with_mc_binning_core():
    from quantark.util.enum.engine_enums import LeverageCalibrationMethod
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    surf = _surface()
    lv = build_dupire_local_vol(surf, spot=100.0, rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    t_grid = np.linspace(0.0, 1.0, 31)
    args = (100.0, p, lv, np.diff(t_grid), np.full(30, 0.02), np.full(30, 0.0))
    fp = calibrate_leverage_surface(*args, eta=1.0, fp_config=FpCalibrationConfig(n_x=161, n_z=101))
    mc = calibrate_leverage_surface(*args, eta=1.0, method=LeverageCalibrationMethod.MC_BINNING,
                                    num_paths=60_000, num_bins=20, seed=5)
    # compare leverage at a mid-maturity, ATM region where both methods are reliable
    t = 0.5
    for S in (90.0, 100.0, 110.0):
        assert fp.leverage(S, t) == pytest.approx(mc.leverage(S, t), rel=0.15)
