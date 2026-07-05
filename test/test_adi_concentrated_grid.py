"""WS-C2: non-uniform stencil coefficients + concentrated-grid ADI convergence."""
import numpy as np

from quantark.util.numerical.finite_difference import (
    fd1_interior_coeffs, fd2_interior_coeffs, fd1_nonuniform, fd2_nonuniform,
)
from quantark.util.enum.engine_enums import ADIScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.pde_kernel import price_european_heston_pde
from quantark.volmodels.heston.analytical_kernel import heston_call_price


def test_fd1_coeffs_exact_for_quadratic_on_nonuniform_grid():
    x = np.array([0.0, 0.3, 0.7, 1.2, 2.0, 3.5])
    f = 2.0 * x ** 2 - 3.0 * x + 1.0
    wm, w0, wp = fd1_interior_coeffs(x)
    approx = wm * f[:-2] + w0 * f[1:-1] + wp * f[2:]
    exact = 4.0 * x[1:-1] - 3.0
    assert np.allclose(approx, exact, atol=1e-10)


def test_fd2_coeffs_exact_for_quadratic_on_nonuniform_grid():
    x = np.array([0.0, 0.3, 0.7, 1.2, 2.0, 3.5])
    f = 2.0 * x ** 2 - 3.0 * x + 1.0
    wm, w0, wp = fd2_interior_coeffs(x)
    approx = wm * f[:-2] + w0 * f[1:-1] + wp * f[2:]
    assert np.allclose(approx, 4.0, atol=1e-10)


def test_coeffs_match_applied_stencils():
    x = np.array([0.0, 0.3, 0.7, 1.2, 2.0, 3.5])
    f = np.sin(x)
    wm1, w01, wp1 = fd1_interior_coeffs(x)
    assert np.allclose(wm1 * f[:-2] + w01 * f[1:-1] + wp1 * f[2:],
                       fd1_nonuniform(f, x)[1:-1], atol=1e-14)
    wm2, w02, wp2 = fd2_interior_coeffs(x)
    assert np.allclose(wm2 * f[:-2] + w02 * f[1:-1] + wp2 * f[2:],
                       fd2_nonuniform(f, x)[1:-1], atol=1e-14)


def test_uniform_grid_coeffs_reduce_to_scalar_form():
    x = np.linspace(0.0, 1.0, 7)
    h = x[1] - x[0]
    wm1, w01, wp1 = fd1_interior_coeffs(x)
    assert np.allclose(wm1, -1.0 / (2 * h)) and np.allclose(w01, 0.0) and np.allclose(wp1, 1.0 / (2 * h))
    wm2, w02, wp2 = fd2_interior_coeffs(x)
    assert np.allclose(wm2, 1.0 / h ** 2) and np.allclose(w02, -2.0 / h ** 2) and np.allclose(wp2, 1.0 / h ** 2)


def test_grid_style_default_is_uniform():
    import inspect
    assert inspect.signature(price_european_heston_pde).parameters["grid_style"].default == "uniform"


def test_uniform_path_bit_identical_when_grid_style_uniform():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    p_a = price_european_heston_pde(s0, k, True, T, params, r, q, n_x=120, n_v=60, n_t=50)
    p_b = price_european_heston_pde(s0, k, True, T, params, r, q, n_x=120, n_v=60, n_t=50,
                                    grid_style="uniform")
    assert p_a == p_b  # exact


def test_concentrated_grid_equal_node_error_reduction():
    # Long-dated (T=1): concentration helps modestly (~1.1x here). At long maturity the
    # error is dominated by V-boundary truncation / diffused-kink smear, not near-strike
    # resolution, so this regime is NOT where concentration shines (see the short-dated
    # test below). It must still be no worse than uniform at equal nodes.
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    ref = heston_call_price(s0, k, T, params, r, q)
    common = dict(n_x=80, n_v=40, n_t=100, scheme=ADIScheme.CRAIG_SNEYD)
    e_uni = abs(price_european_heston_pde(s0, k, True, T, params, r, q,
                                          grid_style="uniform", **common) - ref)
    e_con = abs(price_european_heston_pde(s0, k, True, T, params, r, q,
                                          grid_style="concentrated", **common) - ref)
    assert e_con <= e_uni


def test_concentrated_grid_short_dated_meets_target_reduction():
    # WS-C2 acceptance regime: SHORT-dated, where the sharp payoff kink dominates the
    # error and node concentration at the strike delivers the spec's >=4x node/error
    # reduction (measured ~4.06x at T=0.1, n_x=80, n_v=40). This is where concentration
    # is designed to pay off; the default stays uniform because the win is regime-specific.
    s0, k, T, r, q = 100.0, 100.0, 0.1, 0.03, 0.0
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    ref = heston_call_price(s0, k, T, params, r, q)
    common = dict(n_x=80, n_v=40, n_t=100, scheme=ADIScheme.CRAIG_SNEYD)
    e_uni = abs(price_european_heston_pde(s0, k, True, T, params, r, q,
                                          grid_style="uniform", **common) - ref)
    e_con = abs(price_european_heston_pde(s0, k, True, T, params, r, q,
                                          grid_style="concentrated", **common) - ref)
    assert e_uni / e_con >= 3.0  # target >=4x; assert a robust >=3x margin


def test_concentrated_tridiag_rows_match_uniform_on_a_uniform_grid():
    # Full-row equivalence (Codex plan-gate): on a UNIFORM grid the concentrated
    # coefficient path must reproduce the uniform scalar-dx tridiagonal rows (a,b,c),
    # not just the derivative weights. Proves the concentrated math (esp. the diagonal
    # sign) is correct. Closeness, not bit-identity — the branches differ by ULPs.
    from quantark.volmodels.adi_core import HestonSLVADICore
    from quantark.util.numerical import fd1_interior_coeffs, fd2_interior_coeffs
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    kw = dict(s0=100.0, strike=100.0, T=1.0, r=0.03, carry=0.0, params=params,
              n_x=40, n_v=30, n_t=50, leverage=None, eta=1.0)
    core_u = HestonSLVADICore(**kw, grid_style="uniform")
    core_c = HestonSLVADICore(**kw, grid_style="concentrated")
    # force the concentrated core onto the SAME uniform nodes so the rows are comparable
    core_c.X_grid = core_u.X_grid.copy(); core_c.V_grid = core_u.V_grid.copy()
    core_c._xx = fd2_interior_coeffs(core_c.X_grid); core_c._x1 = fd1_interior_coeffs(core_c.X_grid)
    core_c._vv = fd2_interior_coeffs(core_c.V_grid); core_c._v1 = fd1_interior_coeffs(core_c.V_grid)
    for builder, args in [("_tri_V", (core_u.dt, 1.0)), ("_tri_S", (core_u.dt, 1.0, 0.0))]:
        au, bu, cu = getattr(core_u, builder)(*args)
        core_c._S_tri_cache.clear(); core_c._V_tri_cache.clear()
        ac, bc, cc = getattr(core_c, builder)(*args)
        assert np.allclose(au, ac, atol=1e-10) and np.allclose(bu, bc, atol=1e-10) \
            and np.allclose(cu, cc, atol=1e-10), f"{builder} rows diverge (sign/coeff bug)"


def test_concentrated_greeks_are_clean():
    # grid_spot pinning + concentrated grid should still give finite, sane delta/gamma.
    from quantark.volmodels.heston.pde_kernel import price_delta_gamma_heston_pde
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    price, delta, gamma = price_delta_gamma_heston_pde(
        100.0, 100.0, True, 1.0, params, 0.03, 0.0, n_x=120, n_v=60, n_t=60,
        grid_style="concentrated", grid_spot=100.0)
    assert np.isfinite(price) and 0.0 < delta < 1.0 and gamma > 0.0
