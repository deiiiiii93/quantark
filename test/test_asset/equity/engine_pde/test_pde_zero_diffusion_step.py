"""Zero-diffusion steps: upwind operator + theta=1 monotonicity (spec §4.5)."""
import numpy as np

from quantark.asset.equity.engine.pde.base_pde_solver import StepCoefficients


def test_step_coefficients_zero_diffusion_default_empty():
    sc = StepCoefficients(
        lcu_sets=[(np.zeros(3),) * 3],
        set_index=np.zeros(1, dtype=int),
        n_unique=1,
    )
    assert sc.zero_diffusion_sets == frozenset()


def test_build_marks_zero_diffusion_sets():
    from helpers_pde_zero_diffusion import build_coeffs_with_zero_step
    sc, _dx, _r, _q = build_coeffs_with_zero_step()
    assert len(sc.zero_diffusion_sets) >= 1
    assert all(0 <= k < sc.n_unique for k in sc.zero_diffusion_sets)


def test_upwind_coefficients_signs_positive_drift():
    """The zero-sigma set is upwind: l == 0, u == mu/dx >= 0, c <= 0."""
    from helpers_pde_zero_diffusion import build_coeffs_with_zero_step
    sc, dx, r, q = build_coeffs_with_zero_step(mu_sign=+1)
    k = next(iter(sc.zero_diffusion_sets))
    l, c, u = sc.lcu_sets[k]
    mu = r - q
    assert np.all(l[1:-1] == 0.0)
    assert np.allclose(u[1:-1], mu / dx)
    assert np.all(c[1:-1] <= 0.0)


def test_upwind_coefficients_signs_negative_drift():
    from helpers_pde_zero_diffusion import build_coeffs_with_zero_step
    sc, dx, r, q = build_coeffs_with_zero_step(mu_sign=-1)
    k = next(iter(sc.zero_diffusion_sets))
    l, c, u = sc.lcu_sets[k]
    mu = r - q
    assert np.all(u[1:-1] == 0.0)
    assert np.allclose(l[1:-1], -mu / dx)
    assert np.all(c[1:-1] <= 0.0)


def test_monotone_no_new_extrema_on_kinked_profile():
    """A pure-advection theta=1 upwind step of a kinked payoff creates no new
    extrema (M-matrix property)."""
    from helpers_pde_zero_diffusion import advance_zero_diffusion_step
    x = np.linspace(-0.5, 0.5, 401)
    v0 = np.maximum(0.0, 1.0 - np.abs(x) * 4.0)          # kink at 0 and +-0.25
    v1 = advance_zero_diffusion_step(v0, x, r=0.02, q=0.01, dt=9.0 / 365.0)
    assert float(np.min(v1)) >= -1e-15
    assert float(np.max(v1)) <= float(np.max(v0)) + 1e-15
