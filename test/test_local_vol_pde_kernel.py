import numpy as np
import pytest
from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.util.numerical import solve_tridiag
from quantark.volmodels.localvol.pde_kernel import price_european_lv_pde
from quantark.volmodels.black_scholes import bs_call_price, bs_put_price


def _flat_lv(sigma=0.2):
    return LocalVolSurface(np.array([1.0, 1e6]), np.array([0.0, 100.0]), np.full((2, 2), sigma))


def _const_steps(T, M, r, q):
    dt = np.full(M, T / M)
    return dt, np.full(M, r), np.full(M, q)


def test_flat_lv_pde_matches_bs_call():
    sigma, s0, k, T, r, q = 0.2, 100.0, 100.0, 1.0, 0.03, 0.01
    dt, rf, cf = _const_steps(T, 200, r, q)
    price = price_european_lv_pde(s0, k, True, T, _flat_lv(sigma), dt, rf, cf, n_s=400)
    assert price == pytest.approx(bs_call_price(s0, k, T, sigma, r, q), abs=2e-2)


def test_flat_lv_pde_matches_bs_put():
    sigma, s0, k, T, r, q = 0.3, 100.0, 90.0, 0.75, 0.02, 0.0
    dt, rf, cf = _const_steps(T, 200, r, q)
    price = price_european_lv_pde(s0, k, False, T, _flat_lv(sigma), dt, rf, cf, n_s=400)
    assert price == pytest.approx(bs_put_price(s0, k, T, sigma, r, q), abs=2e-2)


def test_pde_refinement_converges():
    sigma, s0, k, T, r, q = 0.2, 100.0, 100.0, 1.0, 0.03, 0.01
    bs = bs_call_price(s0, k, T, sigma, r, q)
    errs = []
    for ns, nt in [(100, 50), (200, 100), (400, 200), (800, 400)]:
        dt, rf, cf = _const_steps(T, nt, r, q)
        errs.append(abs(price_european_lv_pde(s0, k, True, T, _flat_lv(sigma), dt, rf, cf, n_s=ns) - bs))
    assert errs[-1] < errs[0] and errs[-1] < 5e-3


def test_itm_put_lower_boundary_and_itm_call_upper_boundary():
    # ITM put (high strike) and ITM call (low strike): exercises both boundaries.
    T, r, q = 1.0, 0.05, 0.0
    dt, rf, cf = _const_steps(T, 200, r, q)
    p = price_european_lv_pde(100.0, 150.0, False, T, _flat_lv(0.2), dt, rf, cf, n_s=400)
    assert p == pytest.approx(bs_put_price(100.0, 150.0, T, 0.2, r, q), abs=5e-2)
    c = price_european_lv_pde(100.0, 50.0, True, T, _flat_lv(0.2), dt, rf, cf, n_s=400)
    assert c == pytest.approx(bs_call_price(100.0, 50.0, T, 0.2, r, q), abs=5e-2)


def test_thomas_solver_matches_numpy_including_one_interior_node():
    rng = np.random.default_rng(0)
    for m in (1, 2, 5, 20):
        diag = rng.uniform(3.0, 5.0, m)
        sub = rng.uniform(-1.0, 1.0, max(m - 1, 0))
        sup = rng.uniform(-1.0, 1.0, max(m - 1, 0))
        rhs = rng.standard_normal(m)
        A = np.diag(diag)
        for i in range(m - 1):
            A[i + 1, i] = sub[i]
            A[i, i + 1] = sup[i]
        x = solve_tridiag(sub, diag, sup, rhs)
        assert np.allclose(x, np.linalg.solve(A, rhs), atol=1e-10)


def test_per_step_rates_nonflat_match_constant_equivalent():
    # A piecewise-constant rate path with the SAME integrated rate as a flat rate
    # need NOT match for local vol — but here vol is flat, so only integrals matter:
    # this verifies the per-step plumbing reproduces the flat-rate result.
    sigma, s0, k, T = 0.2, 100.0, 100.0, 1.0
    M = 200
    dt = np.full(M, T / M)
    # two-segment rate: 0.02 first half, 0.04 second half; flat-equivalent avg = 0.03
    rf = np.where(np.arange(M) < M // 2, 0.02, 0.04)
    cf = np.zeros(M)
    price = price_european_lv_pde(s0, k, True, T, _flat_lv(sigma), dt, rf, cf, n_s=400)
    # integrated rate to T = 0.03; BS with flat r=0.03 q=0 is the reference
    assert price == pytest.approx(bs_call_price(s0, k, T, sigma, 0.03, 0.0), abs=3e-2)


def test_rejects_invalid_inputs():
    lv = _flat_lv(0.2)
    with pytest.raises(ValidationError):
        price_european_lv_pde(100.0, 100.0, True, 1.0, lv, np.full(3, 1 / 3), np.zeros(3),
                              np.zeros(3), n_s=2)  # n_s too small
    with pytest.raises(ValidationError):
        price_european_lv_pde(-1.0, 100.0, True, 1.0, lv, np.full(3, 1 / 3), np.zeros(3),
                              np.zeros(3), n_s=100)


def _smile_lv():
    # Non-flat local vol in BOTH strike and time so rate-step ORDER matters.
    K = np.array([50.0, 80.0, 100.0, 125.0, 160.0])
    T = np.array([0.25, 0.5, 1.0])
    base = np.array([0.30, 0.24, 0.20, 0.24, 0.30])
    grid = np.vstack([base + 0.04, base + 0.02, base])  # vol decreases with maturity
    return LocalVolSurface(K, T, grid)


def test_per_step_rate_order_matters_under_nonflat_surface():
    # Same integrated rate, reversed step order: under a state-dependent local-vol
    # surface the prices must differ (proves per-step alignment is real, not collapsed).
    s0, k, T, M = 100.0, 100.0, 1.0, 100
    dt = np.full(M, T / M)
    rising = np.where(np.arange(M) < M // 2, 0.01, 0.06)
    falling = rising[::-1].copy()
    cf = np.zeros(M)
    lv = _smile_lv()
    p_rise = price_european_lv_pde(s0, k, True, T, lv, dt, rising, cf, n_s=400)
    p_fall = price_european_lv_pde(s0, k, True, T, lv, dt, falling, cf, n_s=400)
    assert abs(p_rise - p_fall) > 1e-3


def test_rejects_step_sum_not_equal_T():
    lv = _flat_lv(0.2)
    with pytest.raises(ValidationError):
        price_european_lv_pde(100.0, 100.0, True, 1.0, lv, np.full(50, 0.01),
                              np.zeros(50), np.zeros(50), n_s=200)  # sum=0.5 != T=1.0


def test_price_regression_pinned_for_solver_swap():
    # Captured from the pre-solve_banded implementation (sequential Thomas), WS-B2.
    # Gate: <= 1e-12 relative (LAPACK banded solve differs only in arithmetic order).
    surf = LocalVolSurface(strike_grid=np.array([50.0, 100.0, 200.0]),
                           time_grid=np.array([0.0, 1.0]),
                           lv_grid=np.array([[0.30, 0.22, 0.20], [0.32, 0.24, 0.21]]))
    dt = np.full(50, 0.02)
    price = price_european_lv_pde(100.0, 105.0, True, 1.0, surf, dt,
                                  np.full(50, 0.03), np.full(50, 0.01), n_s=200)
    assert np.isclose(price, 7.931287902952514, rtol=1e-12)
