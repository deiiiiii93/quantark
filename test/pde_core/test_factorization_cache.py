"""Factorization-cache keys: θ exact, dt rounded as today (spec 2 decision 6, §5.7)."""
import numpy as np

from quantark.asset.equity.engine.pde import EuropeanPDESolver, SnowballPDESolver
from quantark.asset.equity.param import PDEParams


def _lcu(n=8):
    l = np.full(n, 1.0); c = np.full(n, -2.5); u = np.full(n, 1.2)
    return l, c, u


def test_two_thetas_agreeing_to_six_decimals_get_two_lu_factorizations():
    import scipy.sparse as sp
    s = EuropeanPDESolver(PDEParams())
    l, c, u = _lcu()
    A = s._build_operator_matrix(l, c, u, 8)
    I = sp.eye(6, format="csc")
    s._matrix_cache.clear()
    s._get_matrices(I, A, 0.01, 0.5, coeff_key=0)
    s._get_matrices(I, A, 0.01, 0.5000001, coeff_key=0)
    assert len(s._matrix_cache) == 2


def test_two_thetas_agreeing_to_twelve_decimals_get_two_banded_systems():
    s = SnowballPDESolver(PDEParams())
    l, c, u = _lcu()
    s._banded_cache.clear()
    s._get_banded_system(l, c, u, 0.01, 0.5, coeff_key=0)
    s._get_banded_system(l, c, u, 0.01, 0.5 + 1e-13, coeff_key=0)
    assert len(s._banded_cache) == 2


def test_dt_values_within_an_ulp_share_one_factorization():
    s = SnowballPDESolver(PDEParams())
    l, c, u = _lcu()
    dt = np.float64(1.0 / 12.0)
    s._banded_cache.clear()
    s._get_banded_system(l, c, u, dt, 0.5)
    s._get_banded_system(l, c, u, np.nextafter(dt, 1.0), 0.5)
    assert len(s._banded_cache) == 1
