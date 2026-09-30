"""Factorization-cache keys: θ exact, dt rounded as today (spec 2 decision 6, §5.7)."""
import numpy as np

from quantark.asset.equity.engine.pde.core.backends import FactorizationCache
from quantark.asset.equity.engine.pde.core.groups import Backend


def _lcu(n=8):
    l = np.full(n, 1.0); c = np.full(n, -2.5); u = np.full(n, 1.2)
    return l, c, u


def _lu(fc, l, c, u, dt, theta):
    from quantark.asset.equity.engine.pde.core.backends import operator_matrix, sparse_lu_factor
    A = operator_matrix(l, c, u, len(l))
    return fc.get(Backend.SPARSE_LU, 0, dt, theta, lambda: sparse_lu_factor(A, dt, theta))


def _banded(fc, l, c, u, dt, theta):
    from quantark.asset.equity.engine.pde.core.backends import banded_factor
    return fc.get(Backend.BANDED, 0, dt, theta, lambda: banded_factor(l, c, u, dt, theta))


def test_two_thetas_agreeing_to_six_decimals_get_two_lu_factorizations():
    fc = FactorizationCache()
    l, c, u = _lcu()
    _lu(fc, l, c, u, 0.01, 0.5)
    _lu(fc, l, c, u, 0.01, 0.5000001)
    assert len(fc.store(Backend.SPARSE_LU)) == 2


def test_two_thetas_agreeing_to_twelve_decimals_get_two_banded_systems():
    fc = FactorizationCache()
    l, c, u = _lcu()
    _banded(fc, l, c, u, 0.01, 0.5)
    _banded(fc, l, c, u, 0.01, 0.5 + 1e-13)
    assert len(fc.store(Backend.BANDED)) == 2


def test_dt_values_within_an_ulp_share_one_factorization():
    fc = FactorizationCache()
    l, c, u = _lcu()
    dt = np.float64(1.0 / 12.0)
    _banded(fc, l, c, u, dt, 0.5)
    _banded(fc, l, c, u, np.nextafter(dt, 1.0), 0.5)
    assert len(fc.store(Backend.BANDED)) == 1


# --- the core's FactorizationCache (spec 2 §5.7) ------------------------------


def test_cache_keys_theta_exact_and_dt_rounded():
    fc = FactorizationCache()
    calls = []
    build = lambda: calls.append(1) or object()  # noqa: E731
    fc.get(Backend.BANDED, 0, np.float64(1 / 12), 0.5, build)
    fc.get(Backend.BANDED, 0, np.nextafter(np.float64(1 / 12), 1.0), 0.5, build)
    fc.get(Backend.BANDED, 0, np.float64(1 / 12), 0.5000001, build)
    fc.get(Backend.SPARSE_LU, 0, np.float64(1 / 12), 0.5, build)
    assert len(calls) == 3


def test_banded_store_is_lru_bounded_and_lu_store_unbounded():
    fc = FactorizationCache(banded_max_entries=2)
    for k in range(3):
        fc.get(Backend.BANDED, k, 0.1, 0.5, object)
        fc.get(Backend.SPARSE_LU, k, 0.1, 0.5, object)
    assert list(fc.store(Backend.BANDED)) == [fc.key(1, 0.1, 0.5), fc.key(2, 0.1, 0.5)]
    assert len(fc.store(Backend.SPARSE_LU)) == 3


def test_seed_is_consulted_first_even_when_disabled():
    sentinel = object()
    fc = FactorizationCache(enabled=False, lu_seed={FactorizationCache.key(0, 0.1, 0.5): sentinel})
    assert fc.get(Backend.SPARSE_LU, 0, 0.1, 0.5, object) is sentinel
    assert fc.get(Backend.SPARSE_LU, 1, 0.1, 0.5, object) is not fc.get(Backend.SPARSE_LU, 1, 0.1, 0.5, object)


def test_monthly_snowball_five_dt_floats_resolve_to_one_key():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.param import PDEParams
    from quantark.modelvalidation.builders.equity_snowball import make_snowball
    from term_structure_benchmarks import make_term_env
    sb = make_snowball(dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0, ko_rate=0.15,
                            rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0))
    plan = SnowballPDESolver(PDEParams()).explain(sb, make_term_env("flat"))
    dts = {float(d) for d in plan.to_dict()["time"]["dt"]}
    keys = {FactorizationCache.key(0, np.float64(d), 0.5)[1] for d in dts}
    assert len(dts) >= 2 and len(keys) < len(dts)   # measured §14: 5 floats -> 1 key
