"""Per-step PDE operator coefficients from term structures."""
import numpy as np
import pytest

from term_structure_benchmarks import make_term_env

from quantark.asset.equity.engine.pde.european_pde_solver import EuropeanPDESolver


def _grids():
    # simple uniform grids: 5 time steps to 1y, 11 space nodes
    t_vec = np.linspace(0.0, 1.0, 6)
    dx_vec = np.full(10, 0.02)
    return t_vec, dx_vec, 11


def test_flat_env_single_unique_set_matches_scalar_path():
    solver = EuropeanPDESolver()
    env = make_term_env("flat")
    t_vec, dx_vec, num_x = _grids()
    sc = solver._build_step_coefficients(env, 100.0, t_vec, dx_vec, num_x)
    assert sc.n_unique == 1
    assert np.all(sc.set_index == 0)
    sc = solver._flat_exact_step_coefficients(sc, 0.03, 0.01, 0.20, dx_vec, num_x)
    l, c, u = solver._calculate_coefficients(0.03, 0.01, 0.20, dx_vec, num_x)
    l2, c2, u2 = sc.lcu_sets[0]
    assert np.array_equal(l, l2) and np.array_equal(c, c2) and np.array_equal(u, u2)


def test_term_env_many_unique_sets_and_correct_per_step_values():
    solver = EuropeanPDESolver()
    env = make_term_env("kinked")
    t_vec, dx_vec, num_x = _grids()
    sc = solver._build_step_coefficients(env, 100.0, t_vec, dx_vec, num_x)
    assert sc.n_unique > 1
    assert sc.set_index.shape == (5,)
    from quantark.priceenv import TermCoefficients

    tc = TermCoefficients.from_env(env, t_vec, ref_strike=100.0)
    for j in range(5):
        # the builder dedupes on 12-decimal-rounded triples (ulp absorption)
        l, c, u = solver._calculate_coefficients(
            round(float(tc.fwd_rates[j]), 12), round(float(tc.fwd_carry[j]), 12),
            round(float(tc.step_vols[j]), 12), dx_vec, num_x,
        )
        l2, c2, u2 = sc.lcu_sets[int(sc.set_index[j])]
        assert np.array_equal(l, l2) and np.array_equal(c, c2) and np.array_equal(u, u2)


def test_flat_exact_substitution_keeps_term_sets_untouched():
    solver = EuropeanPDESolver()
    env = make_term_env("kinked")
    t_vec, dx_vec, num_x = _grids()
    sc = solver._build_step_coefficients(env, 100.0, t_vec, dx_vec, num_x)
    sc2 = solver._flat_exact_step_coefficients(sc, 0.03, 0.01, 0.20, dx_vec, num_x)
    assert sc2 is sc  # n_unique > 1: no substitution
