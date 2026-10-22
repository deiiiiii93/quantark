"""QUAD engine term-parameter builder on observation grids."""
import numpy as np
import pytest

from term_structure_benchmarks import make_term_env

from quantark.asset.equity.engine.quad.term_inputs import build_quad_term_params
from quantark.priceenv.term_sampling import make_df_fn


def test_flat_env_constant_arrays():
    env = make_term_env("flat")
    tp = build_quad_term_params(env, 100.0, [0.25, 0.5, 0.75, 1.0])
    assert tp.rate == pytest.approx(np.full(4, 0.03), abs=1e-12)
    assert tp.div == pytest.approx(np.full(4, 0.01), abs=1e-12)
    assert tp.vol == pytest.approx(np.full(4, 0.20), abs=1e-12)
    assert tp.node_dfs.shape == (5,)


def test_term_env_reproduces_cumulative_quantities():
    env = make_term_env("kinked")
    obs = [0.25, 0.5, 0.75, 1.0]
    tp = build_quad_term_params(env, 100.0, obs)
    dt = np.diff(np.concatenate(([0.0], obs)))
    T = 1.0
    assert float(np.sum(tp.rate * dt)) == pytest.approx(env.get_rate(T) * T, rel=1e-10)
    assert float(np.sum(tp.div * dt)) == pytest.approx(
        env.get_div_yield(T) * T, rel=1e-10
    )
    assert float(np.sum(tp.vol**2 * dt)) == pytest.approx(
        env.get_vol(100.0, T) ** 2 * T, rel=1e-10
    )


def test_make_df_fn_scalar_and_array():
    env = make_term_env("up")
    df = make_df_fn(env)
    assert df(1.0) == pytest.approx(env.get_discount_factor(1.0), rel=1e-14)
    out = df(np.array([0.5, 1.0]))
    assert out.shape == (2,)
    assert out[1] == pytest.approx(env.get_discount_factor(1.0), rel=1e-14)
