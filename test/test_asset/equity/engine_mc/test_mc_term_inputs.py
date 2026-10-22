"""Engine-facing term-input builder on the generator time grid."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.term_inputs import (
    build_mc_term_inputs,
    df_at,
)
from term_structure_benchmarks import make_term_env


def test_flat_env_constant_arrays():
    env = make_term_env("flat")
    ti = build_mc_term_inputs(env, ref_strike=100.0, maturity=1.0, time_steps=12)
    assert ti.rrf == pytest.approx(np.full(12, 0.03), abs=1e-12)
    assert ti.div == pytest.approx(np.full(12, 0.01), abs=1e-12)
    assert ti.vol == pytest.approx(np.full(12, 0.20), abs=1e-12)
    assert ti.times.shape == (13,)
    assert ti.node_dfs[0] == 1.0


def test_term_env_forward_consistency():
    """Compounded step quantities must reproduce cumulative-to-T values."""
    env = make_term_env("kinked")
    T = 2.0
    ti = build_mc_term_inputs(env, ref_strike=100.0, maturity=T, time_steps=24)
    dt = np.diff(ti.times)
    assert float(np.sum(ti.rrf * dt)) == pytest.approx(env.get_rate(T) * T, rel=1e-10)
    assert float(np.sum(ti.div * dt)) == pytest.approx(
        env.get_div_yield(T) * T, rel=1e-10
    )
    assert float(np.sum(ti.vol**2 * dt)) == pytest.approx(
        env.get_vol(100.0, T) ** 2 * T, rel=1e-10
    )
    assert ti.node_dfs[-1] == pytest.approx(env.get_discount_factor(T), rel=1e-12)


def test_df_at_matches_nodes_and_rejects_off_grid():
    env = make_term_env("up")
    ti = build_mc_term_inputs(env, ref_strike=100.0, maturity=1.0, time_steps=4)
    assert df_at(ti, ti.times[2]) == pytest.approx(ti.node_dfs[2])
    with pytest.raises(ValueError):
        df_at(ti, 0.123456789)  # not a grid node
