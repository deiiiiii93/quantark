"""Per-step (term-structured) coefficients in GBMPathGenerator."""
import numpy as np
import pytest

from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.asset.equity.process.bsm.qmc_sobol import PseudoRandomNormalGenerator


def _gen(vol, rrf, div, **kw):
    return GBMPathGenerator(
        initial_value=100.0, vol=vol, rrf=rrf, div=div, maturity=1.0,
        time_steps=12, num_paths=64,
        random_stream=PseudoRandomNormalGenerator(seed=7), **kw,
    )


def test_constant_arrays_bit_identical_to_scalars():
    scal, _ = _gen(0.2, 0.03, 0.01).generate_paths()
    n = 12
    arr, _ = _gen(np.full(n, 0.2), np.full(n, 0.03), np.full(n, 0.01)).generate_paths()
    assert np.array_equal(scal, arr)


def test_terminal_qmc_constant_arrays_bit_identical():
    n = 12
    a = _gen(0.2, 0.03, 0.01).generate_terminal_values_qmc()
    b = _gen(
        np.full(n, 0.2), np.full(n, 0.03), np.full(n, 0.01)
    ).generate_terminal_values_qmc()
    assert np.array_equal(a, b)


def test_per_step_drift_reproduces_forward():
    """Vanishing vol isolates the deterministic drift: S_T must equal
    S0 * exp(sum((r_k - q_k) dt_k))."""
    n = 12
    rrf = np.linspace(0.01, 0.05, n)
    div = np.linspace(0.02, -0.01, n)
    g = _gen(1e-12, rrf, div)
    paths, _ = g.generate_paths()
    dt = g.dt_vector
    expected_T = 100.0 * np.exp(np.sum((rrf - div) * dt))
    assert paths[:, -1] == pytest.approx(expected_T, rel=1e-9)


def test_per_step_vol_scales_increments():
    """Two-step generator with vols [a, b]: log-increment stds scale as a, b."""
    g = GBMPathGenerator(
        initial_value=100.0, vol=np.array([0.1, 0.4]), rrf=0.0, div=0.0,
        maturity=1.0, time_steps=2, num_paths=200_000,
        random_stream=PseudoRandomNormalGenerator(seed=11),
    )
    paths, _ = g.generate_paths()
    logs = np.diff(np.log(paths), axis=1)
    stds = logs.std(axis=0, ddof=1) / np.sqrt(g.dt_vector)
    assert stds == pytest.approx([0.1, 0.4], rel=2e-2)


def test_array_length_mismatch_rejected():
    with pytest.raises(ValueError):
        _gen(np.full(5, 0.2), 0.03, 0.01)  # 5 != time_steps=12
    with pytest.raises(ValueError):
        _gen(0.2, np.full(5, 0.03), 0.01)


def test_negative_vol_entry_rejected():
    arr = np.full(12, 0.2)
    arr[3] = -0.1
    with pytest.raises(ValueError):
        _gen(arr, 0.03, 0.01)
