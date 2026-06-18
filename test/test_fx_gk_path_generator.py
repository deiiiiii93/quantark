"""Forward-consistent Garman-Kohlhagen path generator."""

import math

import numpy as np
import pytest

from quantark.asset.fx.process.fx_gk_path_generator import (
    FxGKPathGenerator,
    FxGKPathGeneratorQMC,
)
from quantark.montecarlo.qmc_sobol import PseudoRandomNormalGenerator

SPOT, RD, RF, SIGMA = 1.20, 0.05, 0.03, 0.10


def flat_forward(t):
    return SPOT * math.exp((RD - RF) * t)


def make_gen(times, num_paths=200_000, seed=7, forward_fn=flat_forward, sigma=SIGMA, **kw):
    return FxGKPathGenerator(
        spot=SPOT, sigma=sigma, forward_fn=forward_fn,
        times=np.asarray(times, dtype=float), num_paths=num_paths,
        random_stream=PseudoRandomNormalGenerator(seed=seed), **kw,
    )


def test_flat_curve_drift_matches_constant():
    times = np.array([0.25, 0.5, 0.75, 1.0])
    gen = make_gen(times, num_paths=10)
    dt = np.diff(np.concatenate([[0.0], times]))
    expected = (RD - RF - 0.5 * SIGMA ** 2) * dt
    np.testing.assert_allclose(gen._drift_term, expected, rtol=1e-12, atol=1e-14)


def test_mean_matches_forward_at_grid_times():
    times = np.array([0.25, 0.5, 1.0])
    gen = make_gen(times, num_paths=400_000)
    paths, _ = gen.generate_paths()
    for i, t in enumerate(times):
        assert paths[:, i + 1].mean() == pytest.approx(flat_forward(t), rel=2e-3)


def test_log_variance_matches_sigma2_t():
    times = np.array([0.5, 1.0])
    gen = make_gen(times, num_paths=400_000)
    paths, _ = gen.generate_paths()
    for i, t in enumerate(times):
        var = np.log(paths[:, i + 1] / SPOT).var()
        assert var == pytest.approx(SIGMA ** 2 * t, rel=3e-3)


def test_non_flat_curve_mean_matches_forward():
    # Humped forward via an arbitrary positive forward curve.
    def fwd(t):
        return SPOT * math.exp((0.06 - 0.01) * t) * (1.0 + 0.02 * math.sin(t))

    times = np.array([0.25, 0.5, 1.0])
    gen = make_gen(times, num_paths=400_000, forward_fn=fwd)
    paths, _ = gen.generate_paths()
    for i, t in enumerate(times):
        assert paths[:, i + 1].mean() == pytest.approx(fwd(t), rel=3e-3)


def test_deterministic_fixed_seed():
    times = np.array([0.5, 1.0])
    p1, _ = make_gen(times, num_paths=1000, seed=11).generate_paths()
    p2, _ = make_gen(times, num_paths=1000, seed=11).generate_paths()
    np.testing.assert_array_equal(p1, p2)


def test_zero_vol_is_deterministic_forward():
    times = np.array([0.25, 0.5, 1.0])
    gen = make_gen(times, num_paths=16, sigma=0.0)
    paths, _ = gen.generate_paths()
    for i, t in enumerate(times):
        np.testing.assert_allclose(paths[:, i + 1], flat_forward(t), rtol=1e-12)


def test_qmc_subclass_runs():
    times = np.array([0.5, 1.0])
    gen = FxGKPathGeneratorQMC(
        spot=SPOT, sigma=SIGMA, forward_fn=flat_forward, times=times, num_paths=1024,
    )
    paths, _ = gen.generate_paths()
    assert paths.shape == (1024, 3)
    assert paths[:, 2].mean() == pytest.approx(flat_forward(1.0), rel=5e-3)
