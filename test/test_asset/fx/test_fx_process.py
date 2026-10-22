"""
Tests for the Garman-Kohlhagen stochastic process.
"""

import math

import numpy as np
import pytest

from quantark.asset.fx.process import GarmanKohlhagenProcess
from quantark.util.exceptions import ValidationError

SPOT = 1.20
VOL = 0.10
R_DOM = 0.05
R_FOR = 0.03
T = 1.0


def make_process(**overrides):
    kwargs = dict(
        spot=SPOT,
        volatility=VOL,
        domestic_rate=R_DOM,
        foreign_rate=R_FOR,
    )
    kwargs.update(overrides)
    return GarmanKohlhagenProcess(**kwargs)


class TestValidation:
    def test_spot_positive(self):
        with pytest.raises(ValidationError):
            make_process(spot=-1.0)

    def test_vol_positive(self):
        with pytest.raises(ValidationError):
            make_process(volatility=0.0)


class TestProperties:
    def test_drift(self):
        assert make_process().drift == pytest.approx(R_DOM - R_FOR)

    def test_forward(self):
        process = make_process()
        assert process.get_forward_rate(T) == pytest.approx(
            SPOT * math.exp((R_DOM - R_FOR) * T)
        )


class TestPathGeneration:
    def test_shape_and_initial_value(self):
        paths = make_process().generate_paths(
            time_steps=50, num_paths=200, maturity=T, seed=42
        )
        assert paths.shape == (200, 51)
        assert np.allclose(paths[:, 0], SPOT)

    def test_time_grid(self):
        paths, grid = make_process().generate_paths(
            time_steps=4, num_paths=10, maturity=T, seed=42, return_time_grid=True
        )
        assert paths.shape == (10, 5)
        assert np.allclose(grid, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_martingale_property(self):
        # E[S_T] = S0 * exp((r_d - r_f) * T) under the domestic measure
        paths = make_process().generate_paths(
            time_steps=12, num_paths=200_000, maturity=T, seed=7, antithetic=True
        )
        expected = SPOT * math.exp((R_DOM - R_FOR) * T)
        assert paths[:, -1].mean() == pytest.approx(expected, rel=2e-3)

    def test_seed_reproducibility(self):
        process = make_process()
        a = process.generate_paths(time_steps=10, num_paths=100, maturity=T, seed=1)
        b = process.generate_paths(time_steps=10, num_paths=100, maturity=T, seed=1)
        assert np.array_equal(a, b)

    def test_antithetic_pairs(self):
        # With antithetic sampling, log-returns of path i and i + n/2 are mirrored
        process = make_process()
        paths = process.generate_paths(
            time_steps=5, num_paths=100, maturity=T, seed=3, antithetic=True
        )
        log_ret = np.log(paths[:, -1] / SPOT)
        drift_total = (R_DOM - R_FOR - 0.5 * VOL**2) * T
        z = log_ret - drift_total
        assert np.allclose(z[:50], -z[50:], atol=1e-12)

    def test_qmc_runs(self):
        paths = make_process().generate_paths(
            time_steps=8, num_paths=1024, maturity=T, seed=11, use_qmc=True
        )
        assert paths.shape == (1024, 9)
        expected = SPOT * math.exp((R_DOM - R_FOR) * T)
        assert paths[:, -1].mean() == pytest.approx(expected, rel=5e-3)

    def test_invalid_steps_rejected(self):
        with pytest.raises(ValidationError):
            make_process().generate_paths(
                time_steps=0, num_paths=10, maturity=T
            )

    def test_invalid_maturity_rejected(self):
        with pytest.raises(ValidationError):
            make_process().generate_paths(
                time_steps=10, num_paths=10, maturity=-1.0
            )
