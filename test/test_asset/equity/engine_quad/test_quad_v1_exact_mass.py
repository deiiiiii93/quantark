"""QUAD V1 exact-mass diffusion kernel (0.12.1).

V1 samples the Gaussian transition density at lattice offsets and sums the samples with trapezoid weights. By Poisson
summation that lattice sum is the exact integral times Theta(r, phi) = 1 + 2 sum_n e^{-2 pi^2 n^2 r^2} cos(2 pi n phi)
(r = s/h, phi the kernel centre's offset from the lattice, in cells). A step's mass error multiplies the value every
step, so it compounds; dividing each step by Theta makes every step's mass exact.
"""
import math

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.quad_math import lattice_mass_factor


def _brute_theta(r: float, phi: float) -> float:
    """The lattice sum itself, h sum_k g(k h - mu) / integral, by direct summation (|k| <= 2000 cells)."""
    k = np.arange(-2000, 2001, dtype=float)
    return float(np.sum(np.exp(-0.5 * ((k - phi) / r) ** 2)) / (r * math.sqrt(2.0 * math.pi)))


def _tau_for(r: float, h: float) -> float:
    return 0.5 * (r * h) ** 2                   # s = sqrt(2 tau) = r h


@pytest.mark.parametrize("r", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("phi", [0.0, 0.1, 0.25, 0.5, 0.77])
def test_factor_is_the_lattice_sum(r, phi):
    h = 0.0031
    got = lattice_mass_factor(_tau_for(r, h), h, phi)
    assert got == pytest.approx(_brute_theta(r, phi), rel=1e-13, abs=0.0)


@pytest.mark.parametrize("r", [1.40, 1.6, 2.5, 13.0])
@pytest.mark.parametrize("phi", [0.0, 0.1, 0.25, 0.5, 0.77])
def test_factor_is_exactly_one_where_the_alias_is_below_rounding(r, phi):
    h = 0.0031
    assert lattice_mass_factor(_tau_for(r, h), h, phi) == 1.0


@pytest.mark.parametrize("r", [0.72, 1.0, 1.3])
@pytest.mark.parametrize("phi", [0.0, 0.1, 0.5, 0.9])
def test_factor_differs_from_one_where_the_alias_survives(r, phi):
    assert abs(math.cos(2.0 * math.pi * phi)) >= 0.5
    h = 0.0031
    assert lattice_mass_factor(_tau_for(r, h), h, phi) != 1.0


def test_factor_depends_only_on_the_fractional_offset():
    h, tau = 0.0031, _tau_for(0.8, 0.0031)
    assert lattice_mass_factor(tau, h, 0.3) == lattice_mass_factor(tau, h, 7.3) == lattice_mass_factor(tau, h, -0.7)
