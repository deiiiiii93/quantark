import numpy as np
import pytest

from quantark.volmodels.heston import (
    HestonParams, price_european_gatheral, price_european_lewis, price_european_weber,
)
from quantark.volmodels.heston.analytical_kernel import _cf_core

P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)

# Pins captured from the pre-refactor Lewis phi at u = [0.15, 0.8, 3.2, 11.0],
# (s0, k, T, r, q) = (100, 95, 1.3, 0.03, 0.01).
LEWIS_PHI = [
    (0.9931763549452637 - 6.718014650260317e-05j),
    (0.9779700172736805 - 0.0011453835484099121j),
    (0.7792544439773821 - 0.03944802579752448j),
    (0.06479668366623559 - 0.12232408299514357j),
]


def test_cf_core_reproduces_lewis_phi_to_1e13():
    T = 1.3
    us = np.array([0.15, 0.8, 3.2, 11.0])
    v = P.sigma ** 2
    kc = us + 0.5j
    beta = P.kappa + 1j * P.rho * P.sigma * kc
    d = np.sqrt(beta ** 2 + v * kc * (kc - 1j))
    A, B = _cf_core(beta, d, T, P)
    phi = np.exp(A * P.theta + B * P.v0)
    ref = np.array(LEWIS_PHI, dtype=complex)
    assert np.max(np.abs(phi - ref) / np.abs(ref)) < 1e-13


def test_three_methods_still_agree_after_refactor():
    s0, T, r, q = 100.0, 0.75, 0.02, 0.0
    for k in (80.0, 100.0, 120.0):
        g = price_european_gatheral(s0, r, q, P, k, T)
        le = price_european_lewis(s0, r, q, P, k, T)
        we = price_european_weber(s0, r, q, P, k, T)
        assert le == pytest.approx(g, abs=2e-3)
        assert we == pytest.approx(g, abs=2e-3)
