"""The sigma -> 0 limit of Heston and Heston-SLV as a local-vol surface: v(t) = theta + (v0 - theta) e^(-kappa t) is
deterministic, so the spot diffuses with sigma_loc(S, t) = L(S, t) sqrt(v(t)) (L = 1 under Heston)."""
import numpy as np
import pytest

from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.deterministic_limit import DeterministicVarianceLocalVol
from quantark.volmodels.slv.leverage import LeverageSurface

PARAMS = HestonParams(v0=0.09, kappa=1.5, theta=0.04, sigma=0.0, rho=-0.5)
SPOTS = np.array([60.0, 100.0, 140.0])


def _skew_leverage():
    ks = np.array([50.0, 80.0, 100.0, 120.0, 160.0])
    rows = np.array([[1.3, 1.1, 1.0, 0.9, 0.8], [1.2, 1.05, 1.0, 0.95, 0.85]])
    return LeverageSurface(time_grid=np.array([0.0, 1.0]), strike_grid=ks, leverage_grid=rows)


@pytest.mark.parametrize("t", [0.0, 0.3, 1.7])
def test_heston_local_vol_is_the_deterministic_variance_path(t):
    expected = np.sqrt(PARAMS.theta + (PARAMS.v0 - PARAMS.theta) * np.exp(-PARAMS.kappa * t))
    np.testing.assert_allclose(DeterministicVarianceLocalVol(PARAMS).local_vol(SPOTS, t), np.full(3, expected),
                               rtol=1e-15)


def test_without_mean_reversion_the_variance_stays_at_v0():
    frozen = HestonParams(v0=0.09, kappa=0.0, theta=0.04, sigma=0.0, rho=0.0)
    np.testing.assert_array_equal(DeterministicVarianceLocalVol(frozen).local_vol(SPOTS, 2.0), np.full(3, 0.3))


def test_slv_scales_the_variance_path_by_the_leverage_at_each_node():
    lev = _skew_leverage()
    t = 0.4
    vol = np.sqrt(PARAMS.theta + (PARAMS.v0 - PARAMS.theta) * np.exp(-PARAMS.kappa * t))
    np.testing.assert_array_equal(DeterministicVarianceLocalVol(PARAMS, lev).local_vol(SPOTS, t),
                                  lev.leverage(SPOTS, t) * vol)


def test_a_unit_leverage_is_the_heston_surface_bit_for_bit():
    ks = np.array([50.0, 100.0, 200.0])
    unit = LeverageSurface(time_grid=np.array([0.0, 1.0]), strike_grid=ks, leverage_grid=np.ones((2, 3)))
    for t in (0.0, 0.25, 3.0):
        np.testing.assert_array_equal(DeterministicVarianceLocalVol(PARAMS, unit).local_vol(SPOTS, t),
                                      DeterministicVarianceLocalVol(PARAMS).local_vol(SPOTS, t))


def test_the_surface_fingerprints_so_sessions_can_key_on_it():
    a, b = DeterministicVarianceLocalVol(PARAMS), DeterministicVarianceLocalVol(PARAMS, _skew_leverage())
    assert try_fingerprint(a) is not None and try_fingerprint(b) is not None
    assert try_fingerprint(a) != try_fingerprint(b)
