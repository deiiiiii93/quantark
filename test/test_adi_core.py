import numpy as np
import pytest

from quantark.util.enum.engine_enums import ADIScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.volmodels.slv.leverage import LeverageSurface

P = HestonParams(v0=0.04, kappa=1.5, theta=0.05, sigma=0.4, rho=-0.6)


def _unit_leverage_surface(s0, T):
    # L(S,t) == 1 everywhere on a coarse grid (flat extrapolation fills the rest).
    strikes = s0 * np.exp(np.linspace(-1.0, 1.0, 5))
    times = np.linspace(0.0, T, 4)
    return LeverageSurface(time_grid=times, strike_grid=strikes,
                           leverage_grid=np.ones((4, 5)))


@pytest.mark.parametrize("scheme", [ADIScheme.DOUGLAS, ADIScheme.CRAIG_SNEYD])
def test_heston_config_equals_slv_config_unit_leverage(scheme):
    s0, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    heston = HestonSLVADICore(s0, K, T, r, q, P, 60, 40, 30, leverage=None, eta=1.0)
    slv = HestonSLVADICore(s0, K, T, r, q, P, 60, 40, 30,
                           leverage=_unit_leverage_surface(s0, T), eta=1.0)
    Uh = heston.solve(True, scheme, 0.5, True)
    Us = slv.solve(True, scheme, 0.5, True)
    assert np.max(np.abs(Uh - Us)) < 1e-13 * max(1.0, np.max(np.abs(Uh)))
    ph = heston.interpolate(Uh, np.log(s0), P.v0)
    ps = slv.interpolate(Us, np.log(s0), P.v0)
    assert abs(ph - ps) < 1e-13 * max(1.0, abs(ph))


def test_grid_spot_pinning_available_on_both_configs():
    # Both paths accept grid_spot (SLV gains it for free — F24).
    s0, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    core = HestonSLVADICore(s0, K, T, r, q, P, 40, 30, 20,
                            leverage=_unit_leverage_surface(s0, T), grid_spot=95.0)
    assert core.S_grid[0] < 95.0 < core.S_grid[-1]
