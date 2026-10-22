"""WS-C3: degenerate v=0 boundary for the Heston/SLV ADI core."""
import numpy as np
import pytest

from quantark.util.enum.engine_enums import ADIScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.pde_kernel import price_european_heston_pde
from quantark.volmodels.heston.analytical_kernel import heston_call_price


def _feller_violated():
    # 2*kappa*theta = 2*0.5*0.04 = 0.04 << sigma^2 = 0.81  (strongly Feller-violated)
    return HestonParams(kappa=0.5, theta=0.04, sigma=0.9, rho=-0.5, v0=0.04)


def test_default_is_neumann():
    import inspect
    assert inspect.signature(price_european_heston_pde).parameters["v0_boundary"].default == "neumann"


def test_degenerate_boundary_reduces_feller_violated_error():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    params = _feller_violated()
    ref = heston_call_price(s0, k, T, params, r, q)
    common = dict(n_x=200, n_v=80, n_t=100, scheme=ADIScheme.CRAIG_SNEYD)
    p_neu = price_european_heston_pde(s0, k, True, T, params, r, q,
                                      v0_boundary="neumann", **common)
    p_deg = price_european_heston_pde(s0, k, True, T, params, r, q,
                                      v0_boundary="degenerate_pde", **common)
    assert abs(p_deg - ref) <= abs(p_neu - ref) + 1e-6  # no worse; expected better


def test_feller_satisfied_case_essentially_unchanged():
    # 2*kappa*theta = 2*3*0.04 = 0.24 > sigma^2 = 0.04 (Feller satisfied): boundary inert.
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    params = HestonParams(kappa=3.0, theta=0.04, sigma=0.2, rho=-0.5, v0=0.04)
    common = dict(n_x=200, n_v=80, n_t=100, scheme=ADIScheme.CRAIG_SNEYD)
    p_neu = price_european_heston_pde(s0, k, True, T, params, r, q,
                                      v0_boundary="neumann", **common)
    p_deg = price_european_heston_pde(s0, k, True, T, params, r, q,
                                      v0_boundary="degenerate_pde", **common)
    assert abs(p_deg - p_neu) < 5e-3


def test_invalid_v0_boundary_raises():
    with pytest.raises(Exception):
        price_european_heston_pde(100.0, 100.0, True, 1.0, _feller_violated(), 0.03, 0.0,
                                  v0_boundary="bogus")
