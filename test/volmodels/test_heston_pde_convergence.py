import numpy as np

from quantark.util.enum.engine_enums import ADIScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.pde_kernel import price_european_heston_pde
from quantark.volmodels.heston.analytical_kernel import heston_call_price


def _cs_price(P, r, n_t, n_x=500, n_v=250):
    return price_european_heston_pde(100.0, 100.0, True, 1.0, P, r, 0.0,
                                     n_x=n_x, n_v=n_v, n_t=n_t,
                                     scheme=ADIScheme.CRAIG_SNEYD, rannacher=True)


def _time_self_conv_orders(P, r, ladder):
    # fixed fine space; isolate the temporal error via successive-halving differences so
    # the (fixed) spatial error does not contaminate the measured time order. n_t ladders
    # are chosen per-rate to keep the temporal error above the spatial floor (at r=0 the
    # scheme is nearly time-exact, so high n_t hits the floor — a measurement artifact).
    ps = [_cs_price(P, r, n_t) for n_t in ladder]
    d = [abs(ps[i] - ps[i + 1]) for i in range(len(ps) - 1)]
    return [np.log(d[i] / d[i + 1]) / np.log(2.0) for i in range(len(d) - 1)]


def test_cs_second_order_in_time_nonzero_rate():
    # WS-C1: implicit -rU + the corrector fix -> CS is 2nd-order in time at r != 0.
    P = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)
    orders = _time_self_conv_orders(P, 0.05, (15, 30, 60, 120))
    assert min(orders) > 1.8, orders   # observed ~2.08 / 1.96


def test_cs_second_order_in_time_zero_rate():
    # The corrector fix (Y0 base) restores 2nd order independent of the reaction term.
    P = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)
    orders = _time_self_conv_orders(P, 0.0, (10, 20, 40, 80))
    assert min(orders) > 1.6, orders   # observed ~3.89 / 1.83 (coarse ladder; floor-limited)


def test_cs_cross_family_default_resolution():
    # PDE <-> analytical agreement at default resolution stays within tolerance after the flip.
    P = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)
    s0, T, r, q = 100.0, 1.0, 0.05, 0.0
    for K in (90.0, 100.0, 110.0):
        ref = heston_call_price(s0, K, T, P, r, q, method="lewis")
        p = price_european_heston_pde(s0, K, True, T, P, r, q, n_x=200, n_v=100, n_t=100,
                                      scheme=ADIScheme.CRAIG_SNEYD, rannacher=True)
        assert abs(p - ref) < 1e-1, (K, p, ref)
