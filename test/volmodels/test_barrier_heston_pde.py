import numpy as np

from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.pde_kernel import price_barrier_heston_pde, price_european_heston_pde
from quantark.volmodels.heston.mc_kernel import price_barrier_heston_mc

HEST = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)


def test_the_kernel_refuses_the_deterministic_limit():
    # sigma -> 0 is a 1D local-vol barrier: HestonBarrierPDESolver marches it on the core
    # (test_barrier_vol_pde_engines.py::test_the_flat_deterministic_limit_is_the_black_scholes_barrier)
    import pytest
    from quantark.util.exceptions import ValidationError
    flat = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=1e-9, rho=0.0)
    with pytest.raises(ValidationError, match="HestonBarrierPDESolver"):
        price_barrier_heston_pde(100., 100., True, 1.0, flat, 0.03, 0.01, barrier=130., is_up=True, is_out=True,
                                 continuous=True, n_x=240, n_v=10, n_t=120)


def test_barrier_far_reproduces_european():
    p_eur = price_european_heston_pde(100., 100., True, 1.0, HEST, 0.03, 0.01, n_x=160, n_v=64, n_t=80)
    p_bar = price_barrier_heston_pde(100., 100., True, 1.0, HEST, 0.03, 0.01, barrier=400., is_up=True,
                                     is_out=True, continuous=True, n_x=160, n_v=64, n_t=80)
    assert abs(p_bar - p_eur) < 0.15


def test_discrete_terminal_observation_ignores_rannacher_half_step():
    p_rannacher = price_barrier_heston_pde(
        100., 100., True, 1.0, HEST, 0.03, 0.01, barrier=125.,
        is_up=True, is_out=True, continuous=False, observe_taus=[0.0],
        n_x=160, n_v=64, n_t=80, rannacher=True,
    )
    p_plain = price_barrier_heston_pde(
        100., 100., True, 1.0, HEST, 0.03, 0.01, barrier=125.,
        is_up=True, is_out=True, continuous=False, observe_taus=[0.0],
        n_x=160, n_v=64, n_t=80, rannacher=False,
    )
    assert abs(p_rannacher - p_plain) < 0.02


def test_discrete_pinned_barrier_uses_surviving_side_limit():
    p = price_barrier_heston_pde(
        100., 100., True, 1.0, HEST, 0.03, 0.01, barrier=125.,
        is_up=True, is_out=True, continuous=False,
        observe_taus=[1.0, 0.75, 0.5, 0.25, 0.0],
        n_x=160, n_v=64, n_t=80,
    )
    # Regression: zeroing the grid node pinned exactly at the discrete barrier
    # over-knocked the coarse PDE and produced about 4.29 for this case.
    assert p > 4.35


def test_mc_pde_agreement():
    dt = np.full(120, 1.0 / 120); rf = np.full(120, 0.03); cf = np.full(120, 0.01); df = np.exp(-0.03)
    p_pde = price_barrier_heston_pde(100., 100., True, 1.0, HEST, 0.03, 0.01, barrier=125., is_up=True,
                                     is_out=True, continuous=True, n_x=200, n_v=80, n_t=120)
    p_mc, se = price_barrier_heston_mc(100., 100., True, HEST, dt, rf, cf, df, barrier=125., is_up=True,
                                       is_out=True, rebate=0., pay_at_hit=False, continuous=True,
                                       num_paths=150_000, seed=17, return_stderr=True)
    # The continuous PDE truncates the domain at the barrier; MC uses Brownian-bridge survival
    # over simulated Heston paths. They should agree within MC/statistical and path-discretization
    # error; the BSM-flat analytical test covers the exact continuous-monitoring ground truth.
    assert abs(p_pde - p_mc) < max(0.5, 4 * se)
