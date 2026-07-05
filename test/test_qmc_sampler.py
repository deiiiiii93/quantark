"""WS-D7: opt-in QMC sampler on the volmodels MC kernels."""
import numpy as np
import pytest

from quantark.montecarlo.qmc_sobol import PseudoRandomNormalGenerator, SobolNormalGenerator
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.mc_kernel import price_european_heston_mc
from quantark.volmodels.localvol.mc_kernel import price_european_lv_mc
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.black_scholes import bs_call_price


def _flat_surface(vol=0.2):
    # Constant local-vol surface (no LocalVolSurface.flat helper exists).
    return LocalVolSurface(
        strike_grid=np.array([1.0, 1.0e6]),
        time_grid=np.array([0.0, 100.0]),
        lv_grid=np.full((2, 2), vol),
    )


def test_generators_uniform_shape_and_range():
    for gen in (PseudoRandomNormalGenerator(seed=1), SobolNormalGenerator(base_seed=1)):
        u = gen.uniform(64, 3)
        assert u.shape == (64, 3)
        assert np.all(u > 0.0) and np.all(u < 1.0)


def test_sampler_none_is_bit_identical_pseudo():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0
    params = HestonParams(kappa=1.5, theta=0.04, sigma=0.5, rho=-0.6, v0=0.04)
    dt = np.full(8, T / 8); rf = np.full(8, r); cf = np.full(8, q); df = np.exp(-r * T)
    p_default = price_european_heston_mc(s0, k, True, params, dt, rf, cf, df,
                                         scheme=HestonMCScheme.QUADEXP, num_paths=8192, seed=3)
    p_explicit_none = price_european_heston_mc(s0, k, True, params, dt, rf, cf, df,
                                               scheme=HestonMCScheme.QUADEXP, num_paths=8192,
                                               seed=3, sampler=None)
    assert p_default == p_explicit_none  # exact


def test_sampler_and_antithetic_are_mutually_exclusive():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0
    params = HestonParams(kappa=1.5, theta=0.04, sigma=0.5, rho=-0.6, v0=0.04)
    dt = np.full(4, T / 4); rf = np.full(4, r); cf = np.full(4, q); df = np.exp(-r * T)
    with pytest.raises(Exception):
        price_european_heston_mc(s0, k, True, params, dt, rf, cf, df, num_paths=1024,
                                 use_antithetic=True, sampler=SobolNormalGenerator(base_seed=1))


def test_heston_qmc_reprices_within_mc_error():
    from quantark.volmodels.heston.analytical_kernel import heston_call_price
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    dt = np.full(8, T / 8); rf = np.full(8, r); cf = np.full(8, q); df = np.exp(-r * T)
    analytic = heston_call_price(s0, k, T, params, r, q)
    price = price_european_heston_mc(s0, k, True, params, dt, rf, cf, df,
                                     scheme=HestonMCScheme.QUADEXP, num_paths=16384,
                                     sampler=SobolNormalGenerator(base_seed=5))
    assert abs(price - analytic) < 0.25


def test_qmc_rmse_beats_pseudo_on_lv_european():
    # Flat LV: QMC should reach a lower RMSE-vs-analytical than pseudo at equal path counts.
    s0, k, r, q, vol = 100.0, 100.0, 0.03, 0.0, 0.2
    surface = _flat_surface(vol)
    T = 1.0
    dt = np.full(4, T / 4); rf = np.full(4, r); cf = np.full(4, q); df = np.exp(-r * T)
    analytic = bs_call_price(s0, k, T, vol, r, q)
    n = 4096
    err_pseudo, err_qmc = [], []
    for b in range(16):
        p_ps = price_european_lv_mc(s0, k, True, surface, dt, rf, cf, df,
                                    num_paths=n, seed=100 + b)
        p_q = price_european_lv_mc(s0, k, True, surface, dt, rf, cf, df,
                                   num_paths=n, sampler=SobolNormalGenerator(base_seed=100 + b))
        err_pseudo.append((p_ps - analytic) ** 2)
        err_qmc.append((p_q - analytic) ** 2)
    rmse_pseudo = np.sqrt(np.mean(err_pseudo))
    rmse_qmc = np.sqrt(np.mean(err_qmc))
    assert rmse_qmc < rmse_pseudo
