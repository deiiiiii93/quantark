import numpy as np
import pytest
from quantark.param import GridVolSurface, FlatRateCurve
from quantark.volmodels.black_scholes import bs_call_price
from quantark.volmodels.localvol import build_dupire_local_vol, LocalVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv import (
    BinMethod, LeverageSurface, calibrate_leverage_surface, price_european_slv_mc,
)


def _flat_lv_surface(vol=0.2):
    # Flat implied surface -> Dupire flat local vol == vol everywhere.
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    iv = np.full((6, 9), vol)
    surf = GridVolSurface(strikes, mats, iv)
    return build_dupire_local_vol(surf, spot=100.0, rate_curve=FlatRateCurve(0.0), div_yield=lambda t: 0.0)


def _const(T, M, r, q):
    return np.full(M, T / M), np.full(M, r), np.full(M, q)


P = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.5)


def test_slv_reprices_flat_vanilla_exact_vanilla_property():
    # SLV calibrated to a flat local-vol leg must reprice the flat-vol vanilla (BS).
    lv = _flat_lv_surface(0.2)
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.0, 0.0
    dt, rf, cf = _const(T, 100, r, q)
    price, se = price_european_slv_mc(
        s0, k, True, P, lv, dt, rf, cf, disc_factor=1.0, eta=1.0,
        num_paths=80_000, num_bins=25, seed=7, return_stderr=True,
    )
    bs = bs_call_price(s0, k, T, 0.2, r, q)
    assert abs(price - bs) < 4 * se + 0.03


def test_slv_eta_zero_degenerates_to_local_vol():
    # eta=0 freezes the variance at v0; leverage L=sigma_LV/sqrt(v0) makes the effective
    # spot vol sigma_LV -> reproduces the (flat) local-vol price ~ BS.
    lv = _flat_lv_surface(0.2)
    s0, k, T = 100.0, 100.0, 1.0
    dt, rf, cf = _const(T, 100, 0.0, 0.0)
    price = price_european_slv_mc(s0, k, True, P, lv, dt, rf, cf, disc_factor=1.0,
                                  eta=0.0, num_paths=60_000, num_bins=25, seed=3)
    assert price == pytest.approx(bs_call_price(s0, k, T, 0.2, 0.0, 0.0), abs=0.08)


def test_calibrate_leverage_surface_flat_is_near_one_over_sqrt_v():
    # Feller-satisfied (2*k*theta=0.12 > sigma^2=0.04) so E[v|S]~theta and the calibrated
    # leverage L = sigma_LV/sqrt(E[v]) ~ 0.2/sqrt(0.04) = 1.0.
    lv = _flat_lv_surface(0.2)
    p_feller = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.2, rho=-0.5)
    dt, rf, cf = _const(1.0, 80, 0.0, 0.0)
    surf = calibrate_leverage_surface(100.0, p_feller, lv, dt, rf, cf, eta=1.0,
                                      num_paths=60_000, num_bins=20, seed=5)
    assert isinstance(surf, LeverageSurface)
    L_atm = float(surf.leverage(100.0, 0.5))
    assert L_atm == pytest.approx(1.0, rel=0.2)


def test_slv_validation():
    from quantark.util.exceptions import ValidationError
    lv = _flat_lv_surface(0.2)
    dt, rf, cf = _const(1.0, 10, 0.0, 0.0)
    with pytest.raises(ValidationError):
        price_european_slv_mc(-1.0, 100.0, True, P, lv, dt, rf, cf, disc_factor=1.0, num_paths=100)


def test_slv_mc_accepts_precomputed_leverage_surface_deterministically():
    lv = _flat_lv_surface(0.2)
    dt, rf, cf = _const(1.0, 60, 0.0, 0.0)
    leverage = calibrate_leverage_surface(
        100.0, P, lv, dt, rf, cf, eta=1.0, num_paths=30_000, num_bins=20, seed=5,
    )
    kwargs = dict(
        s0=100.0,
        strike=100.0,
        is_call=True,
        params=P,
        lv_surface=lv,
        step_dt=dt,
        r_fwd=rf,
        carry_fwd=cf,
        disc_factor=1.0,
        eta=1.0,
        num_paths=30_000,
        num_bins=20,
        seed=11,
        leverage_surface=leverage,
    )
    first = price_european_slv_mc(**kwargs)
    second = price_european_slv_mc(**kwargs)
    assert first == pytest.approx(second, abs=0.0)
    assert first > 0.0
