import numpy as np
import pytest
from quantark.param import GridVolSurface, FlatRateCurve
from quantark.volmodels.black_scholes import bs_call_price
from quantark.volmodels.localvol import build_dupire_local_vol, LocalVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv import (
    BinMethod, LeverageSurface, calibrate_leverage_surface, price_european_slv_mc,
)
from quantark.util.enum.engine_enums import LeverageCalibrationMethod

_MC = LeverageCalibrationMethod.MC_BINNING


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
    surf = calibrate_leverage_surface(100.0, p_feller, lv, dt, rf, cf, eta=1.0, method=_MC,
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
        100.0, P, lv, dt, rf, cf, eta=1.0, method=_MC, num_paths=30_000, num_bins=20, seed=5,
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


# --- WS-A2: unified leverage clip band (2026-07-04 volmodels spec) ---

from quantark.volmodels.slv.slv_mc_kernel import _calibrate_mc_binning
from quantark.volmodels.slv.leverage import DEFAULT_LEVERAGE_CLIP


def _direct_flat_lv(sigma):
    return LocalVolSurface(strike_grid=np.array([1.0, 1000.0]),
                           time_grid=np.array([0.0, 5.0]),
                           lv_grid=np.full((2, 2), sigma))


def test_clip_band_default_is_ffp_band():
    from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig
    assert DEFAULT_LEVERAGE_CLIP == (0.05, 20.0)
    assert FpCalibrationConfig().leverage_clip == DEFAULT_LEVERAGE_CLIP


def test_recorded_leverage_reaches_four_with_new_band():
    # eta=0, kappa=0 -> variance frozen at v0=0.01 on every path; flat sigma_LV=0.4
    # -> true leverage 0.4/0.1 = 4.0 everywhere. Old band capped this at sqrt(10)=3.162.
    params = HestonParams(v0=0.01, kappa=0.0, theta=0.01, sigma=0.5, rho=0.0)
    dt = np.full(4, 0.25)
    surf = _calibrate_mc_binning(100.0, params, _direct_flat_lv(0.4), dt,
                                 np.zeros(4), np.zeros(4), eta=0.0,
                                 num_paths=20_000, num_bins=10, seed=7)
    assert np.allclose(surf.leverage_grid, 4.0, rtol=1e-10)
    assert surf.diagnostics is not None
    assert surf.diagnostics["method"] == "mc_binning"
    assert surf.diagnostics["n_clipped"] == 0


def test_recorded_leverage_clips_at_upper_band():
    # frozen v = 0.0025 (sqrt = 0.05), sigma_LV = 1.2 -> raw L = 24 -> clipped to 20.
    params = HestonParams(v0=0.0025, kappa=0.0, theta=0.0025, sigma=0.5, rho=0.0)
    dt = np.full(2, 0.5)
    surf = _calibrate_mc_binning(100.0, params, _direct_flat_lv(1.2), dt,
                                 np.zeros(2), np.zeros(2), eta=0.0,
                                 num_paths=20_000, num_bins=10, seed=7)
    assert np.allclose(surf.leverage_grid, 20.0, rtol=1e-12)
    assert surf.diagnostics["n_clipped"] == surf.leverage_grid.size


def test_antithetic_default_off_is_bit_identical():
    # Default (use_antithetic omitted) must equal use_antithetic=False exactly — the
    # z-stream is unchanged when antithetic is off (WS-D3/F27).
    lv = _flat_lv_surface(0.2)
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.0, 0.0
    dt, rf, cf = _const(T, 40, r, q)
    p_default = price_european_slv_mc(s0, k, True, P, lv, dt, rf, cf, disc_factor=1.0,
                                      num_paths=20_000, num_bins=20, seed=11)
    p_off = price_european_slv_mc(s0, k, True, P, lv, dt, rf, cf, disc_factor=1.0,
                                  num_paths=20_000, num_bins=20, seed=11, use_antithetic=False)
    assert p_default == p_off


def test_antithetic_reduces_stderr():
    lv = _flat_lv_surface(0.2)
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.0, 0.0
    dt, rf, cf = _const(T, 40, r, q)
    _, se_plain = price_european_slv_mc(s0, k, True, P, lv, dt, rf, cf, disc_factor=1.0,
                                        num_paths=20_000, num_bins=20, seed=11, return_stderr=True)
    _, se_anti = price_european_slv_mc(s0, k, True, P, lv, dt, rf, cf, disc_factor=1.0,
                                       num_paths=20_000, num_bins=20, seed=11,
                                       return_stderr=True, use_antithetic=True)
    assert se_anti < se_plain
